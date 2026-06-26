#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WAN-like video continuation training (single GPU, no g_phi).
- Frozen VAE (WanVAE wrapper)
- Train DiT to predict eps on FUTURE latent part given CLEAN conditional latent part.
- Visualization: random K samples from val_set each time -> DDIM multi-step sampling -> export MP4 montage:
      top row: GT videos (K clips concatenated horizontally)
      bottom : Pred videos (K clips concatenated horizontally)
- Similarity metrics (future frames only): MSE / PSNR / SSIM
- Checkpoint saving is atomic (tmp -> rename) to avoid corruption.

Outputs:
  save_dir/
    args.json
    latest.pt
    best.pt
    metrics.csv
    metrics.jsonl
    vis/
      step00001000_k04_t0800_dd25_montage.mp4
"""

import os
import sys
import json
import math
import time
import csv
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
# Video writing is implemented locally in this script.

# 或者直接在训练脚本中定义（如果不想单独文件）
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- WAN imports ----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "backbone", "Wan2.1"))
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

from wan.modules.vae import WanVAE
from worldmodel_dit_mp import create_worldmodel_mp_auto, WorldModelDiTMPWithCheckpoint
from cholec80_dataset import Cholec80

# video writer
try:
    import torchvision
    _HAS_TVIO = True
except Exception:
    torchvision = None
    _HAS_TVIO = False

try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


# ---------------- Utils ----------------
def now_ts():
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_t_cond_latent(Tz: int, cond_frames: int, total_frames: int) -> int:
    """Map pixel-time cond_frames -> latent-time t_cond"""
    t_cond = int(round(Tz * cond_frames / total_frames))
    return max(1, min(Tz - 1, t_cond))


# ---------------- Dataset ----------------
class VideoDataset(Dataset):
    def __init__(self, seqs, video_frames: int = 13, stride: int = None):
        self.seqs = seqs
        self.video_frames = video_frames
        self.indices = []
        if stride is None:
            stride = max(1, video_frames // 2)

        for si, seq in enumerate(self.seqs):
            n = len(seq.image_files)
            if n < self.video_frames:
                continue
            for st in range(0, n - self.video_frames + 1, stride):
                self.indices.append((si, st))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        si, st = self.indices[idx]
        seq = self.seqs[si]
        frames = []
        for k in range(self.video_frames):
            img, _ = seq[st + k]
            frames.append(img)
        # [C, T, H, W] in [0,1]
        return torch.stack(frames, dim=1)


# ---------------- Diffusion (alpha_bar schedule) ----------------
def make_cosine_schedule(num_steps: int, device: torch.device) -> torch.Tensor:
    """
    return alpha_bar[t] for t in [0..num_steps-1]
    """
    s = 0.008
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, device=device) / num_steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[:-1].clamp(1e-5, 1.0)  # [num_steps]


def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """
    z_t = sqrt(a)*z0 + sqrt(1-a)*eps, a = alpha_bar[t]
    z0: [B,C,T,H,W], t:[B]
    """
    B = z0.shape[0]
    a = alphas[t].view(B, 1, 1, 1, 1)
    return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * noise


def predict_x0_from_eps(z_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    B = z_t.shape[0]
    a = alphas[t].view(B, 1, 1, 1, 1)
    return (z_t - torch.sqrt(1 - a) * eps) / torch.sqrt(a)


# ---------------- VAE encode/decode ----------------
@torch.no_grad()
def encode_clip_to_latent(vae: WanVAE, x_0_1: torch.Tensor) -> torch.Tensor:
    """
    x_0_1: [B,C,T,H,W] in [0,1]
    return: z0 [B,16,Tz,Hz,Wz]
    """
    B = x_0_1.shape[0]
    x = x_0_1 * 2.0 - 1.0
    z_list = vae.encode([x[b] for b in range(B)])
    return torch.stack(z_list, dim=0)


@torch.no_grad()
def decode_latent_to_video(vae: WanVAE, z: torch.Tensor) -> torch.Tensor:
    """
    z: [B,16,Tz,Hz,Wz]
    return x_0_1: [B,3,T,H,W] in [0,1]
    """
    B = z.shape[0]
    x_list = vae.decode([z[b] for b in range(B)])  # each: [3,T,H,W] in [-1,1]
    x = torch.stack(x_list, dim=0)
    return ((x + 1) / 2).clamp(0, 1)


# ---------------- Video writing ----------------
@torch.no_grad()
@torch.no_grad()
def write_mp4(video_TCHW_0_1: torch.Tensor, path: str, fps: int = 8):
    """
    修复版的视频写入函数
    """
    if video_TCHW_0_1.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(video_TCHW_0_1.shape)}")
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # 转换为 numpy [T, H, W, C]
    v = (video_TCHW_0_1.clamp(0, 1) * 255.0).to(torch.uint8)
    v = v.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    
    T, H, W, C = v.shape
    
    # ⭐ 关键修复1: 确保尺寸是偶数
    if H % 2 != 0:
        new_H = H + 1
        new_v = np.zeros((T, new_H, W, C), dtype=v.dtype)
        new_v[:, :H, :, :] = v
        v = new_v
        H = new_H
    
    if W % 2 != 0:
        new_W = W + 1
        new_v = np.zeros((T, H, new_W, C), dtype=v.dtype)
        new_v[:, :, :W, :] = v
        v = new_v
        W = new_W
    
    # ⭐ 关键修复2: 使用 OpenCV (最稳定)
    try:
        # MP4V 编码器（兼容性最好）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, float(fps), (W, H))
        
        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")
        
        for frame in v:
            # OpenCV 需要 BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        print(f"[Video] ✅ Saved: {path}", flush=True)
        return
        
    except Exception as e:
        print(f"[Video] OpenCV failed: {e}, trying fallback...", flush=True)
    
    # ⭐ 备选方案: 保存为图片序列
    try:
        from PIL import Image
        frame_dir = path.replace('.mp4', '_frames')
        os.makedirs(frame_dir, exist_ok=True)
        
        for i, frame in enumerate(v):
            Image.fromarray(frame).save(f"{frame_dir}/frame_{i:04d}.png")
        
        print(f"[Video] ⚠️  Saved as frames: {frame_dir}", flush=True)
        print(f"[Video] Convert with: ffmpeg -r {fps} -i {frame_dir}/frame_%04d.png "
              f"-c:v libx264 -pix_fmt yuv420p {path}", flush=True)
        
    except Exception as e:
        print(f"[Video] ❌ All methods failed: {e}", flush=True)



@torch.no_grad()
def make_montage_video(gt_BCTHW: torch.Tensor, pr_BCTHW: torch.Tensor) -> torch.Tensor:
    """
    Build montage:
      top row: GT (K clips horizontally concatenated)
      bottom : Pred (K clips horizontally concatenated)

    gt_BCTHW, pr_BCTHW: [K,3,T,H,W]
    return: [T,3,2H,K*W]
    """
    assert gt_BCTHW.shape == pr_BCTHW.shape
    K, C, T, H, W = gt_BCTHW.shape
    # [K,T,C,H,W]
    gt = gt_BCTHW.permute(0, 2, 1, 3, 4).contiguous()
    pr = pr_BCTHW.permute(0, 2, 1, 3, 4).contiguous()

    frames = []
    for t in range(T):
        # each: [C,H,W]
        gt_row = torch.cat([gt[k, t] for k in range(K)], dim=2)  # concat width
        pr_row = torch.cat([pr[k, t] for k in range(K)], dim=2)  # concat width
        frame = torch.cat([gt_row, pr_row], dim=1)               # concat height
        frames.append(frame)
    return torch.stack(frames, dim=0)  # [T,C,2H,K*W]


# ---------------- Metrics: MSE / PSNR / SSIM ----------------
def psnr_from_mse(mse: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=eps))


def _gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
              C1: float = 0.01**2, C2: float = 0.03**2):
    """
    x,y: [N,C,H,W] in [0,1]
    returns mean SSIM over N
    """
    device = x.device
    C = x.shape[1]
    g = _gaussian_window(window_size, sigma, device=device)
    g2d = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    g2d = g2d.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    padding = window_size // 2
    mu_x = F.conv2d(x, g2d, padding=padding, groups=C)
    mu_y = F.conv2d(y, g2d, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, g2d, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, g2d, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, g2d, padding=padding, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


@torch.no_grad()
def compute_video_metrics_one(gt_1CTHW: torch.Tensor, pr_1CTHW: torch.Tensor, cond_frames: int):
    """
    gt,pr: [1,3,T,H,W] in [0,1]
    metrics computed on future frames only: [cond_frames .. T-1]
    """
    gt_f = gt_1CTHW[:, :, cond_frames:]
    pr_f = pr_1CTHW[:, :, cond_frames:]
    mse = F.mse_loss(pr_f, gt_f)
    psnr = psnr_from_mse(mse)

    # SSIM frame-wise then mean
    B, C, Tf, H, W = gt_f.shape
    gt_frames = gt_f.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    pr_frames = pr_f.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    ssim = ssim_torch(pr_frames, gt_frames)

    return float(mse.item()), float(psnr.item()), float(ssim.item())


# ---------------- DDIM sampling for continuation ----------------
@torch.no_grad()
def ddim_sample_continue(
    dit,
    z0: torch.Tensor,             # [B,16,Tz,Hz,Wz] clean
    t_cond: int,
    alphas: torch.Tensor,         # alpha_bar schedule [num_steps]
    t_start: int,
    num_ddim_steps: int = 25,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    """
    Deterministic DDIM (eta=0), update the WHOLE video latent per step (NOT frame-by-frame).
    Keep cond part clean at every step.
    """
    device = z0.device
    B = z0.shape[0]
    num_steps = alphas.shape[0]
    t_start = int(np.clip(t_start, 1, num_steps - 1))

    # init z_t at t_start: cond clean + future ~ N(0,1)
    z = torch.randn_like(z0)
    z[:, :, :t_cond] = z0[:, :, :t_cond]

    # decreasing timestep list
    ts = np.linspace(t_start, 0, num_ddim_steps, dtype=np.int64)
    ts = np.unique(ts)[::-1]
    if ts[-1] != 0:
        ts = np.concatenate([ts, np.array([0], dtype=np.int64)], axis=0)

    for i in range(len(ts) - 1):
        t_cur = int(ts[i])
        t_prev = int(ts[i + 1])

        t_vec = torch.full((B,), t_cur, device=device, dtype=torch.long)
        a_cur = alphas[t_vec].view(B, 1, 1, 1, 1)
        a_prev = alphas[torch.full((B,), t_prev, device=device, dtype=torch.long)].view(B, 1, 1, 1, 1)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            eps_pred = dit(z, t_vec, t_cond)

        eps_pred = eps_pred.float()
        z_f = z.float()
        x0 = (z_f - torch.sqrt(1 - a_cur) * eps_pred) / torch.sqrt(a_cur)
        z_prev = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps_pred

        z = z_prev
        z[:, :, :t_cond] = z0[:, :, :t_cond]

    return z


# ---------------- Logger ----------------
class MetricsLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / "metrics.csv"
        self.jsonl_path = self.save_dir / "metrics.jsonl"

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time", "type", "epoch", "step",
                    "loss_eps", "loss_x0", "loss_total",
                    "mse_future", "psnr_future", "ssim_future",
                    "vis_idxs",
                    "lr"
                ])

    def log(self, row: dict):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("time"), row.get("type"), row.get("epoch"), row.get("step"),
                row.get("loss_eps"), row.get("loss_x0"), row.get("loss_total"),
                row.get("mse_future"), row.get("psnr_future"), row.get("ssim_future"),
                row.get("vis_idxs"),
                row.get("lr"),
            ])


# ---------------- LR schedule ----------------
class WarmupCosine:
    def __init__(self, opt, warmup_steps, total_steps, base_lr, min_lr):
        self.opt = opt
        self.warmup_steps = int(warmup_steps)
        self.total_steps = max(1, int(total_steps))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            denom = max(1, (self.total_steps - self.warmup_steps))
            progress = (self.step_count - self.warmup_steps) / denom
            progress = float(np.clip(progress, 0.0, 1.0))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ---------------- Build DiT ----------------
def build_dit_single(model_size: str, use_checkpoint: bool, dit_init: str):
    device = torch.device("cuda:0")

    if use_checkpoint:
        configs = {
            "base": dict(dim=768, depth=12, num_heads=12, ffn_hidden=3072),
            "large": dict(dim=1024, depth=24, num_heads=16, ffn_hidden=4096),
            "xlarge": dict(dim=1536, depth=30, num_heads=24, ffn_hidden=8960),
        }
        dit = WorldModelDiTMPWithCheckpoint(
            latent_channels=16,
            **configs[model_size],
            patch_size=(1, 2, 2),
            devices=["cuda:0"],
        )
    else:
        dit = create_worldmodel_mp_auto(model_size=model_size, max_gpus=1)

    if dit_init and str(dit_init).strip() != "" and os.path.isfile(dit_init):
        ckpt = torch.load(dit_init, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = dit.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded DiT init: {dit_init} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    dit.to(device)
    return dit


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--vae_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--dit_init", type=str, default="")
    ap.add_argument("--model_size", type=str, default="base", choices=["base", "large", "xlarge"])
    ap.add_argument("--use_checkpoint", action="store_true")

    ap.add_argument("--video_frames", type=int, default=13)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--cond_frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=None)

    ap.add_argument("--num_steps", type=int, default=1000)
    ap.add_argument("--t_max_ratio", type=float, default=0.3, help="train t in [1, t_max_ratio*num_steps)")
    ap.add_argument("--lambda_x0", type=float, default=0.0, help="optional x0 loss weight (0 disables)")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--vis_interval", type=int, default=500)

    # ---- VIS controls ----
    ap.add_argument("--vis_num", type=int, default=4, help="how many random val clips to visualize each time")
    ap.add_argument("--ddim_steps", type=int, default=25)
    ap.add_argument("--sample_t_ratio", type=float, default=0.8, help="DDIM start t = sample_t_ratio*(num_steps-1)")
    ap.add_argument("--fps", type=int, default=8)

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    amp_enabled = bool(args.amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = (amp_enabled and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print("[INFO] Loading data...", flush=True)
    base = Cholec80(args.data_root, target_size=(args.height, args.width), split="train")
    seqs = base.video_datasets
    n_train = int(0.8 * len(seqs))

    train_set = VideoDataset(seqs[:n_train], args.video_frames, stride=args.stride)
    val_set = VideoDataset(seqs[n_train:], args.video_frames, stride=args.stride)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, min(args.batch_size, 2)), shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    print(f"[INFO] Train clips={len(train_set)} | Val clips={len(val_set)}", flush=True)

    print("[INFO] Loading VAE (frozen wrapper)...", flush=True)
    vae = WanVAE(z_dim=16, vae_pth=args.vae_path, dtype=torch.float32, device=device)

    print(f"[INFO] Building DiT={args.model_size} on cuda:0 ...", flush=True)
    dit = build_dit_single(args.model_size, args.use_checkpoint, args.dit_init)
    dit.train()

    opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) * args.epochs) // max(1, args.accum_steps)
    sched = WarmupCosine(opt, args.warmup_steps, total_steps, args.lr, args.min_lr)

    alphas = make_cosine_schedule(args.num_steps, device=device).float()

    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ckpt_latest = save_dir / "latest.pt"
    ckpt_best = save_dir / "best.pt"

    logger = MetricsLogger(args.save_dir)

    global_step = 0
    start_epoch = 1
    best_val = float("inf")

    def atomic_save(obj, path: Path):
        tmp = str(path) + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, str(path))

    if args.resume and ckpt_latest.exists():
        ckpt = torch.load(str(ckpt_latest), map_location="cpu")
        dit.load_state_dict(ckpt["dit"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        sched.step_count = int(ckpt.get("sched_step", 0))
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[RESUME] epoch={start_epoch} step={global_step} best_val={best_val}", flush=True)

    eff_batch = args.batch_size * args.accum_steps
    print(f"[INFO] Effective batch={eff_batch} (batch={args.batch_size}, accum={args.accum_steps})", flush=True)
    print(f"[INFO] Train t in [1, {max(2, int(args.num_steps*args.t_max_ratio))})", flush=True)
    print(f"[INFO] VIS random K={args.vis_num} | DDIM steps={args.ddim_steps} | sample_t_ratio={args.sample_t_ratio}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        dit.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for bi, clip in enumerate(train_loader):
            x = clip.to(device, non_blocking=True)  # [B,C,T,H,W] in [0,1]
            B = x.shape[0]

            with torch.no_grad():
                z0 = encode_clip_to_latent(vae, x).float()  # [B,16,Tz,Hz,Wz]

            _, _, Tz, _, _ = z0.shape
            t_cond = compute_t_cond_latent(Tz, args.cond_frames, args.video_frames)

            t_max = max(2, int(args.num_steps * args.t_max_ratio))
            t_max = min(t_max, args.num_steps)
            t = torch.randint(1, t_max, (B,), device=device, dtype=torch.long)

            eps_true = torch.randn_like(z0)
            zt = q_sample(z0, t, eps_true, alphas)

            # input to DiT: cond clean, future noisy
            z_in = zt.clone()
            z_in[:, :, :t_cond] = z0[:, :, :t_cond]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                eps_pred = dit(z_in, t, t_cond)

            loss_eps = F.mse_loss(eps_pred.float()[:, :, t_cond:], eps_true.float()[:, :, t_cond:])

            if args.lambda_x0 > 0:
                z0_pred = predict_x0_from_eps(z_in.float(), t, eps_pred.float(), alphas)
                loss_x0 = F.mse_loss(z0_pred[:, :, t_cond:], z0[:, :, t_cond:])
            else:
                loss_x0 = torch.tensor(0.0, device=device)

            loss_total = loss_eps + args.lambda_x0 * loss_x0

            if use_scaler:
                scaler.scale(loss_total / args.accum_steps).backward()
            else:
                (loss_total / args.accum_steps).backward()

            if (bi + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)

                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)
                lr = sched.step()
                global_step += 1

                if global_step % args.log_interval == 0:
                    dt = time.time() - t0
                    print(
                        f"[E{epoch}] step={global_step} "
                        f"| total={loss_total.item():.6f} eps={loss_eps.item():.6f} x0={loss_x0.item():.6f} "
                        f"| lr={lr:.2e} | dt={dt:.1f}s",
                        flush=True
                    )
                    t0 = time.time()

                mse_f = psnr_f = ssim_f = None
                vis_idxs = None

                # ---- VIS: random K samples each time ----
                if len(val_set) > 0 and (global_step % args.vis_interval == 0):
                    dit.eval()
                    try:
                        K = int(max(1, min(args.vis_num, len(val_set))))
                        # random sample indices (no repeat)
                        idxs = random.sample(range(len(val_set)), K)
                        vis_idxs = ",".join(str(i) for i in idxs)

                        gt_list = []
                        pr_list = []
                        m_list, p_list, s_list = [], [], []

                        t_start = int((args.num_steps - 1) * float(args.sample_t_ratio))
                        t_start = max(1, min(args.num_steps - 1, t_start))

                        for idx in idxs:
                            x_vis = val_set[idx].unsqueeze(0).to(device)  # [1,3,T,H,W]
                            with torch.no_grad():
                                z0_vis = encode_clip_to_latent(vae, x_vis).float()
                            _, _, Tzv, _, _ = z0_vis.shape
                            t_cond_v = compute_t_cond_latent(Tzv, args.cond_frames, args.video_frames)

                            z0_pred_vis = ddim_sample_continue(
                                dit=dit,
                                z0=z0_vis,
                                t_cond=t_cond_v,
                                alphas=alphas,
                                t_start=t_start,
                                num_ddim_steps=int(args.ddim_steps),
                                amp_enabled=amp_enabled,
                                amp_dtype=amp_dtype,
                            ).float()

                            pred_vid = decode_latent_to_video(vae, z0_pred_vis)  # [1,3,T,H,W]
                            gt_vid = x_vis                                   # [1,3,T,H,W]

                            gt_list.append(gt_vid[0].detach().cpu())
                            pr_list.append(pred_vid[0].detach().cpu())

                            mse_i, psnr_i, ssim_i = compute_video_metrics_one(
                                gt_vid.detach(), pred_vid.detach(), args.cond_frames
                            )
                            m_list.append(mse_i); p_list.append(psnr_i); s_list.append(ssim_i)

                        # stack to [K,3,T,H,W]
                        gt_stack = torch.stack(gt_list, dim=0)  # cpu
                        pr_stack = torch.stack(pr_list, dim=0)  # cpu

                        # montage video [T,3,2H,K*W]
                        montage = make_montage_video(gt_stack, pr_stack)  # cpu

                        mse_f = float(np.mean(m_list))
                        psnr_f = float(np.mean(p_list))
                        ssim_f = float(np.mean(s_list))

                        tag = f"step{global_step:08d}_k{K:02d}_t{t_start:04d}_dd{int(args.ddim_steps):02d}"
                        out_path = str(vis_dir / f"{tag}_montage.mp4")
                        write_mp4(montage, out_path, fps=int(args.fps))

                        print(f"[VIS] saved montage: {out_path}", flush=True)
                        print(f"[VIS] idxs={idxs}", flush=True)
                        print(f"[SIM] future(mean over K) MSE={mse_f:.6f} PSNR={psnr_f:.2f} SSIM={ssim_f:.4f}", flush=True)

                    except Exception as e:
                        print(f"[VIS][WARN] failed: {e}", flush=True)
                    dit.train()

                # ---- log train row ----
                logger.log({
                    "time": now_ts(),
                    "type": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss_eps": float(loss_eps.item()),
                    "loss_x0": float(loss_x0.item()),
                    "loss_total": float(loss_total.item()),
                    "mse_future": None if mse_f is None else float(mse_f),
                    "psnr_future": None if psnr_f is None else float(psnr_f),
                    "ssim_future": None if ssim_f is None else float(ssim_f),
                    "vis_idxs": vis_idxs,
                    "lr": float(lr),
                })

                # ---- save ckpt ----
                if global_step % args.save_interval == 0:
                    atomic_save({
                        "epoch": epoch,
                        "step": global_step,
                        "dit": dit.state_dict(),
                        "opt": opt.state_dict(),
                        "sched_step": sched.step_count,
                        "best_val": best_val,
                    }, ckpt_latest)
                    print(f"[SAVE] latest.pt updated at step={global_step}", flush=True)

        # ---- epoch end: quick val loss on a few batches ----
        dit.eval()
        val_losses = []
        with torch.no_grad():
            for vi, clip in enumerate(val_loader):
                x = clip.to(device, non_blocking=True)
                B = x.shape[0]
                z0 = encode_clip_to_latent(vae, x).float()
                _, _, Tz, _, _ = z0.shape
                t_cond = compute_t_cond_latent(Tz, args.cond_frames, args.video_frames)

                t_max = max(2, int(args.num_steps * args.t_max_ratio))
                t_max = min(t_max, args.num_steps)
                t = torch.randint(1, t_max, (B,), device=device, dtype=torch.long)

                eps_true = torch.randn_like(z0)
                zt = q_sample(z0, t, eps_true, alphas)
                z_in = zt.clone()
                z_in[:, :, :t_cond] = z0[:, :, :t_cond]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    eps_pred = dit(z_in, t, t_cond)

                loss_eps = F.mse_loss(eps_pred.float()[:, :, t_cond:], eps_true.float()[:, :, t_cond:])
                if args.lambda_x0 > 0:
                    z0_pred = predict_x0_from_eps(z_in.float(), t, eps_pred.float(), alphas)
                    loss_x0 = F.mse_loss(z0_pred[:, :, t_cond:], z0[:, :, t_cond:])
                else:
                    loss_x0 = torch.tensor(0.0, device=device)
                loss_total = loss_eps + args.lambda_x0 * loss_x0
                val_losses.append(float(loss_total.item()))

                if vi >= 2:
                    break

        val_loss = float(np.mean(val_losses)) if len(val_losses) else 0.0
        print(f"[VAL E{epoch}] loss={val_loss:.6f}", flush=True)

        logger.log({
            "time": now_ts(),
            "type": "val",
            "epoch": epoch,
            "step": global_step,
            "loss_eps": None,
            "loss_x0": None,
            "loss_total": val_loss,
            "mse_future": None,
            "psnr_future": None,
            "ssim_future": None,
            "vis_idxs": None,
            "lr": None,
        })

        if val_loss < best_val:
            best_val = val_loss
            atomic_save({
                "epoch": epoch,
                "step": global_step,
                "dit": dit.state_dict(),
                "opt": opt.state_dict(),
                "sched_step": sched.step_count,
                "best_val": best_val,
            }, ckpt_best)
            print(f"[SAVE] best.pt updated: best_val={best_val:.6f}", flush=True)

        atomic_save({
            "epoch": epoch,
            "step": global_step,
            "dit": dit.state_dict(),
            "opt": opt.state_dict(),
            "sched_step": sched.step_count,
            "best_val": best_val,
        }, ckpt_latest)

    print("[DONE] Training finished.", flush=True)


if __name__ == "__main__":
    main()
