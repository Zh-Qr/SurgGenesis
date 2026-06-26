#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import random
from typing import List, Tuple

import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "backbone", "Wan2.1"))
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.utils.utils import str2bool
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES

try:
    import cv2
    USE_OPENCV = True
except Exception:
    USE_OPENCV = False


def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


def read_video_segment_rgb01(
    video_path: str,
    start_frame: int,
    num_frames: int,
    target_hw: Tuple[int, int],  # (H,W)
    pad_mode: str = "repeat_last",
) -> Tuple[torch.Tensor, float, int]:
    """
    Returns:
      frames: [T,3,H,W] RGB float in [0,1]
      fps, total_frames
    """
    if not USE_OPENCV:
        raise RuntimeError("OpenCV not available; cannot read mp4 via cv2.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    H, W = target_hw
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames = []
    last_bgr = None
    for i in range(num_frames):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            if pad_mode == "error":
                cap.release()
                raise RuntimeError(f"Video ended early at i={i}/{num_frames}")
            if last_bgr is None:
                bgr = (torch.zeros((H, W, 3), dtype=torch.uint8)).numpy()
            else:
                if pad_mode == "repeat_last":
                    bgr = last_bgr.copy()
                else:
                    bgr = (0 * last_bgr)

        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        last_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        frames.append(t)

    cap.release()
    return torch.stack(frames, dim=0), float(fps), total


def write_rgb01_mp4(frames_t3hw_01: torch.Tensor, out_path: str, fps: float):
    """
    frames: [T,3,H,W] RGB [0,1]
    """
    if not USE_OPENCV:
        raise RuntimeError("OpenCV not available; cannot write mp4.")
    _ensure_dir(out_path)

    T, C, H, W = frames_t3hw_01.shape
    assert C == 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    x = (frames_t3hw_01.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
    x = x.permute(0, 2, 3, 1).numpy()  # [T,H,W,3] RGB
    for i in range(T):
        bgr = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def write_mask_mp4(cond_len: int, total_len: int, out_path: str, fps: float, hw: Tuple[int, int]):
    """
    mask mp4 is 3ch for pipe.prepare_source
      - known0: cond=0 (black), future=1 (white)  ✅推荐：未来作为生成/编辑区域
      - known1: cond=1, future=0
    """
    if not USE_OPENCV:
        raise RuntimeError("OpenCV not available; cannot write mask mp4.")
    _ensure_dir(out_path)

    H, W = hw
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    for t in range(total_len):
        val = 0 if t < cond_len else 255
        frame = (val * torch.ones((H, W, 3), dtype=torch.uint8)).numpy()
        writer.write(frame)
    writer.release()


def extract_refs(frames_01: torch.Tensor, out_dir: str, indices: List[int]) -> List[str]:
    """
    frames_01: [T,3,H,W] RGB [0,1]
    Save selected frames to png, return list paths (for src_ref_images).
    """
    if not USE_OPENCV:
        raise RuntimeError("OpenCV not available; cannot dump ref images.")
    os.makedirs(out_dir, exist_ok=True)

    T = frames_01.shape[0]
    x = (frames_01.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
    paths = []
    for k, idx in enumerate(indices):
        idx = int(idx)
        if idx < 0 or idx >= T:
            raise ValueError(f"ref idx out of range: {idx} (T={T})")
        rgb = x[idx].permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p = os.path.join(out_dir, f"ref_{idx:03d}_{k:02d}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)
    return paths


def save_video_m11_rgb(video_3thw_m11: torch.Tensor, out_path: str, fps: int):
    """
    video: [3,T,H,W] in [-1,1], RGB
    """
    if not USE_OPENCV:
        raise RuntimeError("OpenCV not available; cannot save mp4.")
    _ensure_dir(out_path)

    x = ((video_3thw_m11.clamp(-1, 1) + 1) / 2 * 255.0).round().to(torch.uint8).cpu()
    x = x.permute(1, 2, 3, 0).numpy()  # [T,H,W,3] RGB

    T, H, W = x.shape[0], x.shape[1], x.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    for i in range(T):
        bgr = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def parse_args():
    p = argparse.ArgumentParser("VACE: read 81 GT frames, use first 41 as condition; generate last 40; output stacked (top=pred, bottom=GT).")

    p.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", type=str, required=True)

    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--start_frame", type=int, default=0)

    p.add_argument("--cond_len", type=int, default=41)
    p.add_argument("--gen_len", type=int, default=40)
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "zeros", "error"])

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--n_prompt", type=str, default="")

    # 关键：未来段不要填噪声（否则会“条件化在噪声上”）
    p.add_argument("--future_fill", type=str, default="black", choices=["black", "repeat_last", "zeros"])

    # refs from cond
    p.add_argument("--use_ref_from_cond", type=str2bool, default=True)
    p.add_argument("--ref_indices", type=str, default="0,20,40")

    # sampling
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sample_steps", type=int, default=40)
    p.add_argument("--sample_shift", type=float, default=16.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
    p.add_argument("--context_scale", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=-1)
    p.add_argument("--offload_model", type=str2bool, default=True)

    # output
    p.add_argument("--save_vs_gt_file", type=str, default="outputs/extend_vs_gt.mp4")
    p.add_argument("--save_pred_only", type=str2bool, default=True)
    p.add_argument("--save_pred_file", type=str, default="outputs/extend_41plus40.mp4")
    p.add_argument("--save_gt_file", type=str, default="outputs/extend_gt_81.mp4")
    p.add_argument("--save_fps", type=int, default=0)
    p.add_argument("--tmp_dir", type=str, default="outputs/tmp_vace_extend_vs_gt")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    assert args.task in WAN_CONFIGS
    assert args.size in SIZE_CONFIGS
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size {args.size} for {args.task}"

    total_len = args.cond_len + args.gen_len
    assert (total_len - 1) % 4 == 0, f"cond_len+gen_len must be 4n+1, got {total_len}"
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    cfg = WAN_CONFIGS[args.task]
    W, H = SIZE_CONFIGS[args.size]
    os.makedirs(args.tmp_dir, exist_ok=True)

    logging.info(f"[cfg] total_len={total_len} cond_len={args.cond_len} gen_len={args.gen_len}")
    logging.info(f"[cfg] future_fill={args.future_fill} seed={args.base_seed}")
    logging.info(f"[data] video={args.video_path} start_frame={args.start_frame}")

    # 1) 连续读取 81 帧 GT（用于下半部分展示）
    gt_full_01, fps, total = read_video_segment_rgb01(
        args.video_path,
        args.start_frame,
        total_len,
        (H, W),
        args.pad_mode,
    )
    logging.info(f"[data] fps={fps:.3f} total_frames={total}")

    cond_01 = gt_full_01[:args.cond_len].clone()  # [41,3,H,W]

    # 2) build src clip fed to VACE: first 41 real, last 40 filled (NOT GT, NOT noise)
    src_01 = torch.zeros((total_len, 3, H, W), dtype=torch.float32)
    src_01[:args.cond_len] = cond_01

    if args.future_fill in ("black", "zeros"):
        src_01[args.cond_len:] = 0.0
    else:  # repeat_last
        last = cond_01[-1:].repeat(args.gen_len, 1, 1, 1)
        src_01[args.cond_len:] = last

    # 3) write temp mp4s
    clip_path = os.path.join(args.tmp_dir, f"src_clip_f{args.start_frame}_n{total_len}.mp4")
    mask_path = os.path.join(args.tmp_dir, f"mask_clip_f{args.start_frame}_n{total_len}.mp4")
    write_rgb01_mp4(src_01, clip_path, fps=fps)
    write_mask_mp4(args.cond_len, total_len, mask_path, fps=fps, hw=(H, W))
    logging.info(f"[tmp] src_clip={clip_path}")
    logging.info(f"[tmp] mask_clip={mask_path}")

    # 4) refs from cond
    ref_paths = None
    if args.use_ref_from_cond:
        idxs = [int(x.strip()) for x in args.ref_indices.split(",") if x.strip() != ""]
        idxs = [min(max(i, 0), args.cond_len - 1) for i in idxs]
        ref_dir = os.path.join(args.tmp_dir, f"ref_frames_{os.path.basename(args.video_path)}_f{args.start_frame}")
        ref_paths = extract_refs(cond_01, ref_dir, idxs)
        logging.info(f"[ref] {len(ref_paths)} refs -> {ref_paths}")

    # 5) init pipe
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    pipe = wan.WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    # 6) official prepare_source (ONLY uses src_01 + mask + refs; GT not used)
    src_video, src_mask, src_ref_images = pipe.prepare_source(
        [clip_path],
        [mask_path],
        [ref_paths] if ref_paths is not None else [None],
        total_len,
        SIZE_CONFIGS[args.size],
        pipe.device,
    )
    logging.info("[vace] prepared source via pipe.prepare_source")

    # 7) official generate
    pred = pipe.generate(
        args.prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[args.size],
        frame_num=total_len,
        context_scale=args.context_scale,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        n_prompt=args.n_prompt,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    if rank != 0:
        return

    # 8) prefix 强制一致：top 的前 41 帧 = cond
    pred = pred.detach().cpu()
    cond_m11 = (cond_01.permute(1, 0, 2, 3) * 2.0 - 1.0).to(pred.dtype)
    pred[:, :args.cond_len] = cond_m11

    # 9) GT 转成 [-1,1]
    gt_m11 = (gt_full_01.permute(1, 0, 2, 3) * 2.0 - 1.0).to(pred.dtype)  # [3,81,H,W]

    # 10) 拼接成上下对比：height 维度拼接 => [3,81,2H,W]
    vs_gt = torch.cat([pred, gt_m11], dim=2)

    # fps
    save_fps = args.save_fps
    if save_fps <= 0:
        save_fps = int(getattr(cfg, "sample_fps", 24))

    # save
    save_video_m11_rgb(vs_gt, args.save_vs_gt_file, fps=save_fps)
    logging.info(f"[save] vs_gt -> {args.save_vs_gt_file}")

    if args.save_pred_only:
        save_video_m11_rgb(pred, args.save_pred_file, fps=save_fps)
        logging.info(f"[save] pred81 -> {args.save_pred_file}")
        save_video_m11_rgb(gt_m11, args.save_gt_file, fps=save_fps)
        logging.info(f"[save] gt81 -> {args.save_gt_file}")

    logging.info("Finished.")


if __name__ == "__main__":
    main()
