#!/usr/bin/env python3
"""Prediction-gap / GT-latent upper-bound experiment for Wan2.2-TI2V.

This script builds a random real-video slice benchmark:

1. Randomly sample real clips and split each clip into:
   condition frames + real future frames.
2. Generate predicted future videos from the last condition frame.
3. Encode both predicted future clips and GT future clips with the Wan VAE.
4. Measure latent-space prediction gap over long horizons.
5. Optionally decode GT latents back to pixels to estimate a VAE/decoder upper
   bound for frame-space metrics.

The output is intended to support the paper's "Prediction Gap / GT-latent upper
bound" analysis.  It is not a downstream segmentation/triplet decoder by itself,
but it produces the same predicted-vs-GT-latent table format used by
build_prediction_gap_table.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DIFFSYNTH_DIR = os.environ.get("DIFFSYNTH_DIR")
if DIFFSYNTH_DIR:
    sys.path.insert(0, DIFFSYNTH_DIR)

try:
    import torch
    from PIL import Image

    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import VideoData, save_video
except ModuleNotFoundError as exc:  # pragma: no cover - friendly CLI failure.
    raise SystemExit(
        "This script must run in the DiffSynth/Wan environment. "
        "Example: DIFFSYNTH_DIR=/home/xuecheng/WAN_clean/DiffSynth-Studio "
        "/mnt/data3/zhongqirui/env/SAM/bin/python src/eval/prediction_gap_gt_latent_experiment.py ..."
    ) from exc


DEFAULT_PROMPT = (
    "realistic laparoscopic cholecystectomy video, intraoperative endoscopic view, "
    "surgical instruments interacting with soft tissue, physically plausible motion, "
    "natural camera movement, stable surgical lighting"
)
DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, underexposed, blurry, low quality, static image, text, watermark, "
    "cartoon, painting, unrealistic motion, distorted anatomy, duplicated instruments"
)


@dataclass
class StageSpec:
    label: str
    lora_path: Path | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random-slice predicted-vs-GT-latent gap experiment for Wan2.2-TI2V."
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--manifest", default="", help="Existing manifest for eval-only mode.")
    p.add_argument("--skip_generation", action="store_true", help="Only run latent/frame analysis from --manifest.")
    p.add_argument("--reuse_existing", action="store_true", help="Do not regenerate videos already present in output_dir.")

    p.add_argument("--model_dir", default="/mnt/data3/zhongqirui/WAN2_2")
    p.add_argument("--metadata_csv", default="/home/xuecheng/WAN_SAM/finetune/diffsynth_wan22_ti2v/work/cholec80_wan22_ti2v/metadata_val.csv")
    p.add_argument("--dataset_base_path", default="/mnt/data3/zhongqirui/Cholec80_slice")
    p.add_argument("--stage_lora", action="append", default=[], help="label=/path/to/step-N.safetensors")
    p.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--num_cases", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=20260610)
    p.add_argument("--cond_frames", type=int, default=41)
    p.add_argument("--num_frames", type=int, default=81, help="Generated clip length; future length is num_frames-1.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lora_alpha", type=float, default=1.5)
    p.add_argument("--only_labels", default="", help="Comma/space separated labels to generate/evaluate.")
    p.add_argument("--skip_existing_outputs", action="store_true")

    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tile_size", default="34,34")
    p.add_argument("--tile_stride", default="18,16")
    p.add_argument("--horizons", default="1,2,3,4,5")
    p.add_argument("--decode_gt_upper_bound", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_recon_videos", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--plot_metric", default="rmse", choices=["rmse", "latent_rmse", "psnr", "ssim"])
    return p.parse_args()


def parse_stage_spec(text: str) -> StageSpec:
    if "=" not in text:
        raise ValueError(f"Bad --stage_lora value: {text}. Expected label=/path/to/checkpoint.safetensors")
    label, path = text.split("=", 1)
    label = label.strip()
    ckpt = Path(path).expanduser()
    if not label:
        raise ValueError(f"Empty stage label in --stage_lora={text}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found for {label}: {ckpt}")
    return StageSpec(label=label, lora_path=ckpt)


def parse_labels(text: str) -> set[str]:
    return {part.strip() for part in text.replace(",", " ").split() if part.strip()}


def parse_pair(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated integers, got {text}")
    return vals[0], vals[1]


def torch_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def model_configs(model_dir: Path, include_dit: bool) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    if include_dit:
        shards = sorted(model_dir.glob("diffusion_pytorch_model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No diffusion_pytorch_model-*.safetensors under {model_dir}")
        configs.append(ModelConfig([str(path) for path in shards]))
        configs.append(ModelConfig(str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")))
    configs.append(ModelConfig(str(model_dir / "Wan2.2_VAE.pth")))
    return configs


def load_pipe(model_dir: Path, dtype: torch.dtype, device: str, include_dit: bool) -> WanVideoPipeline:
    tokenizer_config = None if not include_dit else ModelConfig(str(model_dir / "google" / "umt5-xxl"))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs(model_dir, include_dit=include_dit),
        tokenizer_config=tokenizer_config,
    )
    return pipe


def read_metadata(metadata_csv: str, dataset_base_path: str) -> list[dict[str, str]]:
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in metadata CSV: {metadata_csv}")
    for row in rows:
        video = row["video"]
        row["video_path"] = video if os.path.isabs(video) else os.path.join(dataset_base_path, video)
    return rows


def sample_random_clip(row: dict[str, str], args: argparse.Namespace, rng: random.Random):
    video = VideoData(row["video_path"], height=args.height, width=args.width)
    needed = args.cond_frames + max(1, args.num_frames - 1)
    if len(video) < needed:
        raise ValueError(f"Video too short: {row['video_path']}, frames={len(video)}, needed={needed}")
    start = rng.randint(0, max(0, len(video) - needed))
    condition = [video[start + i] for i in range(args.cond_frames)]
    future = [video[start + args.cond_frames + i] for i in range(args.num_frames - 1)]
    gt_with_condition = [condition[-1], *future]
    return {
        "start_frame": start,
        "condition": condition,
        "future": future,
        "gt_with_condition": gt_with_condition,
        "input_image": condition[-1],
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_one(pipe: WanVideoPipeline, prompt: str, negative_prompt: str, input_image: Image.Image, args: argparse.Namespace):
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=args.tiled,
        tile_size=parse_pair(args.tile_size),
        tile_stride=parse_pair(args.tile_stride),
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
    )


def build_random_manifest(args: argparse.Namespace, stages: list[StageSpec]) -> dict:
    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.random_seed)
    rows = read_metadata(args.metadata_csv, args.dataset_base_path)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    all_stages = ([StageSpec("base", None)] if args.include_base else []) + stages
    active = parse_labels(args.only_labels)
    if active:
        all_stages = [stage for stage in all_stages if stage.label in active]
    if not all_stages:
        raise ValueError("No active stages. Pass --include_base and/or --stage_lora.")

    manifest = {
        "experiment": "prediction_gap_gt_latent",
        "model_dir": args.model_dir,
        "metadata_csv": args.metadata_csv,
        "dataset_base_path": args.dataset_base_path,
        "num_cases": args.num_cases,
        "random_seed": args.random_seed,
        "cond_frames": args.cond_frames,
        "num_frames": args.num_frames,
        "future_frames": args.num_frames - 1,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "lora_alpha": args.lora_alpha,
        "stages": [{"label": s.label, "lora_path": "" if s.lora_path is None else str(s.lora_path)} for s in all_stages],
        "cases": [],
    }

    selected = 0
    for row in shuffled:
        if selected >= args.num_cases:
            break
        case_name = f"case_{selected:03d}"
        case_dir = cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            sample = sample_random_clip(row, args, rng)
        except Exception as exc:
            print(f"[skip] {row.get('video_path')} reason={exc}", flush=True)
            continue

        condition_path = case_dir / "condition.mp4"
        gt_future_path = case_dir / "ground_truth_future.mp4"
        gt_with_condition_path = case_dir / "gt_with_condition.mp4"
        if not condition_path.exists() or not args.reuse_existing:
            save_video(sample["condition"], str(condition_path), fps=args.fps, quality=5)
        if not gt_future_path.exists() or not args.reuse_existing:
            save_video(sample["future"], str(gt_future_path), fps=args.fps, quality=5)
        if not gt_with_condition_path.exists() or not args.reuse_existing:
            save_video(sample["gt_with_condition"], str(gt_with_condition_path), fps=args.fps, quality=5)

        prompt = row.get("prompt") or args.prompt
        record = {
            "case": case_name,
            "source_video": row["video_path"],
            "source_row": row,
            "start_frame": sample["start_frame"],
            "prompt": prompt,
            "condition_video": str(condition_path),
            "ground_truth_future": str(gt_future_path),
            "gt_with_condition": str(gt_with_condition_path),
            "generated": {},
        }
        write_json(case_dir / "case_metadata.json", record)
        manifest["cases"].append(record)
        selected += 1

    if len(manifest["cases"]) < args.num_cases:
        raise RuntimeError(f"Only selected {len(manifest['cases'])}/{args.num_cases} valid random clips.")

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    if args.skip_generation:
        return manifest

    model_dir = Path(args.model_dir)
    for stage in all_stages:
        pending = []
        for record in manifest["cases"]:
            out_path = output_dir / "cases" / record["case"] / f"{stage.label}_generated.mp4"
            if (args.reuse_existing or args.skip_existing_outputs) and out_path.is_file():
                record["generated"][stage.label] = str(out_path)
                continue
            pending.append(record)
        if not pending:
            print(f"[generate] {stage.label} already complete", flush=True)
            write_json(manifest_path, manifest)
            continue

        print(f"[generate] loading Wan2.2-TI2V split={stage.label}", flush=True)
        pipe = load_pipe(model_dir, torch_dtype(args.torch_dtype), args.device, include_dit=True)
        if stage.lora_path is not None:
            print(f"[generate] loading LoRA {stage.label}: {stage.lora_path}", flush=True)
            pipe.load_lora(pipe.dit, str(stage.lora_path), alpha=args.lora_alpha)
        for record in pending:
            case_dir = output_dir / "cases" / record["case"]
            gt_video = read_video_pil(record["gt_with_condition"])
            input_image = gt_video[0]
            print(f"[generate] {stage.label} {record['case']}", flush=True)
            video = generate_one(pipe, record["prompt"], args.negative_prompt, input_image, args)
            out_path = case_dir / f"{stage.label}_generated.mp4"
            save_video(video, str(out_path), fps=args.fps, quality=5)
            record["generated"][stage.label] = str(out_path)
            write_json(case_dir / "case_metadata.json", record)
            write_json(manifest_path, manifest)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(manifest_path, manifest)
    return manifest


def read_video_pil(path: str | Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            for frame in reader:
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=-1)
                frames.append(Image.fromarray(frame[..., :3].astype(np.uint8)))
        finally:
            reader.close()
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("Video decoding requires imageio or opencv-python.") from exc
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame.astype(np.uint8)))
        finally:
            cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def align_clip_with_gt(gt_clip: list[Image.Image], predicted: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image], int]:
    """Return aligned GT and predicted clips that include the condition frame at index 0."""
    if len(predicted) == len(gt_clip):
        return gt_clip, predicted, 0
    if len(predicted) == len(gt_clip) - 1:
        return gt_clip, [gt_clip[0], *predicted], -1
    limit = min(len(gt_clip), len(predicted))
    return gt_clip[:limit], predicted[:limit], 0


def frames_to_numpy(frames: Sequence[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames], axis=0)


def frame_metrics(gt_frames: Sequence[Image.Image], pred_frames: Sequence[Image.Image]) -> dict[str, float]:
    gt = frames_to_numpy(gt_frames).astype(np.float64) / 255.0
    pred = frames_to_numpy(pred_frames).astype(np.float64) / 255.0
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]
    diff = gt - pred
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = 99.0 if mse <= 1e-12 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    mu_x = float(gt.mean())
    mu_y = float(pred.mean())
    var_x = float(gt.var())
    var_y = float(pred.var())
    cov = float(((gt - mu_x) * (pred - mu_y)).mean())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return {"mse": mse, "rmse": rmse, "psnr": psnr, "ssim": float(ssim)}


@torch.no_grad()
def encode_clip(pipe: WanVideoPipeline, frames: list[Image.Image], args: argparse.Namespace) -> torch.Tensor:
    pipe.load_models_to_device(["vae"])
    video_tensor = pipe.preprocess_video(frames, torch_dtype=pipe.torch_dtype, device=pipe.device)
    latents = pipe.vae.encode(
        video_tensor,
        device=pipe.device,
        tiled=args.tiled,
        tile_size=parse_pair(args.tile_size),
        tile_stride=parse_pair(args.tile_stride),
    )
    return latents.detach().to(dtype=torch.float32, device="cpu")


@torch.no_grad()
def decode_clip(pipe: WanVideoPipeline, latents_cpu: torch.Tensor, args: argparse.Namespace) -> list[Image.Image]:
    pipe.load_models_to_device(["vae"])
    latents = latents_cpu.to(dtype=pipe.torch_dtype, device=pipe.device)
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=args.tiled,
        tile_size=parse_pair(args.tile_size),
        tile_stride=parse_pair(args.tile_stride),
    )
    return pipe.vae_output_to_video(video)


def latent_metrics(z_gt: torch.Tensor, z_pred: torch.Tensor, fps: float) -> list[dict[str, float]]:
    _, _, t, h, w = z_gt.shape
    t = min(t, z_pred.shape[2])
    h = min(h, z_pred.shape[3])
    w = min(w, z_pred.shape[4])
    gt = z_gt[:, :, :t, :h, :w].float()
    pred = z_pred[:, :, :t, :h, :w].float()
    gt_anchor = gt[:, :, 0]
    pred_anchor = pred[:, :, 0]
    rows = []
    for latent_idx in range(1, t):
        a = gt[:, :, latent_idx]
        b = pred[:, :, latent_idx]
        diff = b - a
        mse = float(torch.mean(diff * diff).item())
        rmse = math.sqrt(mse)
        mae = float(torch.mean(torch.abs(diff)).item())
        gt_rms = math.sqrt(float(torch.mean(a * a).item()))
        pred_rms = math.sqrt(float(torch.mean(b * b).item()))
        normalized_rmse = rmse / max(gt_rms, 1e-12)
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        denom = float(torch.linalg.norm(a_flat).item() * torch.linalg.norm(b_flat).item())
        cosine_similarity = 0.0 if denom <= 1e-12 else float(torch.dot(a_flat, b_flat).item() / denom)

        gt_delta = a - gt_anchor
        pred_delta = b - pred_anchor
        delta_diff = pred_delta - gt_delta
        delta_mse = float(torch.mean(delta_diff * delta_diff).item())
        delta_rmse = math.sqrt(delta_mse)
        gt_delta_rms = math.sqrt(float(torch.mean(gt_delta * gt_delta).item()))
        pred_delta_rms = math.sqrt(float(torch.mean(pred_delta * pred_delta).item()))
        normalized_delta_rmse = delta_rmse / max(gt_delta_rms, 1e-12)
        gt_delta_flat = gt_delta.reshape(-1)
        pred_delta_flat = pred_delta.reshape(-1)
        delta_denom = float(torch.linalg.norm(gt_delta_flat).item() * torch.linalg.norm(pred_delta_flat).item())
        delta_cosine_similarity = (
            0.0 if delta_denom <= 1e-12 else float(torch.dot(gt_delta_flat, pred_delta_flat).item() / delta_denom)
        )
        rows.append(
            {
                "latent_index": latent_idx,
                "latent_time_sec": float((latent_idx * 4) / max(fps, 1e-9)),
                "latent_mse": mse,
                "latent_rmse": rmse,
                "latent_mae": mae,
                "gt_latent_rms": gt_rms,
                "pred_latent_rms": pred_rms,
                "normalized_latent_rmse": normalized_rmse,
                "latent_cosine_similarity": cosine_similarity,
                "latent_cosine_distance": 1.0 - cosine_similarity,
                "delta_latent_mse": delta_mse,
                "delta_latent_rmse": delta_rmse,
                "gt_delta_latent_rms": gt_delta_rms,
                "pred_delta_latent_rms": pred_delta_rms,
                "normalized_delta_latent_rmse": normalized_delta_rmse,
                "delta_latent_cosine_similarity": delta_cosine_similarity,
                "delta_latent_cosine_distance": 1.0 - delta_cosine_similarity,
            }
        )
    return rows


def mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(np.mean(arr)) if arr else float("nan")


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "sem": std / math.sqrt(max(1, len(arr)))}


def fmt(value: float) -> str:
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.8f}"


def summarize_case_horizons(step_rows: list[dict], horizons: list[float], metrics: list[str]) -> list[dict]:
    out = []
    cases = sorted({row["case"] for row in step_rows})
    labels = sorted({row.get("label", "") for row in step_rows})
    methods = sorted({row.get("method", "") for row in step_rows})
    for case in cases:
        for label in labels:
            for method in methods:
                subset_base = [
                    row
                    for row in step_rows
                    if row["case"] == case and row.get("label", "") == label and row.get("method", "") == method
                ]
                if not subset_base:
                    continue
                for horizon in horizons:
                    subset = [row for row in subset_base if float(row["time_sec"]) <= horizon + 1e-9]
                    if not subset:
                        continue
                    record = {
                        "case": case,
                        "label": label,
                        "method": method,
                        "horizon_sec": horizon,
                        "num_steps": len(subset),
                    }
                    for metric in metrics:
                        record[metric] = mean(float(row[metric]) for row in subset if row.get(metric) not in ("", None))
                    out.append(record)
    return out


def summarize_global(case_rows: list[dict], metrics: list[str]) -> list[dict]:
    out = []
    labels = sorted({row.get("label", "") for row in case_rows})
    methods = sorted({row.get("method", "") for row in case_rows})
    horizons = sorted({float(row["horizon_sec"]) for row in case_rows})
    for label in labels:
        for method in methods:
            for horizon in horizons:
                subset = [
                    row
                    for row in case_rows
                    if row.get("label", "") == label
                    and row.get("method", "") == method
                    and abs(float(row["horizon_sec"]) - horizon) < 1e-9
                ]
                if not subset:
                    continue
                record = {
                    "label": label,
                    "method": method,
                    "horizon_sec": horizon,
                    "num_cases": len(subset),
                    "num_steps": sum(int(row["num_steps"]) for row in subset),
                }
                for metric in metrics:
                    metric_stats = stats([float(row[metric]) for row in subset if row.get(metric) not in ("", None)])
                    record[f"{metric}_mean"] = metric_stats["mean"]
                    record[f"{metric}_std"] = metric_stats["std"]
                    record[f"{metric}_sem"] = metric_stats["sem"]
                out.append(record)
    return out


def direction(metric: str) -> str:
    lower_names = ["rmse", "mse", "mae", "distance", "gap", "lpips", "fvd"]
    return "lower" if any(name in metric.lower() for name in lower_names) else "higher"


def gap_values(pred: float, gt: float, metric: str) -> tuple[float, float]:
    if direction(metric) == "lower":
        abs_gap = pred - gt
    else:
        abs_gap = gt - pred
    rel_gap = float("nan") if abs(gt) <= 1e-12 else abs_gap / abs(gt)
    return abs_gap, rel_gap


def build_prediction_gap_tables(latent_summary: list[dict], frame_summary: list[dict], output_dir: Path) -> None:
    long_rows = []
    for row in latent_summary:
        if row["method"] != "predicted":
            continue
        h = float(row["horizon_sec"])
        label = row["label"]
        for metric in ("latent_rmse", "latent_mae", "latent_cosine_distance"):
            value = row.get(f"{metric}_mean")
            if value is None or not np.isfinite(value):
                continue
            long_rows.append({"task": f"latent_planning/{label}", "metric": f"{metric}@{h:g}s", "method": "predicted", "value": value})
            long_rows.append({"task": f"latent_planning/{label}", "metric": f"{metric}@{h:g}s", "method": "gt_latent", "value": 0.0})

    for row in frame_summary:
        h = float(row["horizon_sec"])
        label = row["label"]
        method = row["method"]
        if method not in {"predicted", "gt_latent"}:
            continue
        for metric in ("rmse", "psnr", "ssim"):
            value = row.get(f"{metric}_mean")
            if value is None or not np.isfinite(value):
                continue
            long_rows.append({"task": f"frame_future/{label}", "metric": f"{metric}@{h:g}s", "method": method, "value": value})

    write_csv(output_dir / "prediction_gap_input.csv", [{**r, "value": fmt(r["value"])} for r in long_rows])

    values: dict[tuple[str, str], dict[str, float]] = {}
    for row in long_rows:
        key = (str(row["task"]), str(row["metric"]))
        values.setdefault(key, {})[str(row["method"])] = float(row["value"])

    wide_rows = []
    for (task, metric), methods in sorted(values.items()):
        pred = methods.get("predicted")
        gt = methods.get("gt_latent")
        out = {
            "task": task,
            "metric": metric,
            "direction": direction(metric),
            "predicted": "" if pred is None else fmt(pred),
            "gt_latent": "" if gt is None else fmt(gt),
            "prediction_gap_abs": "",
            "prediction_gap_rel": "",
            "prediction_gap_rel_percent": "",
        }
        if pred is not None and gt is not None:
            abs_gap, rel_gap = gap_values(pred, gt, metric)
            out["prediction_gap_abs"] = fmt(abs_gap)
            if np.isfinite(rel_gap):
                out["prediction_gap_rel"] = fmt(rel_gap)
                out["prediction_gap_rel_percent"] = f"{100.0 * rel_gap:.2f}"
        wide_rows.append(out)
    write_csv(output_dir / "prediction_gap_table.csv", wide_rows)
    write_json(
        output_dir / "prediction_gap_table.json",
        {
            "notes": [
                "latent_planning/gt_latent uses zero latent error as the ideal upper bound.",
                "frame_future/gt_latent uses GT future encoded and decoded through the same Wan VAE.",
            ],
            "rows": wide_rows,
        },
    )


def plot_latent_curve(path: Path, summary_rows: list[dict], metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=220)
    labels = sorted({row["label"] for row in summary_rows if row["method"] == "predicted"})
    colors = ["#6c757d", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for idx, label in enumerate(labels):
        rows = [row for row in summary_rows if row["label"] == label and row["method"] == "predicted"]
        rows.sort(key=lambda r: float(r["horizon_sec"]))
        if not rows:
            continue
        xs = [float(row["horizon_sec"]) for row in rows]
        ys = [float(row[f"{metric}_mean"]) for row in rows]
        sem = [float(row.get(f"{metric}_sem", 0.0)) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.9, color=colors[idx % len(colors)], label=label)
        ax.fill_between(xs, np.asarray(ys) - np.asarray(sem), np.asarray(ys) + np.asarray(sem), color=colors[idx % len(colors)], alpha=0.13, linewidth=0)
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Latent Prediction Gap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_upper_bound_curve(path: Path, summary_rows: list[dict], metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=220)
    labels = sorted({row["label"] for row in summary_rows})
    colors = ["#6c757d", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for idx, label in enumerate(labels):
        for method, linestyle in (("predicted", "-"), ("gt_latent", "--")):
            rows = [row for row in summary_rows if row["label"] == label and row["method"] == method]
            rows.sort(key=lambda r: float(r["horizon_sec"]))
            if not rows:
                continue
            xs = [float(row["horizon_sec"]) for row in rows]
            ys = [float(row[f"{metric}_mean"]) for row in rows]
            name = label if method == "predicted" else f"{label} GT-latent"
            ax.plot(xs, ys, marker="o", linewidth=1.7, linestyle=linestyle, color=colors[idx % len(colors)], label=name)
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel(metric.upper())
    ax.set_title("Predicted Future vs GT-latent Upper Bound")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7.2, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def format_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})
    return out


def evaluate_manifest(args: argparse.Namespace, manifest: dict) -> None:
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "prediction_gap_metrics"
    fig_dir = output_dir / "prediction_gap_figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({label for case in manifest.get("cases", []) for label in (case.get("generated") or {}).keys()})
    active = parse_labels(args.only_labels)
    if active:
        labels = [label for label in labels if label in active]
    if not labels:
        raise RuntimeError("No generated labels found for prediction-gap evaluation.")

    horizons = [float(x.strip()) for x in args.horizons.split(",") if x.strip()]
    fps = float(manifest.get("fps") or args.fps)
    model_dir = Path(manifest.get("model_dir") or args.model_dir)
    pipe = load_pipe(model_dir, torch_dtype(args.torch_dtype), args.device, include_dit=False)

    latent_step_rows: list[dict] = []
    frame_step_rows: list[dict] = []
    alignment_rows: list[dict] = []

    for case in manifest.get("cases", []):
        case_name = case["case"]
        gt_clip_path = case.get("gt_with_condition")
        if not gt_clip_path:
            # Older manifests can be upgraded by combining the first condition frame
            # with ground_truth_future.
            condition = read_video_pil(case["condition_video"])
            future = read_video_pil(case["ground_truth_future"])
            gt_clip = [condition[-1], *future]
        else:
            gt_clip = read_video_pil(gt_clip_path)
        print(f"[encode] GT {case_name}", flush=True)
        z_gt = encode_clip(pipe, gt_clip, args)
        gt_recon_future: list[Image.Image] | None = None
        if args.decode_gt_upper_bound:
            print(f"[decode] GT-latent upper bound {case_name}", flush=True)
            recon = decode_clip(pipe, z_gt, args)
            _, recon_aligned, _ = align_clip_with_gt(gt_clip, recon)
            gt_recon_future = recon_aligned[1 : min(len(gt_clip), len(recon_aligned))]
            if args.save_recon_videos:
                recon_path = output_dir / "cases" / case_name / "gt_latent_reconstruction.mp4"
                recon_path.parent.mkdir(parents=True, exist_ok=True)
                save_video(recon_aligned, str(recon_path), fps=int(fps), quality=5)

        for label in labels:
            gen_path = (case.get("generated") or {}).get(label)
            if not gen_path:
                continue
            pred_raw = read_video_pil(gen_path)
            gt_aligned, pred_clip, offset = align_clip_with_gt(gt_clip, pred_raw)
            limit = min(len(gt_aligned), len(pred_clip))
            gt_aligned = gt_aligned[:limit]
            pred_clip = pred_clip[:limit]
            print(f"[encode] predicted {label} {case_name}", flush=True)
            z_pred = encode_clip(pipe, pred_clip, args)
            for row in latent_metrics(z_gt, z_pred, fps):
                latent_step_rows.append(
                    {
                        "case": case_name,
                        "label": label,
                        "method": "predicted",
                        "latent_index": row["latent_index"],
                        "time_sec": row["latent_time_sec"],
                        **{k: v for k, v in row.items() if k not in {"latent_index", "latent_time_sec"}},
                    }
                )

            future_limit = min(len(gt_aligned), len(pred_clip)) - 1
            for frame_idx in range(future_limit):
                t_sec = (frame_idx + 1) / max(fps, 1e-9)
                pred_metrics = frame_metrics([gt_aligned[frame_idx + 1]], [pred_clip[frame_idx + 1]])
                frame_step_rows.append(
                    {
                        "case": case_name,
                        "label": label,
                        "method": "predicted",
                        "frame_index": frame_idx,
                        "time_sec": t_sec,
                        **pred_metrics,
                    }
                )
                if gt_recon_future is not None and frame_idx < len(gt_recon_future):
                    recon_metrics = frame_metrics([gt_aligned[frame_idx + 1]], [gt_recon_future[frame_idx]])
                    frame_step_rows.append(
                        {
                            "case": case_name,
                            "label": label,
                            "method": "gt_latent",
                            "frame_index": frame_idx,
                            "time_sec": t_sec,
                            **recon_metrics,
                        }
                    )
            alignment_rows.append(
                {
                    "case": case_name,
                    "label": label,
                    "gt_clip_frames": len(gt_clip),
                    "generated_frames": len(pred_raw),
                    "aligned_frames": limit,
                    "generated_offset_rule": offset,
                    "gt_latent_shape": list(z_gt.shape),
                    "pred_latent_shape": list(z_pred.shape),
                }
            )

    latent_case_rows = summarize_case_horizons(
        latent_step_rows,
        horizons,
        [
            "latent_mse",
            "latent_rmse",
            "latent_mae",
            "gt_latent_rms",
            "pred_latent_rms",
            "normalized_latent_rmse",
            "latent_cosine_distance",
            "latent_cosine_similarity",
            "delta_latent_mse",
            "delta_latent_rmse",
            "gt_delta_latent_rms",
            "pred_delta_latent_rms",
            "normalized_delta_latent_rmse",
            "delta_latent_cosine_distance",
            "delta_latent_cosine_similarity",
        ],
    )
    latent_summary = summarize_global(
        latent_case_rows,
        [
            "latent_mse",
            "latent_rmse",
            "latent_mae",
            "gt_latent_rms",
            "pred_latent_rms",
            "normalized_latent_rmse",
            "latent_cosine_distance",
            "latent_cosine_similarity",
            "delta_latent_mse",
            "delta_latent_rmse",
            "gt_delta_latent_rms",
            "pred_delta_latent_rms",
            "normalized_delta_latent_rmse",
            "delta_latent_cosine_distance",
            "delta_latent_cosine_similarity",
        ],
    )
    frame_case_rows = summarize_case_horizons(frame_step_rows, horizons, ["mse", "rmse", "psnr", "ssim"])
    frame_summary = summarize_global(frame_case_rows, ["mse", "rmse", "psnr", "ssim"])

    write_csv(metrics_dir / "latent_step_metrics.csv", format_rows(latent_step_rows))
    write_csv(metrics_dir / "latent_horizon_case_metrics.csv", format_rows(latent_case_rows))
    write_csv(metrics_dir / "latent_horizon_summary.csv", format_rows(latent_summary))
    write_csv(metrics_dir / "frame_step_metrics.csv", format_rows(frame_step_rows))
    write_csv(metrics_dir / "frame_horizon_case_metrics.csv", format_rows(frame_case_rows))
    write_csv(metrics_dir / "frame_horizon_summary.csv", format_rows(frame_summary))
    write_csv(metrics_dir / "alignment_audit.csv", alignment_rows)
    build_prediction_gap_tables(latent_summary, frame_summary, metrics_dir)

    plot_latent_curve(fig_dir / "latent_prediction_gap_curve.png", latent_summary, "latent_rmse")
    plot_upper_bound_curve(fig_dir / "frame_upper_bound_gap_curve.png", frame_summary, "rmse")

    report = {
        "manifest": str(output_dir / "manifest.json"),
        "labels": labels,
        "horizons": horizons,
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(fig_dir),
        "notes": [
            "GT-latent upper bound is computed by encoding real future clips with Wan VAE.",
            "Latent upper bound has zero latent prediction error by definition.",
            "Frame upper bound is the VAE reconstruction error from GT latents.",
            "Predicted latent is measured by re-encoding generated future videos with the same Wan VAE.",
        ],
    }
    write_json(metrics_dir / "prediction_gap_report.json", report)
    print(f"[done] metrics={metrics_dir}", flush=True)
    print(f"[done] figures={fig_dir}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [parse_stage_spec(item) for item in args.stage_lora]

    if args.manifest:
        with Path(args.manifest).open(encoding="utf-8") as f:
            manifest = json.load(f)
        # Keep a local copy so output_dir is self-contained.
        local_manifest = output_dir / "manifest.json"
        if Path(args.manifest).resolve() != local_manifest.resolve():
            shutil.copy2(args.manifest, local_manifest)
    else:
        manifest = build_random_manifest(args, stages)

    evaluate_manifest(args, manifest)


if __name__ == "__main__":
    main()
