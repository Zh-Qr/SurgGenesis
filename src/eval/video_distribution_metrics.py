#!/usr/bin/env python3
"""Compute distribution-level video distances for paper revision experiments.

The main use case is FVD-style evaluation from a three-stage generation
manifest.  By default, the script tries to use torchvision's Kinetics-pretrained
R3D-18 features.  A lightweight deterministic feature backend is also provided
for smoke tests and environments without cached video-model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class VideoPairSet:
    label: str
    gt_paths: List[Path]
    gen_paths: List[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Three-stage generation manifest.json.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--labels", default="", help="Comma-separated labels. Defaults to labels found in manifest.")
    p.add_argument(
        "--extra_videos",
        action="append",
        default=[],
        help="Optional label=/path/to/video_dir for external methods such as EndoGen or Surg-World.",
    )
    p.add_argument(
        "--feature_backend",
        default="torchvision-r3d18",
        choices=["torchvision-r3d18", "pixel"],
        help="Use pixel for quick smoke tests; use torchvision-r3d18 for paper numbers.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resize", type=int, default=112)
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--allow_untrained", action="store_true", help="Allow untrained R3D-18 if pretrained weights are unavailable.")
    return p.parse_args()


def read_video(path: Path) -> np.ndarray:
    frames = []
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            for frame in reader:
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=-1)
                frames.append(frame[..., :3].astype(np.uint8))
        finally:
            reader.close()
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("Video decoding requires imageio or opencv-python in the active Python env.") from exc
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.uint8))
        finally:
            cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def sample_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if len(frames) == num_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=max(1, num_frames))
    return frames[np.round(idx).astype(np.int64)]


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image

        out = []
        for frame in frames:
            out.append(np.asarray(Image.fromarray(frame).resize((size, size), Image.BICUBIC)))
        return np.stack(out, axis=0)
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("Frame resizing requires Pillow or opencv-python in the active Python env.") from exc
        return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC) for frame in frames], axis=0)


def pixel_feature(path: Path, num_frames: int) -> np.ndarray:
    frames = resize_frames(sample_frames(read_video(path), num_frames), 64).astype(np.float32) / 255.0
    diffs = np.abs(np.diff(frames, axis=0)) if len(frames) > 1 else np.zeros_like(frames[:1])
    color_mean = frames.mean(axis=(0, 1, 2))
    color_std = frames.std(axis=(0, 1, 2))
    diff_mean = diffs.mean(axis=(0, 1, 2))
    diff_std = diffs.std(axis=(0, 1, 2))
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(frames[..., channel], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    gray = frames.mean(axis=-1)
    pooled = resize_frames((gray.mean(axis=0) * 255).astype(np.uint8)[None, ...], 8)[0].reshape(-1).astype(np.float32) / 255.0
    return np.concatenate([color_mean, color_std, diff_mean, diff_std, *hist_parts, pooled]).astype(np.float64)


class R3D18FeatureExtractor:
    def __init__(self, device: str, resize: int, allow_untrained: bool):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.torch = torch
        self.F = F
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.resize = resize
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.model = r3d_18(weights=weights)
        except Exception as exc:
            if not allow_untrained:
                raise RuntimeError(
                    "Could not load torchvision R3D-18 pretrained weights. "
                    "Use --feature_backend pixel for a smoke test or --allow_untrained for debugging."
                ) from exc
            self.model = r3d_18(weights=None)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

    def __call__(self, path: Path, num_frames: int) -> np.ndarray:
        torch = self.torch
        frames = sample_frames(read_video(path), num_frames)
        clip = torch.from_numpy(frames).to(self.device, dtype=torch.float32) / 255.0
        clip = clip.permute(0, 3, 1, 2)
        clip = self.F.interpolate(clip, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        clip = (clip - self.mean) / self.std
        with torch.inference_mode():
            feat = self.model(clip).detach().cpu().numpy()[0]
        return feat.astype(np.float64)


def load_manifest_pairs(manifest_path: Path, labels_arg: str, max_videos: int) -> Tuple[List[VideoPairSet], List[Path]]:
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    cases = manifest.get("cases", [])
    labels = [x.strip() for x in labels_arg.split(",") if x.strip()]
    if not labels:
        found = []
        for case in cases:
            for label in (case.get("generated") or {}).keys():
                if label not in found:
                    found.append(label)
        labels = found

    pairs = []
    all_gt = []
    for case in cases:
        gt = case.get("ground_truth_future")
        if gt:
            all_gt.append(Path(gt))
    if max_videos > 0:
        all_gt = all_gt[:max_videos]

    for label in labels:
        gt_paths = []
        gen_paths = []
        for case in cases:
            generated = case.get("generated") or {}
            gt = case.get("ground_truth_future")
            gen = generated.get(label)
            if gt and gen:
                gt_paths.append(Path(gt))
                gen_paths.append(Path(gen))
        if max_videos > 0:
            gt_paths = gt_paths[:max_videos]
            gen_paths = gen_paths[:max_videos]
        if gt_paths and gen_paths:
            pairs.append(VideoPairSet(label=label, gt_paths=gt_paths, gen_paths=gen_paths))
    return pairs, all_gt


def parse_extra(extra_specs: Iterable[str], gt_paths: List[Path], max_videos: int) -> List[VideoPairSet]:
    out = []
    for spec in extra_specs:
        if "=" not in spec:
            raise ValueError(f"Bad --extra_videos={spec}; expected label=/path/to/dir")
        label, directory = spec.split("=", 1)
        label = label.strip()
        gen_dir = Path(directory).expanduser()
        gen_paths = sorted(path for path in gen_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTS)
        n = min(len(gt_paths), len(gen_paths))
        if max_videos > 0:
            n = min(n, max_videos)
        if n:
            out.append(VideoPairSet(label=label, gt_paths=gt_paths[:n], gen_paths=gen_paths[:n]))
    return out


def covariance(features: np.ndarray) -> np.ndarray:
    if len(features) <= 1:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.cov(features, rowvar=False)


def sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    sigma_a = covariance(a) + np.eye(a.shape[1]) * eps
    sigma_b = covariance(b) + np.eye(b.shape[1]) * eps
    diff = mu_a - mu_b
    sqrt_a = sqrtm_psd(sigma_a)
    covmean = sqrtm_psd(sqrt_a @ sigma_b @ sqrt_a)
    score = diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean)
    return float(max(score, 0.0))


def feature_matrix(paths: List[Path], extractor, num_frames: int) -> np.ndarray:
    feats = []
    for path in paths:
        feats.append(extractor(path, num_frames))
    return np.stack(feats, axis=0)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(path_png: Path, rows: List[Dict[str, object]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["label"]) for row in rows]
    fvd = np.asarray([float(row["fvd"]) for row in rows], dtype=np.float64)
    l2 = np.asarray([float(row["paired_feature_l2_mean"]) for row in rows], dtype=np.float64)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, fvd, width, color="#4C78A8", label="FVD-style ↓")
    bars2 = ax2.bar(x + width / 2, l2, width, color="#F58518", alpha=0.88, label="Paired feature L2 ↓")

    ax1.set_title(title)
    ax1.set_ylabel("FVD-style distance")
    ax2.set_ylabel("paired feature L2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.grid(axis="y", alpha=0.22, linewidth=0.8)

    max_fvd = max(float(fvd.max()), 1.0)
    max_l2 = max(float(l2.max()), 1.0)
    ax1.set_ylim(0, max_fvd * 1.18)
    ax2.set_ylim(0, max_l2 * 1.18)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max_fvd * 0.025, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max_l2 * 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#7A3E00")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, gt_pool = load_manifest_pairs(Path(args.manifest), args.labels, args.max_videos)
    pairs.extend(parse_extra(args.extra_videos, gt_pool, args.max_videos))
    if not pairs:
        raise RuntimeError("No generated/ground-truth video pairs found.")

    if args.feature_backend == "pixel":
        extractor = pixel_feature
        feature_note = "deterministic low-dimensional pixel/motion features; use for smoke tests only"
    else:
        extractor = R3D18FeatureExtractor(args.device, args.resize, args.allow_untrained)
        feature_note = "torchvision R3D-18 Kinetics feature distance"

    rows: List[Dict[str, object]] = []
    details = {"manifest": args.manifest, "feature_backend": args.feature_backend, "feature_note": feature_note, "labels": {}}
    for pair in pairs:
        gt_features = feature_matrix(pair.gt_paths, extractor, args.num_frames)
        gen_features = feature_matrix(pair.gen_paths, extractor, args.num_frames)
        fvd = frechet_distance(gt_features, gen_features)
        paired_l2 = np.linalg.norm(gt_features - gen_features, axis=1)
        row = {
            "label": pair.label,
            "n_videos": len(pair.gen_paths),
            "feature_backend": args.feature_backend,
            "fvd": f"{fvd:.8f}",
            "paired_feature_l2_mean": f"{float(paired_l2.mean()):.8f}",
            "paired_feature_l2_std": f"{float(paired_l2.std()):.8f}",
        }
        rows.append(row)
        details["labels"][pair.label] = {
            "gt_paths": [str(path) for path in pair.gt_paths],
            "gen_paths": [str(path) for path in pair.gen_paths],
            "metrics": row,
        }

    write_csv(output_dir / "video_distribution_metrics.csv", rows)
    plot_distribution(output_dir / "video_distribution_metrics.png", rows, "Distribution-Level Video Distance")
    with (output_dir / "video_distribution_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    print(f"metrics_csv={output_dir / 'video_distribution_metrics.csv'}")
    print(f"figure_png={output_dir / 'video_distribution_metrics.png'}")
    print(f"figure_pdf={output_dir / 'video_distribution_metrics.pdf'}")
    print(f"metrics_json={output_dir / 'video_distribution_metrics.json'}")


if __name__ == "__main__":
    main()
