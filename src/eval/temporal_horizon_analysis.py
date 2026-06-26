#!/usr/bin/env python3
"""Measure temporal decay against real future clips from a generation manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--labels", default="", help="Comma-separated labels. Defaults to generated labels in manifest.")
    p.add_argument("--horizons", default="1,2,3,4,5", help="Comma-separated horizons in seconds.")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--max_frames_per_case", type=int, default=0)
    p.add_argument("--plot_metric", default="rmse", choices=["rmse", "psnr", "ssim"])
    p.add_argument("--title", default="Temporal Horizon Analysis")
    p.add_argument("--paper_style", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_video(path: str | Path) -> np.ndarray:
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


def align_frames(gt: np.ndarray, gen: np.ndarray, frame_stride: int, max_frames: int) -> Tuple[np.ndarray, np.ndarray, int]:
    offset = 1 if len(gen) == len(gt) + 1 else 0
    limit = min(len(gt), max(0, len(gen) - offset))
    indices = list(range(0, limit, max(1, frame_stride)))
    if max_frames > 0:
        indices = indices[:max_frames]
    return gt[indices], gen[[i + offset for i in indices]], offset


def frame_metrics(gt: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    gt_f = gt.astype(np.float64) / 255.0
    gen_f = gen.astype(np.float64) / 255.0
    diff = gt_f - gen_f
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = 99.0 if mse <= 1e-12 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))

    mu_x = float(gt_f.mean())
    mu_y = float(gen_f.mean())
    var_x = float(gt_f.var())
    var_y = float(gen_f.var())
    cov = float(((gt_f - mu_x) * (gen_f - mu_y)).mean())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
    return {"mse": mse, "rmse": rmse, "psnr": psnr, "ssim": float(ssim)}


def labels_from_manifest(cases: List[dict], labels_arg: str) -> List[str]:
    labels = [x.strip() for x in labels_arg.split(",") if x.strip()]
    if labels:
        return labels
    found = []
    for case in cases:
        for label in (case.get("generated") or {}).keys():
            if label not in found:
                found.append(label)
    return found


def mean_or_empty(values: List[float]) -> str:
    if not values:
        return ""
    return f"{float(np.mean(values)):.8f}"


def stats_or_empty(values: List[float]) -> Dict[str, str]:
    if not values:
        return {"mean": "", "std": "", "sem": ""}
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std / float(np.sqrt(max(1, len(arr))))
    return {"mean": f"{float(arr.mean()):.8f}", "std": f"{std:.8f}", "sem": f"{sem:.8f}"}


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_horizon(path_png: Path, rows: List[Dict[str, object]], labels: List[str], metric: str, title: str, paper_style: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if paper_style:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    colors = {
        "base": "#4C78A8",
        "stage1": "#F58518",
        "stage2": "#54A24B",
        "stage3": "#B279A2",
    }
    markers = {"base": "o", "stage1": "s", "stage2": "^", "stage3": "D"}
    for label in labels:
        xs = []
        ys = []
        sems = []
        for row in rows:
            if row["label"] != label:
                continue
            value = row.get(f"{metric}_mean", row.get(metric))
            if value in ("", None):
                continue
            xs.append(float(row["horizon_sec"]))
            ys.append(float(value))
            sem_value = row.get(f"{metric}_sem", "")
            sems.append(0.0 if sem_value in ("", None) else float(sem_value))
        if xs:
            color = colors.get(label, None)
            ax.plot(
                xs,
                ys,
                marker=markers.get(label, "o"),
                markersize=5,
                linewidth=2.2,
                color=color,
                label=label,
            )
            if any(v > 0 for v in sems):
                ys_arr = np.asarray(ys)
                sem_arr = np.asarray(sems)
                ax.fill_between(xs, ys_arr - sem_arr, ys_arr + sem_arr, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("prediction horizon (seconds)")
    ylabel = {"rmse": "RMSE ↓", "psnr": "PSNR ↑", "ssim": "SSIM ↑"}.get(metric, metric.upper())
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, ncol=min(4, max(1, len(labels))), loc="best")
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(encoding="utf-8") as f:
        manifest = json.load(f)
    cases = manifest.get("cases", [])
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    labels = labels_from_manifest(cases, args.labels)
    fps = float(manifest.get("fps") or 15)
    horizons = [float(x.strip()) for x in args.horizons.split(",") if x.strip()]

    frame_rows: List[Dict[str, object]] = []
    offsets: Dict[str, int] = {}
    for case in cases:
        case_name = case.get("case", "case")
        gt_path = case.get("ground_truth_future")
        if not gt_path:
            continue
        gt_all = read_video(gt_path)
        generated = case.get("generated") or {}
        for label in labels:
            gen_path = generated.get(label)
            if not gen_path:
                continue
            gen_all = read_video(gen_path)
            gt_frames, gen_frames, offset = align_frames(gt_all, gen_all, args.frame_stride, args.max_frames_per_case)
            offsets[label] = offset
            for idx, (gt_frame, gen_frame) in enumerate(zip(gt_frames, gen_frames)):
                metrics = frame_metrics(gt_frame, gen_frame)
                frame_rows.append(
                    {
                        "case": case_name,
                        "label": label,
                        "frame_index": idx,
                        "time_sec": f"{(idx + 1) * args.frame_stride / fps:.8f}",
                        **{key: f"{value:.8f}" for key, value in metrics.items()},
                    }
                )

    horizon_case_rows: List[Dict[str, object]] = []
    for label in labels:
        label_rows = [row for row in frame_rows if row["label"] == label]
        case_names = sorted({str(row["case"]) for row in label_rows})
        for horizon in horizons:
            for case_name in case_names:
                subset = [
                    row
                    for row in label_rows
                    if row["case"] == case_name and float(row["time_sec"]) <= horizon + 1e-9
                ]
                if not subset:
                    continue
                horizon_case_rows.append(
                    {
                        "case": case_name,
                        "label": label,
                        "horizon_sec": f"{horizon:.3f}",
                        "num_frames": len(subset),
                        "mse": mean_or_empty([float(row["mse"]) for row in subset]),
                        "rmse": mean_or_empty([float(row["rmse"]) for row in subset]),
                        "psnr": mean_or_empty([float(row["psnr"]) for row in subset]),
                        "ssim": mean_or_empty([float(row["ssim"]) for row in subset]),
                    }
                )

    horizon_rows: List[Dict[str, object]] = []
    for label in labels:
        for horizon in horizons:
            subset = [
                row
                for row in horizon_case_rows
                if row["label"] == label and abs(float(row["horizon_sec"]) - horizon) < 1e-9
            ]
            metric_stats = {metric: stats_or_empty([float(row[metric]) for row in subset if row.get(metric)]) for metric in ("mse", "rmse", "psnr", "ssim")}
            horizon_rows.append(
                {
                    "label": label,
                    "horizon_sec": f"{horizon:.3f}",
                    "num_cases": len(subset),
                    "num_frames": sum(int(row["num_frames"]) for row in subset),
                    "mse_mean": metric_stats["mse"]["mean"],
                    "mse_std": metric_stats["mse"]["std"],
                    "mse_sem": metric_stats["mse"]["sem"],
                    "rmse_mean": metric_stats["rmse"]["mean"],
                    "rmse_std": metric_stats["rmse"]["std"],
                    "rmse_sem": metric_stats["rmse"]["sem"],
                    "psnr_mean": metric_stats["psnr"]["mean"],
                    "psnr_std": metric_stats["psnr"]["std"],
                    "psnr_sem": metric_stats["psnr"]["sem"],
                    "ssim_mean": metric_stats["ssim"]["mean"],
                    "ssim_std": metric_stats["ssim"]["std"],
                    "ssim_sem": metric_stats["ssim"]["sem"],
                }
            )

    write_rows(output_dir / "horizon_frame_metrics.csv", frame_rows)
    write_rows(output_dir / "horizon_case_metrics.csv", horizon_case_rows)
    write_rows(output_dir / "horizon_summary.csv", horizon_rows)
    plot_horizon(output_dir / "horizon_curve.png", horizon_rows, labels, args.plot_metric, args.title, args.paper_style)
    with (output_dir / "horizon_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": args.manifest,
                "fps": fps,
                "labels": labels,
                "horizons": horizons,
                "generated_frame_offsets": offsets,
                "summary_csv": str(output_dir / "horizon_summary.csv"),
                "case_csv": str(output_dir / "horizon_case_metrics.csv"),
                "figure_png": str(output_dir / "horizon_curve.png"),
                "figure_pdf": str(output_dir / "horizon_curve.pdf"),
            },
            f,
            indent=2,
        )
    print(f"frame_csv={output_dir / 'horizon_frame_metrics.csv'}")
    print(f"case_csv={output_dir / 'horizon_case_metrics.csv'}")
    print(f"summary_csv={output_dir / 'horizon_summary.csv'}")
    print(f"figure_png={output_dir / 'horizon_curve.png'}")
    print(f"figure_pdf={output_dir / 'horizon_curve.pdf'}")


if __name__ == "__main__":
    main()
