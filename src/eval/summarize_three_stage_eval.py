#!/usr/bin/env python3
"""Aggregate VBench and SAM mask metrics across base/stage1/stage2/stage3."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


DIMENSION_LABELS = {
    "subject_consistency": "Subject",
    "background_consistency": "Background",
    "motion_smoothness": "Motion smooth",
    "dynamic_degree": "Dynamic",
    "aesthetic_quality": "Aesthetic",
    "imaging_quality": "Imaging",
    "mask_dimension_score": "Mask score",
    "mean_mask_dice": "Mask Dice",
    "mean_mask_iou": "Mask IoU",
    "mean_boundary_f1": "Boundary F1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--splits", default="base,stage1,stage2,stage3")
    p.add_argument("--output_dir", default="")
    p.add_argument("--title", default="Wan2.2-TI2V Three-Stage Evaluation")
    return p.parse_args()


def latest_result_json(run_dir: Path, split: str) -> Path | None:
    result_dir = run_dir / "vbench_results" / split
    candidates = sorted(result_dir.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def normalize_score(dimension: str, value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    value = float(value)
    if dimension == "imaging_quality" and value > 1.0:
        return value / 100.0
    return value


def load_vbench(path: Path) -> Dict[str, float]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {dimension: normalize_score(dimension, payload[0]) for dimension, payload in data.items()}


def load_mask_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [name for name in rows[0].keys() if name != "metric" and not name.endswith("_minus_base")] if rows else []
    out = {label: {} for label in labels}
    keep = {"mask_dimension_score", "mean_mask_dice", "mean_mask_iou", "mean_boundary_f1"}
    for row in rows:
        metric = row["metric"]
        if metric not in keep:
            continue
        for label in labels:
            if row.get(label):
                out[label][metric] = float(row[label])
    return out


def write_summary_csv(path: Path, labels: list[str], metrics: list[str], scores: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric"] + labels
        if "base" in labels:
            fieldnames += [f"{label}_minus_base" for label in labels if label != "base"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            row = {"metric": metric}
            base = scores.get("base", {}).get(metric)
            for label in labels:
                value = scores.get(label, {}).get(metric)
                row[label] = "" if value is None else f"{value:.8f}"
                if label != "base" and "base" in labels:
                    row[f"{label}_minus_base"] = "" if value is None or base is None else f"{value - base:+.8f}"
            writer.writerow(row)


def plot_summary(path: Path, title: str, labels: list[str], metrics: list[str], scores: Dict[str, Dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels_display = [DIMENSION_LABELS.get(metric, metric) for metric in metrics]
    x = np.arange(len(metrics))
    width = min(0.18, 0.72 / max(1, len(labels)))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.5), 6), dpi=160)
    for idx, label in enumerate(labels):
        values = [scores.get(label, {}).get(metric, 0.0) for metric in metrics]
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)])
    ax.set_title(title)
    ax.set_ylabel("score (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=20, ha="right")
    ax.set_ylim(0, max(1.08, max([scores.get(label, {}).get(metric, 0.0) for label in labels for metric in metrics] + [1.0]) * 1.12))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=len(labels))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    labels = [x.strip() for x in args.splits.split(",") if x.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "three_stage_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: Dict[str, Dict[str, float]] = {label: {} for label in labels}
    for label in labels:
        result_json = latest_result_json(run_dir, label)
        if result_json is not None:
            scores[label].update(load_vbench(result_json))
    mask_scores = load_mask_summary(run_dir / "mask_results" / "mask_summary_multistage.csv")
    for label, metrics in mask_scores.items():
        scores.setdefault(label, {}).update(metrics)

    metrics: list[str] = []
    for label in labels:
        for metric in scores.get(label, {}):
            if metric not in metrics:
                metrics.append(metric)

    write_summary_csv(output_dir / "three_stage_summary.csv", labels, metrics, scores)
    plot_summary(output_dir / "three_stage_summary.png", args.title, labels, metrics, scores)
    with (output_dir / "three_stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "labels": labels, "scores": scores}, f, indent=2)
    print(f"summary_csv={output_dir / 'three_stage_summary.csv'}")
    print(f"summary_png={output_dir / 'three_stage_summary.png'}")
    print(f"summary_json={output_dir / 'three_stage_summary.json'}")


if __name__ == "__main__":
    main()
