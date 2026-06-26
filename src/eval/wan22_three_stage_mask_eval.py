#!/usr/bin/env python3
"""SAM3-adapter mask evaluation for base/stage1/stage2/stage3 generated videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


WAN_SAM_ROOT = Path(os.environ.get("WAN_SAM_ROOT", "/home/xuecheng/WAN_SAM"))
EVAL_DIR = WAN_SAM_ROOT / "wan22_ti2v_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from sam3_video_mask_eval import (  # noqa: E402
    SamMasker,
    compare_masks,
    read_video,
    segment_frames,
    select_aligned_frames,
    summarize_rows,
    write_csv,
    write_summary_compare,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--repo_root", default=str(WAN_SAM_ROOT))
    p.add_argument("--sam_checkpoint", default="/mnt/data3/zhongqirui/sam/sam3.pt")
    p.add_argument(
        "--adapter_checkpoint",
        default=str(WAN_SAM_ROOT / "adapter" / "outputs" / "endovis18_tool_sam3_adapter" / "best_adapter.pt"),
    )
    p.add_argument("--adapter_targets", default=None)
    p.add_argument("--adapter_layers", default=None)
    p.add_argument("--adapter_dim", type=int, default=None)
    p.add_argument("--adapter_dropout", type=float, default=None)
    p.add_argument("--adapter_init_scale", type=float, default=None)
    p.add_argument("--prompt", default="surgical instrument")
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--max_frames_per_case", type=int, default=0)
    p.add_argument("--boundary_tolerance", type=int, default=2)
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--save_mask_videos", action="store_true")
    p.add_argument("--splits", default="", help="Optional comma-separated generated labels. Defaults to all labels.")
    return p.parse_args()


def write_multi_summary(path: Path, summaries: Dict[str, Dict[str, float]]) -> None:
    metrics = [
        "mask_dimension_score",
        "mean_mask_dice",
        "mean_mask_iou",
        "mean_boundary_f1",
        "mean_area_abs_error",
        "mean_edge_complexity_abs_error",
        "mean_component_abs_error",
        "mean_sam_score",
        "num_frames",
    ]
    rows = []
    labels = list(summaries)
    baseline = summaries.get("base", {})
    for metric in metrics:
        row = {"metric": metric}
        for label in labels:
            row[label] = f"{summaries[label].get(metric, 0.0):.8f}"
            if label != "base":
                row[f"{label}_minus_base"] = f"{summaries[label].get(metric, 0.0) - baseline.get(metric, 0.0):+.8f}"
        rows.append(row)
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    cases = manifest.get("cases", [])
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    explicit_splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    all_labels = ["base"] + [item["label"] for item in manifest.get("stages", [])]
    labels = explicit_splits or all_labels

    masker = SamMasker(args)
    rows: List[Dict] = []
    for case_idx, case in enumerate(cases):
        case_name = case.get("case", f"case_{case_idx:02d}")
        gt_path = case.get("ground_truth_future")
        if not gt_path:
            continue
        gt_frames_all = read_video(gt_path)
        generated = case.get("generated", {})
        split_frames = {}
        offsets = {}
        for label in labels:
            gen_path = generated.get(label)
            if not gen_path:
                continue
            gen_frames_all = read_video(gen_path)
            gt_frames, gen_frames, offset = select_aligned_frames(
                gt_frames_all,
                gen_frames_all,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames_per_case,
            )
            split_frames[label] = (gt_frames, gen_frames)
            offsets[label] = offset
        if not split_frames:
            continue

        gt_reference_frames = next(iter(split_frames.values()))[0]
        case_dir = output_dir / "cases" / case_name
        print(f"[mask] {case_name} frames={len(gt_reference_frames)} labels={','.join(split_frames)}", flush=True)
        gt_seg = segment_frames(
            masker,
            gt_reference_frames,
            case_dir,
            args.save_overlays,
            "ground_truth_future",
            progress_desc=f"{case_name} gt",
        )

        for label, (_, gen_frames) in split_frames.items():
            gen_seg = segment_frames(
                masker,
                gen_frames,
                case_dir,
                args.save_overlays,
                label,
                progress_desc=f"{case_name} {label}",
            )
            for frame_idx, (pred_mask, gt_mask) in enumerate(zip(gen_seg.masks, gt_seg.masks)):
                metrics = compare_masks(pred_mask, gt_mask, args.boundary_tolerance)
                rows.append(
                    {
                        "case": case_name,
                        "split": label,
                        "frame_index": frame_idx,
                        "generated_frame_offset": offsets.get(label, 0),
                        "gt_mask": str(gt_seg.mask_paths[frame_idx]),
                        "pred_mask": str(gen_seg.mask_paths[frame_idx]),
                        "sam_score": gen_seg.scores[frame_idx],
                        **metrics,
                    }
                )

    summaries = {label: summarize_rows(rows, label) for label in labels}
    report = {
        "manifest": str(manifest_path),
        "sam_checkpoint": args.sam_checkpoint,
        "adapter_checkpoint": args.adapter_checkpoint,
        "prompt": args.prompt,
        "labels": labels,
        "num_cases": len(cases),
        "summaries": summaries,
        "note": "Ground-truth masks are pseudo labels generated by the same SAM3 adapter on real future frames.",
    }

    write_csv(output_dir / "mask_frame_metrics.csv", rows)
    write_multi_summary(output_dir / "mask_summary_multistage.csv", summaries)
    if "base" in summaries:
        for label in labels:
            if label == "base":
                continue
            pair_dir = output_dir / f"compare_base_{label}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            write_summary_compare(pair_dir / "mask_summary_compare.csv", summaries.get("base", {}), summaries.get(label, {}))
    with (output_dir / "mask_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
