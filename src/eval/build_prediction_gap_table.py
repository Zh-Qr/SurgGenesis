#!/usr/bin/env python3
"""Build predicted-vs-GT-latent prediction-gap tables for downstream tasks.

Input CSV format:

task,metric,method,value
DA,LPIPS,predicted,0.123
DA,LPIPS,gt_latent,0.100
Segmentation,mIoU,predicted,0.711
Segmentation,mIoU,gt_latent,0.760

The script writes a wide CSV that can be copied into the paper table.  It also
computes absolute and relative gaps.  Use --lower_is_better for metrics such as
LPIPS, FVD, RMSE, and MAE.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--predicted_name", default="predicted")
    p.add_argument("--gt_name", default="gt_latent")
    p.add_argument("--lower_is_better", default="LPIPS,FVD,RMSE,MAE,MSE")
    return p.parse_args()


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"task", "metric", "method", "value"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def direction(metric: str, lower_is_better: set[str]) -> str:
    m = metric.lower()
    for item in lower_is_better:
        if item and item.lower() in m:
            return "lower"
    return "higher"


def gap_values(pred: float, gt: float, metric: str, lower_is_better: set[str]) -> Tuple[float, float]:
    if direction(metric, lower_is_better) == "lower":
        abs_gap = pred - gt
    else:
        abs_gap = gt - pred
    rel_gap = abs_gap / max(abs(gt), 1e-12)
    return abs_gap, rel_gap


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["task", "metric"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lower = {x.strip() for x in args.lower_is_better.split(",") if x.strip()}
    rows = read_rows(Path(args.input_csv))

    values: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (row["task"].strip(), row["metric"].strip())
        method = row["method"].strip()
        values.setdefault(key, {})[method] = float(row["value"])

    out_rows: List[dict] = []
    for (task, metric), methods in sorted(values.items()):
        pred = methods.get(args.predicted_name)
        gt = methods.get(args.gt_name)
        out = {
            "task": task,
            "metric": metric,
            "direction": direction(metric, lower),
            args.predicted_name: "" if pred is None else f"{pred:.8f}",
            args.gt_name: "" if gt is None else f"{gt:.8f}",
            "prediction_gap_abs": "",
            "prediction_gap_rel": "",
            "prediction_gap_rel_percent": "",
        }
        if pred is not None and gt is not None:
            abs_gap, rel_gap = gap_values(pred, gt, metric, lower)
            out["prediction_gap_abs"] = f"{abs_gap:.8f}"
            out["prediction_gap_rel"] = f"{rel_gap:.8f}"
            out["prediction_gap_rel_percent"] = f"{100.0 * rel_gap:.2f}"
        for method, value in sorted(methods.items()):
            if method not in out:
                out[method] = f"{value:.8f}"
        out_rows.append(out)

    write_csv(output_dir / "prediction_gap_table.csv", out_rows)
    with (output_dir / "prediction_gap_table.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": args.input_csv,
                "predicted_name": args.predicted_name,
                "gt_name": args.gt_name,
                "lower_is_better": sorted(lower),
                "rows": out_rows,
            },
            f,
            indent=2,
        )
    print(f"table_csv={output_dir / 'prediction_gap_table.csv'}")
    print(f"table_json={output_dir / 'prediction_gap_table.json'}")


if __name__ == "__main__":
    main()
