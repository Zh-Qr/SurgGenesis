#!/usr/bin/env python3
"""Create few-shot / long-tail augmentation splits for downstream validation.

The script is intentionally format-light.  It reads a CSV with at least:

sample_id,class

Optional columns:

source,path,split,is_generated

It writes real-only and real+generated augmentation plans at several shots per
class.  This is the table scaffold needed for the paper TODO about data
scarcity and class imbalance.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shots", default="1,2,4,8,16")
    p.add_argument("--class_column", default="class")
    p.add_argument("--sample_column", default="sample_id")
    p.add_argument("--generated_column", default="is_generated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_per_class", type=int, default=0, help="If >0, add generated samples up to this count per class.")
    return p.parse_args()


def read_samples(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_generated(row: dict, column: str) -> bool:
    value = str(row.get(column, "")).strip().lower()
    return value in {"1", "true", "yes", "y", "generated", "synthetic"}


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(Path(args.samples_csv))
    if not samples:
        raise ValueError(f"No rows in {args.samples_csv}")
    if args.class_column not in samples[0] or args.sample_column not in samples[0]:
        raise ValueError(f"Input must contain {args.sample_column!r} and {args.class_column!r} columns.")

    real_by_class: Dict[str, List[dict]] = defaultdict(list)
    gen_by_class: Dict[str, List[dict]] = defaultdict(list)
    for row in samples:
        cls = row[args.class_column]
        if is_generated(row, args.generated_column):
            gen_by_class[cls].append(row)
        else:
            real_by_class[cls].append(row)

    for group in (real_by_class, gen_by_class):
        for rows in group.values():
            rng.shuffle(rows)

    shots = [int(x.strip()) for x in args.shots.split(",") if x.strip()]
    summary_rows = []
    split_rows = []
    all_classes = sorted(set(real_by_class) | set(gen_by_class))
    for shot in shots:
        for cls in all_classes:
            real_rows = real_by_class.get(cls, [])
            gen_rows = gen_by_class.get(cls, [])
            real_selected = real_rows[: min(shot, len(real_rows))]

            target = args.target_per_class if args.target_per_class > 0 else max(shot, len(real_selected))
            needed_gen = max(0, target - len(real_selected))
            gen_selected = gen_rows[: min(needed_gen, len(gen_rows))]

            for row in real_selected:
                split_rows.append(
                    {
                        "experiment": f"real_only_{shot}shot",
                        "class": cls,
                        "sample_id": row[args.sample_column],
                        "source": row.get("source", "real"),
                        "path": row.get("path", ""),
                        "is_generated": "0",
                    }
                )
                split_rows.append(
                    {
                        "experiment": f"real_plus_generated_{shot}shot",
                        "class": cls,
                        "sample_id": row[args.sample_column],
                        "source": row.get("source", "real"),
                        "path": row.get("path", ""),
                        "is_generated": "0",
                    }
                )
            for row in gen_selected:
                split_rows.append(
                    {
                        "experiment": f"real_plus_generated_{shot}shot",
                        "class": cls,
                        "sample_id": row[args.sample_column],
                        "source": row.get("source", "generated"),
                        "path": row.get("path", ""),
                        "is_generated": "1",
                    }
                )

            summary_rows.append(
                {
                    "class": cls,
                    "shot": shot,
                    "real_available": len(real_rows),
                    "generated_available": len(gen_rows),
                    "real_only_count": len(real_selected),
                    "real_plus_generated_count": len(real_selected) + len(gen_selected),
                    "generated_added": len(gen_selected),
                }
            )

    write_csv(output_dir / "long_tail_split_plan.csv", split_rows)
    write_csv(output_dir / "long_tail_class_summary.csv", summary_rows)
    with (output_dir / "long_tail_plan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples_csv": args.samples_csv,
                "shots": shots,
                "seed": args.seed,
                "target_per_class": args.target_per_class,
                "num_classes": len(all_classes),
                "split_plan_csv": str(output_dir / "long_tail_split_plan.csv"),
                "class_summary_csv": str(output_dir / "long_tail_class_summary.csv"),
            },
            f,
            indent=2,
        )
    print(f"split_plan_csv={output_dir / 'long_tail_split_plan.csv'}")
    print(f"class_summary_csv={output_dir / 'long_tail_class_summary.csv'}")


if __name__ == "__main__":
    main()
