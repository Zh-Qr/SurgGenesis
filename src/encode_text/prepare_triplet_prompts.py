#!/usr/bin/env python3
"""Build frame-level prompts from CholecT50 triplet annotations.

Output format is compatible with batch_encode_text_embeds.py:
{
  "1": "prompt text for frame 1",
  "2": "prompt text for frame 2",
  ...
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

INSTRUMENTS = [
    "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"
]

PHASES = {
    1: "preparation",
    2: "calot triangle dissection",
    3: "clipping and cutting",
    4: "gallbladder dissection",
    5: "gallbladder packaging",
    6: "cleaning and coagulation",
    7: "gallbladder extraction",
}

PHASES = {
    1: "preparation",
    2: "calot triangle dissection",
    3: "clipping and cutting",
    4: "gallbladder dissection",
    5: "gallbladder packaging",
    6: "cleaning and coagulation",
    7: "gallbladder extraction",
}

TARGETS = [
    "gallbladder", "cystic_plate", "cystic_duct", "cystic_artery", "cystic_pedicle",
    "blood_vessel", "fluid", "abdominal_wall_cavity", "liver", "adhesion",
    "omentum", "peritoneum", "gut", "specimen_bag", "null_target",
]

VERBS = [
    "grasp", "retract", "dissect", "coagulate", "clip",
    "cut", "aspirate", "irrigate", "pack", "null_verb",
]


def safe_name(items: List[str], idx: Any, default_prefix: str) -> str:
    try:
        i = int(idx)
        if 0 <= i < len(items):
            return items[i]
        return f"unknown {default_prefix}"
    except Exception:
        return f"unknown {default_prefix}"


def safe_phase(phase_idx: Any) -> str:
    try:
        return PHASES.get(int(phase_idx), "unknown phase")
    except Exception:
        return "unknown phase"


def _row_to_triplet_dicts(row: List[Any]) -> List[Dict[str, Any]]:
    """Convert one compact annotation row to one or more triplet dicts.

    CholecT50 labels can store each frame as nested lists of ints, e.g.
    [[phase, ins1, verb1, target1, ..., ins2, verb2, target2, ...]].
    """
    out: List[Dict[str, Any]] = []
    if not row:
        return out

    phase = row[0] if len(row) > 0 else None

    # Common compact layout in CholecT50 JSONs.
    for start in (1, 7):
        if len(row) <= start + 2:
            continue
        ins, verb, target = row[start], row[start + 1], row[start + 2]
        if int(ins) == -1 and int(verb) == -1 and int(target) == -1:
            continue
        out.append({"phase": phase, "instrument": ins, "verb": verb, "target": target})

    # Fallback: if no compact slots were detected but row still has 4+ entries.
    if not out and len(row) >= 4:
        out.append({"phase": row[0], "instrument": row[1], "verb": row[2], "target": row[3]})

    return out


def normalize_objects(raw_objects: Any) -> List[Dict[str, Any]]:
    """Normalize annotation objects to a list of dict records.

    Supports both styles:
    - [{"phase":..., "instrument":..., "verb":..., "target":...}, ...]
    - [[phase, ...], [phase, ...]] or [[[phase, ...]], ...]
    """
    out: List[Dict[str, Any]] = []

    if raw_objects is None:
        return out

    if isinstance(raw_objects, dict):
        out.append(raw_objects)
        return out

    if not isinstance(raw_objects, list):
        return out

    for item in raw_objects:
        if isinstance(item, dict):
            out.append(item)
            continue

        if isinstance(item, list):
            # Nested compact rows, e.g. [[...], [...]]
            if item and isinstance(item[0], list):
                for row in item:
                    if isinstance(row, list):
                        out.extend(_row_to_triplet_dicts(row))
            else:
                out.extend(_row_to_triplet_dicts(item))

    return out


def _entity_phrase(name: str, kind: str) -> str:
    if name.startswith("unknown "):
        return f"an {name}"
    if kind == "target":
        return f"the {name.replace('_', ' ')}"
    return f"the {name}"


def _triplet_to_phrase(obj: Dict[str, Any]) -> str:
    ins = safe_name(INSTRUMENTS, obj.get("instrument"), "instrument")
    verb = safe_name(VERBS, obj.get("verb"), "verb")
    target = safe_name(TARGETS, obj.get("target"), "target")

    ins_text = _entity_phrase(ins, "instrument")
    target_text = _entity_phrase(target, "target")

    if verb.startswith("unknown "):
        return f"{ins_text} performing {verb} on {target_text}"
    return f"{ins_text} {verb}ing {target_text}"


def build_prompt(frame_id: int, objects: List[Dict[str, Any]], future_horizon: int) -> str:
    norm_objects = normalize_objects(objects)

    if not norm_objects:
        return (
            f"Endoscopic laparoscopic frame {frame_id}. "
            f"No surgical instrument is clearly visible. Predict the next {future_horizon} frames with stable camera motion, "
            "continuous tissue dynamics, and anatomically plausible scene evolution."
        )

    phase = safe_phase(norm_objects[0].get("phase"))
    actions = [_triplet_to_phrase(obj) for obj in norm_objects]
    action_text = "; ".join(actions)

    return (
        f"Laparoscopic cholecystectomy, phase: {phase}. "
        f"At frame {frame_id}, the scene shows {action_text}. "
        f"Predict the next {future_horizon} frames with smooth instrument motion, stable endoscopic viewpoint, "
        "and anatomically consistent tool-tissue interaction."
    )


def process_annotation_file(json_path: Path, video_dir: Path, output_suffix: str, future_horizon: int) -> Path:
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    ann = meta.get("annotations", {})
    prompts: Dict[str, str] = {}
    for key in sorted(ann.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
        try:
            frame_id = int(key)
        except Exception:
            continue
        objs = ann.get(key) or []
        prompts[str(frame_id)] = build_prompt(frame_id, objs, future_horizon)

    out_path = json_path.with_name(f"{json_path.stem}{output_suffix}.json")
    if video_dir != json_path.parent:
        video_dir.mkdir(parents=True, exist_ok=True)
        out_path = video_dir / f"{json_path.stem}{output_suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    p = argparse.ArgumentParser("Prepare CholecT50 frame prompts from triplet annotations")
    p.add_argument("--base_path", type=str, default="", help="Legacy root path containing Training/Testing/Validation")
    p.add_argument("--labels_root", type=str, default="", help="Folder containing VIDxx.json label files")
    p.add_argument("--videos_root", type=str, default="", help="Folder containing VIDxx frame folders")
    p.add_argument("--datasets", type=str, default="Training", help="Legacy dataset subset for base_path mode")
    p.add_argument("--video_prefix", type=str, default="VID", help="Video folder prefix")
    p.add_argument("--output_suffix", type=str, default="_prompts", help="Suffix appended before .json")
    p.add_argument("--future_horizon", type=int, default=5)
    args = p.parse_args()

    labels_root = Path(args.labels_root) if args.labels_root.strip() else None
    videos_root = Path(args.videos_root) if args.videos_root.strip() else None
    root = Path(args.base_path) if args.base_path.strip() else None
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    if labels_root is None and root is None:
        raise ValueError("Provide --labels_root or --base_path")

    if labels_root is None and root is not None:
        labels_root = root
    if videos_root is None and root is not None:
        videos_root = root

    assert labels_root is not None
    assert videos_root is not None

    generated = 0

    # New layout: labels_root/VIDxx.json + videos_root/VIDxx/
    if labels_root.exists() and any(labels_root.glob("*.json")):
        for src in sorted(labels_root.glob("*.json")):
            if not src.stem.startswith(args.video_prefix):
                continue
            video_dir = videos_root / src.stem
            out = process_annotation_file(src, video_dir, args.output_suffix, args.future_horizon)
            print(f"[ok] {src} -> {out}")
            generated += 1
    else:
        # Legacy layout: root/Training/VIDxx/*.json
        for ds in datasets:
            ds_path = labels_root / ds
            if not ds_path.exists():
                continue
            for video_dir in sorted(ds_path.iterdir()):
                if not video_dir.is_dir() or not video_dir.name.startswith(args.video_prefix):
                    continue
                candidates = sorted(video_dir.glob("*.json"))
                if not candidates:
                    continue
                src = candidates[0]
                out = process_annotation_file(src, video_dir, args.output_suffix, args.future_horizon)
                print(f"[ok] {src} -> {out}")
                generated += 1

    print(f"Done. generated_prompt_files={generated}")


if __name__ == "__main__":
    main()
