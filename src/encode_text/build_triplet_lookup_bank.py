#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a lookup embedding bank from CholecT50 triplet annotations.

Output .pt format is compatible with LookupEmbedProvider in unified trainer:
{
  "context_bank": [context_1, context_2, ...],
  "context_null": context_null,
  "meta": {...}
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "backbone", "Wan2.1"))
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan  # type: ignore
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES  # type: ignore

TensorLike = Union[torch.Tensor, Sequence[torch.Tensor]]

INSTRUMENTS = [
    "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"
]

TARGETS = [
    "gallbladder", "cystic_plate", "cystic_duct", "cystic_artery", "cystic_pedicle",
    "blood_vessel", "fluid", "abdominal_wall_cavity", "liver", "adhesion",
    "omentum", "peritoneum", "gut", "specimen_bag", "null_target",
]

VERBS = [
    "grasp", "retract", "dissect", "coagulate", "clip",
    "cut", "aspirate", "irrigate", "pack", "null_verb",
]


def parse_args():
    p = argparse.ArgumentParser("Build triplet lookup embedding bank for stage2 lookup mode")

    p.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", type=str, required=True)

    p.add_argument("--base_path", type=str, required=True, help="Root containing Training/Validation/Testing")
    p.add_argument("--datasets", type=str, default="Training", help="Comma-separated folders under base_path")
    p.add_argument("--video_prefix", type=str, default="VID", help="Video folder prefix")
    p.add_argument("--annotation_glob", type=str, default="*.json", help="Annotation json glob in each video folder")
    p.add_argument("--labels_root", type=str, default="", help="Optional labels directory containing VIDxx.json files")

    p.add_argument("--prompt_template", type=str, default="triplet", choices=["triplet", "verbose"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--n_prompt", type=str, default="", help="Negative prompt, default to model builtin")

    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _safe_name(items: List[str], idx: Any, prefix: str) -> str:
    try:
        i = int(idx)
        if 0 <= i < len(items):
            return items[i]
        return f"{prefix}_{i}"
    except Exception:
        return f"unknown_{prefix}"


def _triplet_key(obj: Dict[str, Any]) -> str:
    ins = _safe_name(INSTRUMENTS, obj.get("instrument"), "instrument")
    verb = _safe_name(VERBS, obj.get("verb"), "verb")
    target = _safe_name(TARGETS, obj.get("target"), "target")
    return f"{ins}|{verb}|{target}"


def _build_prompt_from_key(key: str, mode: str) -> str:
    ins, verb, target = key.split("|", 2)
    if mode == "triplet":
        return f"laparoscopic scene with {ins} {verb} {target}"
    return (
        "Laparoscopic cholecystectomy endoscopic view. "
        f"Visible surgical action: {ins} performs {verb} on {target}. "
        "Generate realistic motion and anatomically consistent tissue interaction."
    )


def _find_annotation_files(
    base_path: Path,
    datasets: List[str],
    video_prefix: str,
    annotation_glob: str,
    labels_root: Optional[Path],
) -> List[Path]:
    files: List[Path] = []
    if labels_root is not None and labels_root.exists():
        files.extend(sorted(labels_root.glob(annotation_glob)))
        if files:
            return files

    for ds in datasets:
        ds_path = base_path / ds
        if not ds_path.exists():
            continue
        for video_dir in sorted(ds_path.iterdir()):
            if not video_dir.is_dir() or not video_dir.name.startswith(video_prefix):
                continue
            candidates = sorted(video_dir.glob(annotation_glob))
            if candidates:
                files.append(candidates[0])
    return files


def _collect_unique_triplet_keys(annotation_files: List[Path]) -> List[str]:
    keys = set()
    for ap in annotation_files:
        try:
            with ap.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            ann = obj.get("annotations", {})
            if not isinstance(ann, dict):
                continue
            for _, frame_objs in ann.items():
                if not isinstance(frame_objs, list):
                    continue
                for o in frame_objs:
                    if isinstance(o, dict):
                        keys.add(_triplet_key(o))
        except Exception:
            continue
    return sorted(keys)


def _as_tuple(x: TensorLike) -> Tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        return (x,)
    return tuple(x)


def _normalize_ctx(x: TensorLike) -> List[torch.Tensor]:
    xs = _as_tuple(x)
    return [t.detach().to(device="cpu", dtype=torch.float32) for t in xs]


def _slice_item(ctx_batch: TensorLike, i: int) -> List[torch.Tensor]:
    if isinstance(ctx_batch, torch.Tensor):
        return [ctx_batch[i:i + 1].detach().to(device="cpu", dtype=torch.float32)]
    return [t[i:i + 1].detach().to(device="cpu", dtype=torch.float32) for t in ctx_batch]


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    assert args.task in WAN_CONFIGS
    assert args.size in SIZE_CONFIGS
    assert args.size in SUPPORTED_SIZES[args.task]

    base_path = Path(args.base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"base_path not found: {base_path}")

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    labels_root = Path(args.labels_root) if args.labels_root.strip() else None
    annotation_files = _find_annotation_files(base_path, datasets, args.video_prefix, args.annotation_glob, labels_root)
    if not annotation_files:
        raise RuntimeError("No annotation files found. Check base_path/datasets/video_prefix/annotation_glob")

    keys = _collect_unique_triplet_keys(annotation_files)
    if not keys:
        raise RuntimeError("No triplet keys parsed from annotations")

    prompts = [_build_prompt_from_key(k, args.prompt_template) for k in keys]
    logging.info(f"[lookup-bank] annotation_files={len(annotation_files)}")
    logging.info(f"[lookup-bank] unique_triplets={len(keys)}")

    cfg = WAN_CONFIGS[args.task]
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    pipe = wan.WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    pipe.text_encoder.model.to(device)

    n_prompt = args.n_prompt or pipe.sample_neg_prompt
    with torch.inference_mode():
        ctx_null = pipe.text_encoder([n_prompt], device)
    context_null = _normalize_ctx(ctx_null)

    context_bank: List[List[torch.Tensor]] = []
    bs = max(1, int(args.batch_size))
    for start in range(0, len(prompts), bs):
        batch_prompts = prompts[start:start + bs]
        with torch.inference_mode():
            ctx_batch = pipe.text_encoder(batch_prompts, device)
        for i in range(len(batch_prompts)):
            context_bank.append(_slice_item(ctx_batch, i))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context_bank": context_bank,
        "context_null": context_null,
        "meta": {
            "task": args.task,
            "size": args.size,
            "base_path": str(base_path),
            "datasets": datasets,
            "annotation_files": len(annotation_files),
            "unique_triplets": len(keys),
            "prompt_template": args.prompt_template,
        },
        "keys": keys,
        "prompts": prompts,
    }
    torch.save(payload, str(out_path))

    logging.info(f"[lookup-bank] saved -> {out_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
