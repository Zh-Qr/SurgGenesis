#!/usr/bin/env python3
"""Generate base/stage1/stage2/stage3 Wan2.2-TI2V comparison clips.

This reuses the same DiffSynth pipeline and data conventions as
WAN_SAM/wan22_ti2v_eval/generate_compare.py, but evaluates multiple LoRA
checkpoints on exactly the same slices and seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import torch


DIFFSYNTH_DIR = os.environ.get("DIFFSYNTH_DIR")
if DIFFSYNTH_DIR:
    sys.path.insert(0, DIFFSYNTH_DIR)

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import VideoData, save_video  # noqa: E402


DEFAULT_PROMPT = (
    "realistic laparoscopic cholecystectomy video, intraoperative endoscopic view, "
    "surgical instruments interacting with soft tissue, physically plausible motion, "
    "natural camera movement, stable surgical lighting"
)
DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, underexposed, blurry, low quality, static image, text, watermark, "
    "cartoon, painting, unrealistic motion, distorted anatomy, duplicated instruments"
)


def parse_stage_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Bad --stage_lora value: {text}. Expected label=/path/to/checkpoint.safetensors")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty stage label in --stage_lora={text}")
    ckpt = Path(path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found for {label}: {ckpt}")
    return label, ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="/mnt/data3/zhongqirui/WAN2_2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--stage_lora", action="append", default=[], help="label=/path/to/step-N.safetensors")
    p.add_argument("--num_cases", type=int, default=8)
    p.add_argument("--cond_frames", type=int, default=8)
    p.add_argument("--case_stride", type=int, default=7)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lora_alpha", type=float, default=1.5)
    p.add_argument("--skip_base", action="store_true")
    p.add_argument(
        "--only_labels",
        default="",
        help="Comma/space separated labels to generate, e.g. stage3. Empty means all labels.",
    )
    p.add_argument(
        "--skip_existing_outputs",
        action="store_true",
        help="Reuse existing <label>_generated.mp4 files and only generate missing labels.",
    )
    return p.parse_args()


def model_configs(model_dir: Path) -> list[ModelConfig]:
    shards = sorted(model_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No diffusion_pytorch_model-*.safetensors under {model_dir}")
    return [
        ModelConfig([str(path) for path in shards]),
        ModelConfig(str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(str(model_dir / "Wan2.2_VAE.pth")),
    ]


def read_metadata(metadata_csv: str, dataset_base_path: str) -> list[dict[str, str]]:
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in metadata CSV: {metadata_csv}")
    for row in rows:
        video = row["video"]
        row["video_path"] = video if os.path.isabs(video) else os.path.join(dataset_base_path, video)
    return rows


def choose_cases(rows: list[dict[str, str]], num_cases: int, case_stride: int) -> list[dict[str, str]]:
    target = min(max(1, num_cases), len(rows))
    stride = max(1, case_stride)
    selected: list[int] = []
    seen: set[int] = set()
    for idx in range(0, len(rows), stride):
        if len(selected) >= target:
            break
        selected.append(idx)
        seen.add(idx)
    for idx in range(len(rows)):
        if len(selected) >= target:
            break
        if idx not in seen:
            selected.append(idx)
    return [rows[idx] for idx in selected]


def sample_frames(video_path: str, height: int, width: int, cond_frames: int, num_frames: int):
    video = VideoData(video_path, height=height, width=width)
    needed = cond_frames + max(1, num_frames - 1)
    if len(video) < needed:
        raise ValueError(f"Video too short: {video_path}, frames={len(video)}, needed={needed}")
    condition = [video[i] for i in range(cond_frames)]
    future = [video[i] for i in range(cond_frames, needed)]
    return condition, future, condition[-1]


def copy_for_vbench(src: Path, dst_dir: Path, case_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{case_name}.mp4"
    shutil.copy2(src, dst)
    return dst


def parse_label_filter(text: str) -> set[str]:
    return {part.strip() for part in text.replace(",", " ").split() if part.strip()}


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_one(pipe, prompt: str, negative_prompt: str, input_image, args: argparse.Namespace):
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
    )


def load_pipe(model_dir: Path):
    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs(model_dir),
        tokenizer_config=ModelConfig(str(tokenizer_dir)),
    )
    return pipe


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    vbench_dir = output_dir / "vbench_videos"
    cases_dir.mkdir(parents=True, exist_ok=True)

    stages = [parse_stage_spec(item) for item in args.stage_lora]
    stage_labels = [label for label, _ in stages]
    all_labels = ([] if args.skip_base else ["base"]) + stage_labels
    only_labels = parse_label_filter(args.only_labels)
    unknown_labels = sorted(only_labels - set(all_labels))
    if unknown_labels:
        raise ValueError(f"Unknown --only_labels values: {unknown_labels}; available labels={all_labels}")
    active_labels = only_labels or set(all_labels)

    rows = read_metadata(args.metadata_csv, args.dataset_base_path)
    cases = choose_cases(rows, args.num_cases, args.case_stride)

    prompt_maps: dict[str, dict[str, str]] = {label: {} for label in all_labels}
    model_dir = Path(args.model_dir)
    manifest = {
        "model_dir": str(model_dir),
        "metadata_csv": args.metadata_csv,
        "dataset_base_path": args.dataset_base_path,
        "num_cases": args.num_cases,
        "cond_frames": args.cond_frames,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "lora_alpha": args.lora_alpha,
        "only_labels": sorted(only_labels) if only_labels else "all",
        "skip_existing_outputs": args.skip_existing_outputs,
        "stages": [{"label": label, "lora_path": str(path)} for label, path in stages],
        "cases": [],
    }
    manifest_path = output_dir / "manifest.json"

    def write_manifest() -> None:
        write_json(manifest_path, manifest)

    def write_case_metadata(case_dir: Path, record: dict) -> None:
        write_json(case_dir / "case_metadata.json", record)

    def register_existing_generated(record: dict, label: str) -> bool:
        case_name = record["case"]
        case_dir = cases_dir / case_name
        out_path = case_dir / f"{label}_generated.mp4"
        if not out_path.is_file():
            return False
        vbench_path = copy_for_vbench(out_path, vbench_dir / label, case_name)
        prompt_maps[label][str(vbench_path.resolve())] = record["prompt"]
        record.setdefault("generated", {})[label] = str(out_path)
        return True

    for case_id, row in enumerate(cases):
        case_name = f"case_{case_id:02d}"
        case_dir = cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        video_path = row["video_path"]
        prompt = row.get("prompt") or args.prompt
        print(f"[case] {case_name} source={video_path}", flush=True)
        condition, future, input_image = sample_frames(
            video_path,
            height=args.height,
            width=args.width,
            cond_frames=args.cond_frames,
            num_frames=args.num_frames,
        )
        condition_path = case_dir / "condition.mp4"
        gt_path = case_dir / "ground_truth_future.mp4"
        if not condition_path.exists():
            save_video(condition, str(condition_path), fps=args.fps, quality=5)
        if not gt_path.exists():
            save_video(future, str(gt_path), fps=args.fps, quality=5)

        metadata_path = case_dir / "case_metadata.json"
        record = None
        if args.skip_existing_outputs and metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as f:
                record = json.load(f)
        if not isinstance(record, dict):
            record = {"generated": {}}
        record.update(
            {
                "case": case_name,
                "source_video": video_path,
                "prompt": prompt,
                "condition_video": str(condition_path),
                "ground_truth_future": str(gt_path),
            }
        )
        record.setdefault("generated", {})
        for label in all_labels:
            register_existing_generated(record, label)
        write_case_metadata(case_dir, record)
        manifest["cases"].append(record)
        write_manifest()

    def run_split(label: str, ckpt: Path | None) -> None:
        if label not in active_labels:
            print(f"[stage] skipping split={label}; not requested by --only_labels", flush=True)
            return

        pending_records = []
        for record in manifest["cases"]:
            case_name = record["case"]
            case_dir = cases_dir / case_name
            out_path = case_dir / f"{label}_generated.mp4"
            if args.skip_existing_outputs and out_path.is_file():
                print(f"[skip] {label} {case_name} existing={out_path}", flush=True)
                register_existing_generated(record, label)
                write_case_metadata(case_dir, record)
            else:
                pending_records.append(record)
        write_manifest()

        if not pending_records:
            print(f"[stage] split={label} already complete; no generation needed", flush=True)
            return

        print(f"[stage] loading fresh Wan2.2-TI2V for split={label}", flush=True)
        pipe = load_pipe(model_dir)
        if ckpt is not None:
            print(f"[stage] loading LoRA {label}: {ckpt}", flush=True)
            pipe.load_lora(pipe.dit, str(ckpt), alpha=args.lora_alpha)
        for record in pending_records:
            case_name = record["case"]
            case_dir = cases_dir / case_name
            video_path = record["source_video"]
            prompt = record["prompt"]
            _condition, _future, input_image = sample_frames(
                video_path,
                height=args.height,
                width=args.width,
                cond_frames=args.cond_frames,
                num_frames=args.num_frames,
            )
            print(f"[generate] {label} {case_name}", flush=True)
            video = generate_one(pipe, prompt, args.negative_prompt, input_image, args)
            out_path = case_dir / f"{label}_generated.mp4"
            save_video(video, str(out_path), fps=args.fps, quality=5)
            vbench_path = copy_for_vbench(out_path, vbench_dir / label, case_name)
            prompt_maps[label][str(vbench_path.resolve())] = prompt
            record["generated"][label] = str(out_path)
            write_case_metadata(case_dir, record)
            write_manifest()
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_base:
        run_split("base", None)
    for label, ckpt in stages:
        run_split(label, ckpt)

    for split, prompt_map in prompt_maps.items():
        if prompt_map:
            write_json(output_dir / f"vbench_prompts_{split}.json", prompt_map)
    write_manifest()
    print(f"[done] outputs={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
