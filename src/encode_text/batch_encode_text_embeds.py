#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

import torch
from tqdm import tqdm

# ==== WAN 路径设置 ====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "backbone", "Wan2.1"))
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan  # type: ignore
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES  # type: ignore


TensorLike = Union[torch.Tensor, Sequence[torch.Tensor]]


def parse_args():
    p = argparse.ArgumentParser("Batch pre-encode *_prompts.json, save ONE .pt per frame.")

    p.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", type=str, required=True)

    p.add_argument("--base_path", type=str, required=True)
    p.add_argument("--pattern", type=str, default="*_prompts.json")
    p.add_argument(
        "--datasets",
        type=str,
        default="Training",
        help='Comma-separated dataset folders under base_path, e.g. "Training" or "Training,Validation". Default: Training',
    )

    p.add_argument("--out_dirname", type=str, default="text_embeds_frames",
                   help="Subfolder name inside each video folder to store per-frame pt files.")
    p.add_argument("--file_digits", type=int, default=6, help="Zero-pad digits for frame filename.")
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="If set, skip frames whose pt already exists (even if --overwrite not set).")

    p.add_argument("--n_prompt", type=str, default="", help="If empty, use pipe.sample_neg_prompt")
    p.add_argument("--save_null_once", action="store_true",
                   help="Save context_null only once per video folder as _context_null.pt (recommended).")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _as_tuple(x: TensorLike) -> Tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        return (x,)
    return tuple(x)


def _slice_item(ctx: TensorLike, i: int) -> TensorLike:
    # Return batch=1 slice for item i
    if isinstance(ctx, torch.Tensor):
        return ctx[i:i+1]
    return [t[i:i+1] for t in ctx]  # type: ignore


def _encode_once(pipe: Any, text: str, device: torch.device) -> TensorLike:
    with torch.inference_mode():
        out = pipe.text_encoder([text], device)
    outs = _as_tuple(out)
    outs_cpu = [t.detach().to(device="cpu", dtype=torch.float32) for t in outs]
    return outs_cpu[0] if len(outs_cpu) == 1 else outs_cpu


def _encode_batch(pipe: Any, texts: List[str], device: torch.device) -> TensorLike:
    with torch.inference_mode():
        out = pipe.text_encoder(texts, device)
    outs = _as_tuple(out)
    outs_cpu = [t.detach().to(device="cpu", dtype=torch.float32) for t in outs]
    return outs_cpu[0] if len(outs_cpu) == 1 else outs_cpu


def _frame_out_path(out_dir: Path, frame_id: int, digits: int) -> Path:
    return out_dir / f"{frame_id:0{digits}d}.pt"


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

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"[per-frame-encode] device={device}")

    # Init pipeline once
    cfg = WAN_CONFIGS[args.task]
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

    # Only scan selected dataset folders under base_path
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    search_roots = []
    for d in datasets:
        p = base_path / d
        if p.exists():
            search_roots.append(p)
        else:
            logging.warning(f"[warn] dataset folder not found: {p}")

    if not search_roots:
        if base_path.exists():
            search_roots = [base_path]
            logging.info(f"[per-frame-encode] falling back to scan base_path directly: {base_path}")
        else:
            logging.warning("[per-frame-encode] No valid dataset folders to scan. Nothing to do.")
            return

    prompt_jsons = []
    for root in search_roots:
        prompt_jsons.extend(root.rglob(args.pattern))
    prompt_jsons = sorted(prompt_jsons)
    
    if not prompt_jsons:
        logging.warning(f"No files matched: {args.pattern} under {base_path}")
        return

    logging.info(f"Found {len(prompt_jsons)} prompt json files.")

    total_frames = 0
    saved_frames = 0
    skipped_frames = 0
    failed_frames = 0

    pbar = tqdm(prompt_jsons, desc="[per-frame-encode] videos", unit="video")
    for jpath in pbar:
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                prompts_dict = json.load(f)

            # collect frames
            frame_ids: List[int] = []
            prompts: List[str] = []
            for k, v in prompts_dict.items():
                try:
                    frame_ids.append(int(k))
                    prompts.append(str(v))
                except Exception:
                    continue

            if not frame_ids:
                logging.warning(f"[skip] empty/invalid json: {jpath}")
                continue

            # sort
            order = sorted(range(len(frame_ids)), key=lambda i: frame_ids[i])
            frame_ids = [frame_ids[i] for i in order]
            prompts = [prompts[i] for i in order]

            video_dir = jpath.parent
            out_dir = video_dir / args.out_dirname
            out_dir.mkdir(parents=True, exist_ok=True)

            # negative prompt
            if args.n_prompt:
                n_prompt = args.n_prompt
            else:
                n_prompt = pipe.sample_neg_prompt

            # compute ctx_null once per video (or per-run, but safe to do per video)
            ctx_null = _encode_once(pipe, n_prompt, device)

            if args.save_null_once:
                null_path = out_dir / "_context_null.pt"
                if (not null_path.exists()) or args.overwrite:
                    torch.save(
                        {
                            "meta": {
                                "task": args.task,
                                "size": args.size,
                                "prompt_json": str(jpath),
                                "n_prompt": n_prompt,
                            },
                            "context_null": ctx_null,
                        },
                        str(null_path),
                    )

            pbar.set_postfix_str(f"{jpath.parent.name}/{jpath.stem} frames={len(frame_ids)} saved={saved_frames} skipped={skipped_frames}")
            logging.info(f"[video] {jpath} -> {out_dir}  (frames={len(frame_ids)})")

            # batch encode + save per frame
            bs = max(1, int(args.batch_size))
            for start in range(0, len(frame_ids), bs):
                end = min(len(frame_ids), start + bs)
                batch_frame_ids = frame_ids[start:end]
                batch_prompts = prompts[start:end]

                # optional skip if all exist
                if args.skip_existing and not args.overwrite:
                    all_exist = True
                    for fid in batch_frame_ids:
                        if not _frame_out_path(out_dir, fid, args.file_digits).exists():
                            all_exist = False
                            break
                    if all_exist:
                        skipped_frames += len(batch_frame_ids)
                        total_frames += len(batch_frame_ids)
                        continue

                ctx_batch = _encode_batch(pipe, batch_prompts, device)

                # save each frame file
                for i_local, fid in enumerate(batch_frame_ids):
                    total_frames += 1
                    out_path = _frame_out_path(out_dir, fid, args.file_digits)

                    if out_path.exists() and not args.overwrite:
                        if args.skip_existing:
                            skipped_frames += 1
                            continue
                        else:
                            skipped_frames += 1
                            continue

                    try:
                        ctx_i = _slice_item(ctx_batch, i_local)
                        payload: Dict[str, Any] = {
                            "meta": {
                                "task": args.task,
                                "size": args.size,
                                "prompt_json": str(jpath),
                                "frame_id": fid,
                                "n_prompt_saved_once": bool(args.save_null_once),
                            },
                            "frame_id": fid,
                            "context": ctx_i,
                        }
                        # 如果你想“每帧 pt 里也带 context_null”，就把下面打开：
                        if not args.save_null_once:
                            payload["context_null"] = ctx_null

                        torch.save(payload, str(out_path))
                        saved_frames += 1
                    except Exception as e:
                        failed_frames += 1
                        logging.warning(f"[fail-frame] {jpath.name} frame={fid}: {e}")

        except Exception as e:
            logging.exception(f"[fail-video] {jpath}: {e}")

    logging.info("============================================================")
    logging.info(f"[done] total_frames={total_frames} saved={saved_frames} skipped={skipped_frames} failed={failed_frames}")
    logging.info("============================================================")


if __name__ == "__main__":
    main()