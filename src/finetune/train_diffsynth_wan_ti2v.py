#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Three-stage DiffSynth-Studio finetuning for Wan2.2-TI2V-5B.

This trainer keeps the repository's stage sampling rules while switching the
model backend to DiffSynth-Studio.  For each old continuation window, the last
conditioning frame becomes the TI2V input image and the model is trained to
generate that frame plus the following ``gen_len`` frames.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:
    def record(fn):
        return fn


warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_PROMPT = (
    "Endoscopic surgical scene, realistic continuous motion, plausible tool "
    "and tissue dynamics, steady camera."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_diffsynth_on_path(diffsynth_root: str) -> None:
    root = Path(diffsynth_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"DiffSynth-Studio root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _parse_size(size: str) -> Tuple[int, int]:
    s = size.lower().replace("x", "*")
    if "*" not in s:
        raise ValueError(f"Bad --size={size}; expected WIDTH*HEIGHT, e.g. 832*480")
    w, h = s.split("*", 1)
    return int(w), int(h)


def _extract_last_int(stem: str) -> Optional[int]:
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _atomic_write_text(path: str, text: str) -> None:
    _ensure_dir_for_file(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _write_loss_svg(path: str, steps: List[int], losses: List[float]) -> None:
    if not steps:
        return
    w, h = 980, 420
    ml, mr, mt, mb = 60, 20, 20, 50
    pw = w - ml - mr
    ph = h - mt - mb
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses), max(losses)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1e-6

    def x_to_px(x: int) -> float:
        return ml + (x - x_min) / (x_max - x_min) * pw

    def y_to_px(y: float) -> float:
        return mt + (y_max - y) / (y_max - y_min) * ph

    pts = " ".join(f"{x_to_px(s):.2f},{y_to_px(l):.2f}" for s, l in zip(steps, losses))
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>
  <rect x='0' y='0' width='{w}' height='{h}' fill='white'/>
  <line x1='{ml}' y1='{mt + ph}' x2='{ml + pw}' y2='{mt + ph}' stroke='#222' stroke-width='1.5'/>
  <line x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt + ph}' stroke='#222' stroke-width='1.5'/>
  <polyline fill='none' stroke='#0a84ff' stroke-width='2' points='{pts}'/>
  <text x='{ml}' y='{h - 14}' font-size='13' fill='#333'>step: {x_min} -> {x_max}</text>
  <text x='{ml + pw - 180}' y='{h - 14}' font-size='13' fill='#333'>loss: {y_min:.6f} -> {y_max:.6f}</text>
  <text x='{ml}' y='{mt - 4}' font-size='14' fill='#111'>DiffSynth Wan2.2-TI2V Training Loss</text>
</svg>
"""
    _atomic_write_text(path, svg)


def _write_progress_json(path: str, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_progress_svg(path: str, payload: Dict[str, object]) -> None:
    total = max(int(payload.get("total_steps") or 0), 1)
    step = max(int(payload.get("step") or 0), 0)
    pct = max(0.0, min(1.0, step / total))
    loss = payload.get("loss")
    ema = payload.get("loss_ema")
    lr = payload.get("lr")
    phase = str(payload.get("phase") or "starting")
    status = str(payload.get("status") or "")
    elapsed = float(payload.get("elapsed_sec") or 0.0)
    eta = payload.get("eta_sec")
    eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
    loss_text = "n/a" if loss is None else f"{float(loss):.6f}"
    ema_text = "n/a" if ema is None else f"{float(ema):.6f}"
    lr_text = "n/a" if lr is None else f"{float(lr):.2e}"
    w, h = 760, 230
    x, y, bw, bh = 48, 92, 664, 28
    fill_w = bw * pct
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>
  <rect x='0' y='0' width='{w}' height='{h}' fill='white'/>
  <text x='48' y='42' font-size='22' font-weight='700' fill='#111'>DiffSynth Wan2.2-TI2V Training Progress</text>
  <text x='48' y='70' font-size='14' fill='#444'>phase: {phase}  status: {status}</text>
  <rect x='{x}' y='{y}' width='{bw}' height='{bh}' rx='4' fill='#e6e8eb'/>
  <rect x='{x}' y='{y}' width='{fill_w:.2f}' height='{bh}' rx='4' fill='#0a84ff'/>
  <text x='48' y='142' font-size='14' fill='#222'>step: {step} / {total} ({pct * 100:.2f}%)</text>
  <text x='48' y='168' font-size='14' fill='#222'>loss: {loss_text}   ema: {ema_text}   lr: {lr_text}</text>
  <text x='48' y='194' font-size='14' fill='#222'>elapsed: {elapsed:.1f}s   eta: {eta_text}</text>
</svg>
"""
    _atomic_write_text(path, svg)


def _trainable_param_signature(model: torch.nn.Module, max_names: int = 12) -> Dict[str, object]:
    items = [(name, tuple(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
    digest = hashlib.sha256(
        "\n".join(f"{name}:{','.join(map(str, shape))}" for name, shape in items).encode("utf-8")
    ).hexdigest()
    return {
        "count": len(items),
        "digest": digest,
        "first": items[:max_names],
        "last": items[-max_names:],
    }


class _StepProgress:
    def __init__(self, total: int, initial: int = 0, disable: bool = False):
        self.total = max(int(total), 1)
        self.current = int(initial)
        self.disable = disable
        self._pbar = None
        if not disable:
            try:
                from tqdm.auto import tqdm

                self._pbar = tqdm(
                    total=self.total,
                    initial=self.current,
                    desc="training",
                    unit="step",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            except Exception:
                self._pbar = None
                self._print()

    def update(self, current: int, loss: Optional[float] = None, lr: Optional[float] = None) -> None:
        if self.disable:
            return
        current = int(current)
        delta = current - self.current
        self.current = current
        if self._pbar is not None:
            if delta > 0:
                self._pbar.update(delta)
            postfix = {}
            if loss is not None:
                postfix["loss"] = f"{loss:.6f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.2e}"
            if postfix:
                self._pbar.set_postfix(postfix)
        else:
            self._print(loss=loss, lr=lr)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        elif not self.disable:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _print(self, loss: Optional[float] = None, lr: Optional[float] = None) -> None:
        pct = max(0.0, min(1.0, self.current / self.total))
        width = 32
        filled = int(width * pct)
        bar = "#" * filled + "-" * (width - filled)
        extras = []
        if loss is not None:
            extras.append(f"loss={loss:.6f}")
        if lr is not None:
            extras.append(f"lr={lr:.2e}")
        suffix = " ".join(extras)
        sys.stderr.write(f"\rtraining [{bar}] {self.current}/{self.total} {pct * 100:6.2f}% {suffix}")
        sys.stderr.flush()


def _load_prompt_map(path: Path) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, str] = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        raise ValueError(f"Unsupported prompt json format: {path}")
    for k, v in items:
        try:
            fid = int(k)
        except Exception:
            fid = _extract_last_int(str(k))
        if fid is None:
            continue
        if isinstance(v, dict):
            text = v.get("prompt") or v.get("text") or v.get("caption") or ""
        else:
            text = str(v)
        if text:
            out[int(fid)] = text
    return out


def _find_prompt_json(video_dir: Path, pattern: str) -> Optional[Path]:
    if not pattern:
        return None
    matches = sorted(video_dir.glob(pattern))
    if not matches:
        matches = sorted(video_dir.parent.glob(pattern))
    return matches[0] if matches else None


def _pick_prompt(
    prompt_map: Optional[Dict[int, str]],
    frame_id: int,
    mode: str,
    strict: bool,
    fallback: str,
) -> str:
    if not prompt_map:
        if fallback:
            return fallback
        if strict:
            raise KeyError("No prompt map is available and no fallback prompt was provided.")
        return DEFAULT_PROMPT

    if frame_id in prompt_map:
        return prompt_map[frame_id]

    fids = sorted(prompt_map)
    if mode == "exact":
        if strict:
            raise KeyError(f"Exact prompt not found for frame_id={frame_id}")
        return fallback or prompt_map[fids[0]]

    if mode == "nearest_prev":
        prev = [x for x in fids if x <= frame_id]
        if prev:
            return prompt_map[prev[-1]]
        if strict:
            raise KeyError(f"No prompt <= frame_id={frame_id}")
        return prompt_map[fids[0]]

    nearest = min(fids, key=lambda x: abs(x - frame_id))
    return prompt_map[nearest]


def _resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    if img.size == (width, height):
        return img.convert("RGB")
    return img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)


def _load_image(path: str, width: int, height: int) -> Image.Image:
    with Image.open(path) as img:
        return _resize_image(img, width, height)


def _read_video_frames(
    video_path: str,
    frame_indices: Sequence[int],
    width: int,
    height: int,
    pad_mode: str,
) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[Image.Image] = []
    last_rgb = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for fid in frame_indices:
        rgb = None
        if 0 <= int(fid) < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb is None:
            if pad_mode == "error":
                cap.release()
                raise RuntimeError(f"Video ended early: {video_path}")
            if last_rgb is None:
                import numpy as np

                rgb = np.zeros((height, width, 3), dtype=np.uint8)
            elif pad_mode == "repeat_last":
                rgb = last_rgb.copy()
            else:
                rgb = 0 * last_rgb
        last_rgb = rgb
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def _video_frame_count(video_path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total


@dataclass
class WindowSource:
    kind: str
    name: str
    path: str
    total_frames: int
    frame_paths: Optional[List[str]] = None
    frame_ids: Optional[List[int]] = None
    prompt_map: Optional[Dict[int, str]] = None
    prompt_fallback: str = ""
    anchors: Optional[List[int]] = None


class WanTi2VWindowDataset(Dataset):
    def __init__(
        self,
        stage: str,
        width: int,
        height: int,
        cond_len: int,
        gen_len: int,
        frame_step: int,
        pad_mode: str,
        max_items: int,
        prompt: str = "",
        train_video_dir: str = "",
        train_frames_root: str = "",
        dataset_root: str = "",
        splits: Sequence[str] = (),
        frames_dirname: str = "Frames",
        frame_embed_dirname: str = "text_embeds_frames",
        sample_stride: int = 1,
        prompt_pattern: str = "*_prompts.json",
        prompt_match: str = "nearest_prev",
        prompt_strict: bool = False,
        strict_anchor_match: bool = True,
        replay_video_dir: str = "",
        replay_frames_root: str = "",
        replay_ratio: float = 0.0,
        replay_prompt: str = "",
        prompt_prefix: str = "",
    ):
        self.stage = stage
        self.width = int(width)
        self.height = int(height)
        self.cond_len = int(cond_len)
        self.gen_len = int(gen_len)
        self.frame_step = max(1, int(frame_step))
        self.pad_mode = pad_mode
        self.max_items = max(1, int(max_items))
        self.prompt = prompt
        self.prompt_pattern = prompt_pattern
        self.prompt_match = prompt_match
        self.prompt_strict = bool(prompt_strict)
        self.sample_stride = max(1, int(sample_stride))
        self.strict_anchor_match = bool(strict_anchor_match)
        self.replay_ratio = float(max(0.0, min(1.0, replay_ratio)))
        self.prompt_prefix = str(prompt_prefix).strip()

        self.sources: List[WindowSource] = []
        if stage == "stage3":
            self._discover_stage3(dataset_root, splits, frames_dirname, frame_embed_dirname)
        else:
            self._discover_stage12(train_video_dir, train_frames_root)

        if not self.sources:
            raise RuntimeError(f"No trainable samples discovered for {stage}")

        self.weights = [
            float(len(src.anchors)) if src.anchors is not None else float(self._random_slots(src))
            for src in self.sources
        ]

        self.replay_sources: List[WindowSource] = []
        self.replay_weights: List[float] = []
        if self.replay_ratio > 0.0 and (replay_video_dir.strip() or replay_frames_root.strip()):
            self._discover_replay(replay_video_dir, replay_frames_root, replay_prompt)

        logging.info(
            "[data] stage=%s sources=%d weighted_slots=%d output_frames=%d replay_sources=%d replay_ratio=%.2f",
            self.stage,
            len(self.sources),
            int(sum(self.weights)),
            self.gen_len + 1,
            len(self.replay_sources),
            self.replay_ratio,
        )

    def _required_total_span(self) -> int:
        return (self.cond_len + self.gen_len - 1) * self.frame_step + 1

    def _random_slots(self, src: WindowSource) -> int:
        return max(1, src.total_frames - self._required_total_span() + 1)

    def _discover_stage12(self, train_video_dir: str, train_frames_root: str) -> None:
        if bool(train_video_dir.strip()) == bool(train_frames_root.strip()):
            raise ValueError("Set exactly one data source: --train_video_dir OR --train_frames_root")
        required_span = self._required_total_span()
        if train_video_dir.strip():
            root = Path(train_video_dir)
            for p in sorted(root.iterdir()):
                if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                    continue
                total = _video_frame_count(str(p))
                if total < required_span:
                    continue
                self.sources.append(
                    WindowSource(
                        kind="video",
                        name=p.name,
                        path=str(p),
                        total_frames=total,
                        prompt_fallback=self.prompt,
                    )
                )
            return

        for frame_dir, frame_paths, frame_ids in self._iter_frame_dirs(Path(train_frames_root)):
            if len(frame_paths) < required_span:
                continue
            prompt_map = None
            prompt_path = _find_prompt_json(frame_dir, self.prompt_pattern)
            if prompt_path is not None:
                prompt_map = _load_prompt_map(prompt_path)
            self.sources.append(
                WindowSource(
                    kind="frames",
                    name=frame_dir.name,
                    path=str(frame_dir),
                    total_frames=len(frame_paths),
                    frame_paths=frame_paths,
                    frame_ids=frame_ids,
                    prompt_map=prompt_map,
                    prompt_fallback=self.prompt,
                )
            )

    def _discover_replay(
        self,
        replay_video_dir: str,
        replay_frames_root: str,
        replay_prompt: str,
    ) -> None:
        """Populate replay_sources from a prior-stage dataset for catastrophic-forgetting prevention."""
        required_span = self._required_total_span()
        fallback = replay_prompt.strip() or DEFAULT_PROMPT

        if replay_video_dir.strip():
            root = Path(replay_video_dir)
            for p in sorted(root.iterdir()):
                if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                    continue
                total = _video_frame_count(str(p))
                if total < required_span:
                    continue
                self.replay_sources.append(
                    WindowSource(
                        kind="video",
                        name=p.name,
                        path=str(p),
                        total_frames=total,
                        prompt_fallback=fallback,
                    )
                )
        elif replay_frames_root.strip():
            for frame_dir, frame_paths, frame_ids in self._iter_frame_dirs(Path(replay_frames_root)):
                if len(frame_paths) < required_span:
                    continue
                self.replay_sources.append(
                    WindowSource(
                        kind="frames",
                        name=frame_dir.name,
                        path=str(frame_dir),
                        total_frames=len(frame_paths),
                        frame_paths=frame_paths,
                        frame_ids=frame_ids,
                        prompt_map=None,
                        prompt_fallback=fallback,
                    )
                )

        self.replay_weights = [float(self._random_slots(src)) for src in self.replay_sources]
        logging.info("[data] replay discovered %d sources", len(self.replay_sources))

    def _discover_stage3(
        self,
        dataset_root: str,
        splits: Sequence[str],
        frames_dirname: str,
        frame_embed_dirname: str,
    ) -> None:
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"dataset_root not found: {root}")
        split_names = [s.strip() for s in splits if s.strip()]
        if not split_names:
            raise ValueError("--splits is empty")

        for split in split_names:
            split_dir = root / split
            if not split_dir.exists():
                logging.warning("[scan] split not found, skip: %s", split_dir)
                continue
            for video_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                frames_dir = video_dir / frames_dirname
                if not frames_dir.exists():
                    logging.warning("[scan] skip %s: missing %s", video_dir, frames_dirname)
                    continue
                frame_paths, frame_ids = self._frame_entries(frames_dir)
                if len(frame_paths) < self._required_total_span():
                    continue
                prompt_path = _find_prompt_json(video_dir, self.prompt_pattern)
                prompt_map = _load_prompt_map(prompt_path) if prompt_path is not None else None
                anchors = self._stage3_anchors(video_dir, frame_ids, frame_embed_dirname)
                anchors = anchors[:: self.sample_stride]
                anchors = [pos for pos in anchors if self._anchor_is_valid(pos, len(frame_paths))]
                if not anchors:
                    continue
                self.sources.append(
                    WindowSource(
                        kind="frames",
                        name=f"{split}/{video_dir.name}",
                        path=str(frames_dir),
                        total_frames=len(frame_paths),
                        frame_paths=frame_paths,
                        frame_ids=frame_ids,
                        prompt_map=prompt_map,
                        prompt_fallback=self.prompt,
                        anchors=anchors,
                    )
                )

    def _stage3_anchors(self, video_dir: Path, frame_ids: List[int], frame_embed_dirname: str) -> List[int]:
        id_to_pos = {fid: i for i, fid in enumerate(frame_ids)}
        embed_dir = video_dir / frame_embed_dirname
        anchor_fids: List[int] = []
        if embed_dir.exists():
            for pt in sorted(embed_dir.glob("*.pt")):
                if pt.name == "_context_null.pt":
                    continue
                fid = _extract_last_int(pt.stem)
                if fid is not None:
                    anchor_fids.append(fid)
        if not anchor_fids:
            prompt_path = _find_prompt_json(video_dir, self.prompt_pattern)
            if prompt_path is not None:
                anchor_fids = sorted(_load_prompt_map(prompt_path))

        anchors: List[int] = []
        for fid in anchor_fids:
            pos = id_to_pos.get(fid)
            if pos is None and not self.strict_anchor_match:
                pos = min(range(len(frame_ids)), key=lambda i: abs(frame_ids[i] - fid))
            if pos is not None:
                anchors.append(pos)
        return sorted(set(anchors))

    def _anchor_is_valid(self, anchor_pos: int, total_frames: int) -> bool:
        start_pos = anchor_pos - (self.cond_len - 1) * self.frame_step
        end_pos = anchor_pos + self.gen_len * self.frame_step
        return start_pos >= 0 and end_pos < total_frames

    def _iter_frame_dirs(self, root: Path):
        if not root.exists():
            raise FileNotFoundError(f"train_frames_root not found: {root}")
        dirs = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
        for d in dirs:
            frame_paths, frame_ids = self._frame_entries(d)
            if frame_paths:
                yield d, frame_paths, frame_ids

    def _frame_entries(self, frame_dir: Path) -> Tuple[List[str], List[int]]:
        imgs = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            return [], []
        entries = []
        for i, p in enumerate(imgs):
            fid = _extract_last_int(p.stem)
            entries.append((i if fid is None else fid, str(p)))
        entries.sort(key=lambda x: (x[0], x[1]))
        return [x[1] for x in entries], [int(x[0]) for x in entries]

    def __len__(self) -> int:
        return self.max_items

    def _sample_source(self) -> WindowSource:
        return random.choices(self.sources, weights=self.weights, k=1)[0]

    def _sample_anchor(self, src: WindowSource) -> int:
        if src.anchors is not None:
            return random.choice(src.anchors)
        max_start = max(0, src.total_frames - self._required_total_span())
        start_pos = random.randint(0, max_start)
        return start_pos + (self.cond_len - 1) * self.frame_step

    def _frame_id_at(self, src: WindowSource, pos: int) -> int:
        if src.frame_ids is None:
            return int(pos)
        return int(src.frame_ids[pos])

    def __getitem__(self, index: int) -> Dict[str, object]:
        use_replay = (
            len(self.replay_sources) > 0
            and self.replay_ratio > 0.0
            and random.random() < self.replay_ratio
        )

        if use_replay:
            src = random.choices(self.replay_sources, weights=self.replay_weights, k=1)[0]
        else:
            src = self._sample_source()

        anchor_pos = self._sample_anchor(src)
        positions = [anchor_pos + i * self.frame_step for i in range(self.gen_len + 1)]

        if src.kind == "video":
            frames = _read_video_frames(src.path, positions, self.width, self.height, self.pad_mode)
            anchor_fid = int(anchor_pos)
        else:
            assert src.frame_paths is not None
            frames = [_load_image(src.frame_paths[pos], self.width, self.height) for pos in positions]
            anchor_fid = self._frame_id_at(src, anchor_pos)

        prompt = _pick_prompt(
            src.prompt_map,
            anchor_fid,
            mode=self.prompt_match,
            strict=self.prompt_strict and not use_replay,
            fallback=src.prompt_fallback,
        )

        # Prepend scene-description prefix to primary (non-replay) prompts only.
        # Replay prompts are already in Stage-1 format and need no prefix.
        if self.prompt_prefix and not use_replay:
            prompt = self.prompt_prefix + " " + prompt

        return {
            "video": frames,
            "prompt": prompt,
            "meta": {
                "source": src.name,
                "anchor_pos": int(anchor_pos),
                "anchor_fid": int(anchor_fid),
                "stage": self.stage,
                "replay": bool(use_replay),
            },
        }


class DiffSynthWanTi2VTrainingModule(torch.nn.Module):
    def __init__(
        self,
        model_dir: str,
        torch_dtype: torch.dtype,
        device: torch.device | str,
        trainable_models: Optional[str],
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
        lora_last_n_blocks: int,
        lora_checkpoint: Optional[str],
        frozen_lora_ckpt: Optional[str],
        frozen_lora_rank: int,
        frozen_lora_alpha: int,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
        cfg_dropout_p: float,
        max_timestep_boundary: float,
        min_timestep_boundary: float,
        tiled: bool,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
    ):
        super().__init__()
        from diffsynth.core import load_state_dict
        from diffsynth.diffusion import DiffusionTrainingModule, FlowMatchSFTLoss
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        self._training_helper = DiffusionTrainingModule()
        self.loss_fn = FlowMatchSFTLoss
        self.cfg_dropout_p = float(cfg_dropout_p)
        self.max_timestep_boundary = float(max_timestep_boundary)
        self.min_timestep_boundary = float(min_timestep_boundary)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

        model_configs, tokenizer_config = self._build_model_configs(model_dir, ModelConfig)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )
        self.pipe = self._training_helper.split_pipeline_units(
            "sft",
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model="dit" if use_lora else None,
        )
        self._switch_to_training_mode(
            load_state_dict=load_state_dict,
            trainable_models=trainable_models,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_last_n_blocks=lora_last_n_blocks,
            lora_checkpoint=lora_checkpoint,
            frozen_lora_ckpt=frozen_lora_ckpt,
            frozen_lora_rank=frozen_lora_rank,
            frozen_lora_alpha=frozen_lora_alpha,
        )

    @staticmethod
    def _build_model_configs(model_dir: str, model_config_cls):
        root = Path(model_dir)
        if not root.exists():
            raise FileNotFoundError(f"Wan2.2 model directory not found: {root}")
        dit_paths = sorted(glob.glob(str(root / "diffusion_pytorch_model*.safetensors")))
        dit_paths = [p for p in dit_paths if not p.endswith(".index.json")]
        if not dit_paths:
            raise FileNotFoundError(f"No diffusion_pytorch_model*.safetensors found under {root}")
        t5_path = root / "models_t5_umt5-xxl-enc-bf16.pth"
        vae_path = root / "Wan2.2_VAE.pth"
        tokenizer_path = root / "google" / "umt5-xxl"
        for p in (t5_path, vae_path, tokenizer_path):
            if not p.exists():
                raise FileNotFoundError(f"Required Wan2.2 file not found: {p}")
        return (
            [
                model_config_cls(path=dit_paths),
                model_config_cls(path=str(t5_path)),
                model_config_cls(path=str(vae_path)),
            ],
            model_config_cls(path=str(tokenizer_path)),
        )

    @staticmethod
    def _expand_lora_target_modules(
        model: torch.nn.Module,
        lora_target_modules: str,
        lora_last_n_blocks: int = 0,
    ) -> List[str]:
        specs = [x.strip() for x in lora_target_modules.split(",") if x.strip()]
        if not specs:
            return []
        targets: List[str] = []
        for name, module in model.named_modules():
            if not name or not isinstance(module, torch.nn.Linear):
                continue
            if any(name == spec or name.endswith(f".{spec}") for spec in specs):
                targets.append(name)
        targets = sorted(set(targets))

        if lora_last_n_blocks > 0 and targets:
            def _block_idx(name: str) -> Optional[int]:
                m = re.search(r"blocks?\.(\d+)\.", name)
                return int(m.group(1)) if m else None

            block_idxs = {_block_idx(t) for t in targets}
            block_idxs.discard(None)
            if block_idxs:
                max_block = max(block_idxs)  # type: ignore[type-var]
                cutoff = max_block - lora_last_n_blocks + 1
                old_count = len(targets)
                targets = [t for t in targets if (_block_idx(t) or 0) >= cutoff]
                logging.info(
                    "[lora] lora_last_n_blocks=%d: blocks %d-%d, kept %d/%d modules",
                    lora_last_n_blocks, cutoff, max_block, len(targets), old_count,
                )

        if not targets:
            raise RuntimeError(f"No LoRA target modules matched specs={specs}")
        return targets

    @staticmethod
    def _apply_lora_delta_to_base_weights(
        model: torch.nn.Module,
        lora_state_dict: Dict[str, torch.Tensor],
        lora_alpha: float,
        lora_rank: int,
    ) -> int:
        """Permanently fuse a LoRA checkpoint into base model weights.

        Computes delta = (lora_B @ lora_A) * (alpha/rank) for every matched
        module and adds it in-place to the module's .weight.  This bakes the
        prior-stage LoRA into the base so a fresh new LoRA can be stacked on
        top without any weight-overwrite catastrophic-forgetting.
        """
        scaling = float(lora_alpha) / max(int(lora_rank), 1)
        lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, val in lora_state_dict.items():
            if ".lora_A.default.weight" in key:
                mod_path = key[: key.index(".lora_A.default.weight")]
                lora_pairs.setdefault(mod_path, {})["A"] = val
            elif ".lora_B.default.weight" in key:
                mod_path = key[: key.index(".lora_B.default.weight")]
                lora_pairs.setdefault(mod_path, {})["B"] = val

        named_modules = dict(model.named_modules())
        applied = 0
        for mod_path, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                logging.warning("[frozen_lora] incomplete pair for %s, skipping", mod_path)
                continue
            module = named_modules.get(mod_path)
            if module is None or not hasattr(module, "weight"):
                logging.warning("[frozen_lora] module not found: %s", mod_path)
                continue
            lora_A = pair["A"].to(dtype=module.weight.dtype, device=module.weight.device)
            lora_B = pair["B"].to(dtype=module.weight.dtype, device=module.weight.device)
            with torch.no_grad():
                module.weight.data.add_((lora_B @ lora_A) * scaling)
            applied += 1
        logging.info(
            "[frozen_lora] fused %d LoRA deltas into base weights (scaling=%.4f)",
            applied, scaling,
        )
        return applied

    def _switch_to_training_mode(
        self,
        load_state_dict,
        trainable_models: Optional[str],
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
        lora_last_n_blocks: int,
        lora_checkpoint: Optional[str],
        frozen_lora_ckpt: Optional[str],
        frozen_lora_rank: int,
        frozen_lora_alpha: int,
    ) -> None:
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([] if use_lora or trainable_models is None else trainable_models.split(","))
        if not use_lora:
            return

        if self.pipe.dit is None:
            raise RuntimeError("WanVideoPipeline has no DiT model; cannot apply LoRA.")

        # Step 1: Fuse prior-stage frozen LoRA into base weights (additive LoRA stacking).
        # This bakes previous-stage knowledge permanently so the new trainable LoRA
        # starts from an initialisation that already encodes stage-N-1 capabilities,
        # instead of overwriting them via the same parameter set.
        if frozen_lora_ckpt:
            frozen_state = load_state_dict(frozen_lora_ckpt)
            lora_loader = self.pipe.lora_loader(
                torch_dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
            frozen_state = lora_loader.convert_state_dict(frozen_state)
            frozen_state = self._training_helper.mapping_lora_state_dict(frozen_state)
            applied = self._apply_lora_delta_to_base_weights(
                self.pipe.dit,
                frozen_state,
                lora_alpha=float(frozen_lora_alpha),
                lora_rank=int(frozen_lora_rank),
            )
            if applied == 0:
                raise RuntimeError(
                    f"frozen_lora_ckpt={frozen_lora_ckpt} applied 0 deltas — "
                    "check that the checkpoint path and key format are correct."
                )

        # Step 2: Add new (optionally block-restricted) trainable LoRA on top.
        target_modules = self._expand_lora_target_modules(
            self.pipe.dit, lora_target_modules, lora_last_n_blocks=lora_last_n_blocks
        )
        logging.info(
            "[lora] matched %d target modules; first=%s last=%s",
            len(target_modules),
            target_modules[:3],
            target_modules[-3:],
        )
        self.pipe.dit = self._training_helper.add_lora_to_model(
            self.pipe.dit,
            target_modules=target_modules,
            lora_rank=int(lora_rank),
            lora_alpha=int(lora_alpha),
            upcast_dtype=self.pipe.torch_dtype,
        )

        # Step 3: Optionally warm-start the new trainable LoRA from a checkpoint
        # (used when resuming a stage mid-training, not for cross-stage transfer).
        if lora_checkpoint:
            lora = load_state_dict(lora_checkpoint)
            lora_loader = self.pipe.lora_loader(torch_dtype=self.pipe.torch_dtype, device=self.pipe.device)
            lora = lora_loader.convert_state_dict(lora)
            lora = self._training_helper.mapping_lora_state_dict(lora)
            result = self.pipe.dit.load_state_dict(lora, strict=False)
            logging.info(
                "[resume] loaded LoRA checkpoint=%s keys=%d missing=%d unexpected=%d",
                lora_checkpoint,
                len(lora),
                len(result.missing_keys),
                len(result.unexpected_keys),
            )

    def trainable_modules(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def trainable_param_names(self):
        return self._training_helper.trainable_param_names.__get__(self, type(self))()

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return self._training_helper.export_trainable_state_dict.__get__(self, type(self))(
            state_dict,
            remove_prefix=remove_prefix,
        )

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        return self._training_helper.transfer_data_to_device(data, device, torch_float_dtype)

    def get_pipeline_inputs(self, data: Dict[str, object]):
        video = data["video"]
        assert isinstance(video, list) and video, "data['video'] must be a non-empty PIL image list"
        prompt = str(data.get("prompt", ""))
        if self.cfg_dropout_p > 0.0 and random.random() < self.cfg_dropout_p:
            prompt = ""
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "input_image": video[0],
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": self.tiled,
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "denoising_strength": 1.0,
            "seed": None,
            "vace_reference_image": None,
            "end_image": None,
            "control_video": None,
            "reference_image": None,
            "camera_control_direction": None,
            "camera_control_speed": 1 / 54,
            "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
            "vace_video": None,
            "vace_video_mask": None,
            "motion_bucket_id": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "input_audio": None,
            "audio_sample_rate": 16000,
            "s2v_pose_video": None,
            "audio_embeds": None,
            "s2v_pose_latents": None,
            "motion_video": None,
            "animate_pose_video": None,
            "animate_face_video": None,
            "animate_inpaint_video": None,
            "animate_mask_video": None,
            "vap_video": None,
            "vap_prompt": " ",
            "negative_vap_prompt": " ",
            "wantodance_music_path": None,
            "wantodance_reference_image": None,
            "wantodance_fps": 30,
            "wantodance_keyframes": None,
            "wantodance_keyframes_mask": None,
            "framewise_decoding": False,
        }
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _inputs_nega = inputs
        return self.loss_fn(self.pipe, **inputs_shared, **inputs_posi)


def _extract_step_from_path(path: str) -> int:
    if not path:
        return 0
    m = re.search(r"(?:step[-_]?)(\d+)", Path(path).stem)
    return int(m.group(1)) if m else 0


def parse_args(default_stage: Optional[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("DiffSynth-Studio Wan2.2-TI2V-5B three-stage finetuning")
    p.add_argument("--stage", type=str, default=default_stage or "stage1", choices=["stage1", "stage2", "stage3"])
    p.add_argument("--diffsynth_root", type=str, default=str(_repo_root() / "DiffSynth-Studio"))
    p.add_argument("--wan_root", type=str, default="", help="Deprecated compatibility arg; ignored.")
    p.add_argument("--task", type=str, default="wan2.2-ti2v-5b")
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Local Wan2.2-TI2V-5B weight directory.")
    p.add_argument("--model_dir", type=str, default="", help="Alias for --ckpt_dir.")

    p.add_argument("--train_video_dir", type=str, default="")
    p.add_argument("--train_frames_root", type=str, default="")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--splits", type=str, default="Training")
    p.add_argument("--frames_dirname", type=str, default="Frames")
    p.add_argument("--frame_embed_dirname", type=str, default="text_embeds_frames")
    p.add_argument("--sample_stride", type=int, default=1)
    p.add_argument("--strict_anchor_match", action="store_true")
    p.add_argument("--non_strict_anchor_match", action="store_true")

    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--prompt_pattern", type=str, default="*_prompts.json")
    p.add_argument("--prompt_match", type=str, default="nearest_prev", choices=["exact", "nearest_prev", "nearest"])
    p.add_argument("--prompt_strict", action="store_true")

    p.add_argument("--cond_len", type=int, default=41)
    p.add_argument("--gen_len", type=int, default=40)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "zeros", "error"])
    p.add_argument("--future_fill", type=str, default="black", help="Deprecated compatibility arg; ignored.")

    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_steps_add", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--dataset_num_workers", type=int, default=0)
    p.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", "--save_steps", dest="save_steps", type=int, default=500)
    p.add_argument("--keep_latest_ckpts", type=int, default=1,
                   help="Keep only the N most recent checkpoints; older ones are deleted. 0 = keep all.")
    p.add_argument("--out_dir", "--output_path", dest="out_dir", type=str, default="finetune_ckpts")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--resume_optim", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "no"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--cfg_dropout_p", type=float, default=0.1)
    p.add_argument("--max_timestep_boundary", type=float, default=1.0)
    p.add_argument("--min_timestep_boundary", type=float, default=0.0)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--tile_size", type=str, default="30,52")
    p.add_argument("--tile_stride", type=str, default="15,26")

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0, help="Accepted for compatibility; PEFT default is used.")
    p.add_argument("--lora_last_n_blocks", type=int, default=0,
                   help="Restrict trainable LoRA to the last N transformer blocks (0 = all blocks).")
    p.add_argument("--lora_keywords", type=str, default="", help="Alias for --lora_target_modules.")
    p.add_argument("--lora_target_modules", type=str, default="")
    # Additive LoRA stacking: fuse a frozen prior-stage LoRA into base weights before adding new trainable LoRA.
    p.add_argument("--frozen_lora_ckpt", type=str, default="",
                   help="Path to a prior-stage LoRA safetensors checkpoint to fuse (frozen) into base weights.")
    p.add_argument("--frozen_lora_rank", type=int, default=32,
                   help="LoRA rank of the frozen prior-stage checkpoint.")
    p.add_argument("--frozen_lora_alpha", type=int, default=32,
                   help="LoRA alpha of the frozen prior-stage checkpoint.")
    # Data replay for catastrophic-forgetting prevention.
    p.add_argument("--replay_video_dir", type=str, default="",
                   help="Video directory of a prior-stage dataset to replay (mixed with primary data).")
    p.add_argument("--replay_frames_root", type=str, default="",
                   help="Frame-folder root of a prior-stage dataset to replay.")
    p.add_argument("--replay_ratio", type=float, default=0.0,
                   help="Fraction [0, 1) of training samples drawn from the replay dataset.")
    p.add_argument("--replay_prompt", type=str, default="",
                   help="Fixed text prompt used for all replay samples (defaults to Stage-1 scene description).")
    # Prompt prefix: prepend a common scene-description prefix to primary-data prompts.
    p.add_argument("--prompt_prefix", type=str, default="",
                   help="Text prefix prepended to every primary-data prompt (not applied to replay samples).")
    p.add_argument("--trainable_models", type=str, default="")
    p.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")
    p.add_argument("--use_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_gradient_checkpointing_offload", action="store_true")

    p.add_argument("--metrics_csv", type=str, default="")
    p.add_argument("--metrics_svg", type=str, default="")
    p.add_argument("--progress_json", type=str, default="")
    p.add_argument("--progress_svg", type=str, default="")
    p.add_argument("--progress_interval", type=int, default=1)
    p.add_argument("--disable_progress_bar", action="store_true")
    p.add_argument("--plot_interval", type=int, default=1)
    p.add_argument("--loss_type", type=str, default="mse", help="Deprecated compatibility arg; DiffSynth SFT uses MSE.")
    p.add_argument("--loss_huber_delta", type=float, default=0.05, help="Deprecated compatibility arg; ignored.")
    p.add_argument("--loss_ema_decay", type=float, default=0.98)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Deprecated rollout/sigma args accepted so old callers fail less abruptly.
    p.add_argument("--sample_solver", type=str, default="")
    p.add_argument("--sample_steps", type=int, default=0)
    p.add_argument("--sample_shift", type=float, default=0.0)
    p.add_argument("--sample_guide_scale", type=float, default=0.0)
    p.add_argument("--context_scale", type=float, default=1.0)
    p.add_argument("--train_objective", type=str, default="")
    p.add_argument("--train_last_n_steps", type=int, default=1)
    p.add_argument("--denoise_log_interval", type=int, default=0)
    p.add_argument("--sigma_min", type=float, default=0.0)
    p.add_argument("--sigma_max", type=float, default=1.0)
    p.add_argument("--sigma_sampling", type=str, default="")
    p.add_argument("--sigma_min_start", type=float, default=0.0)
    p.add_argument("--sigma_curriculum_steps", type=int, default=0)
    p.add_argument("--fixed_sigma", type=float, default=0.0)
    p.add_argument("--text_embed_path", type=str, default="")
    p.add_argument("--frame_embed_root", type=str, default="")
    p.add_argument("--lookup_embed_path", type=str, default="")
    p.add_argument("--slice_prompt_embed_root", type=str, default="")
    p.add_argument("--slice_prompt_embed_dirname", type=str, default="")
    p.add_argument("--slice_prompt_match", type=str, default="")
    p.add_argument("--slice_prompt_frame_id_offset", type=int, default=0)
    p.add_argument("--slice_prompt_strict", action="store_true")
    p.add_argument("--dit_fsdp", action="store_true")
    p.add_argument("--t5_fsdp", action="store_true")
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--ring_size", type=int, default=1)
    p.add_argument("--save_full_model", action="store_true")
    return p.parse_args()


def _parse_pair(value: str) -> Tuple[int, int]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected pair like '30,52', got {value}")
    return int(parts[0]), int(parts[1])


def _model_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision in ("fp32", "no"):
        return torch.float32
    return torch.bfloat16


def _accelerator_precision(mixed_precision: str) -> str:
    return "no" if mixed_precision == "fp32" else mixed_precision


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


def _save_checkpoint(accelerator, model, optimizer, args, step: int) -> None:
    from diffsynth.diffusion import ModelLogger

    logger = ModelLogger(args.out_dir, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    logger.save_model(accelerator, model, f"step-{step}.safetensors")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_path = os.path.join(args.out_dir, f"train_state_step-{step}.pt")
        accelerator.save(
            {
                "step": int(step),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            state_path,
        )
        # Delete old checkpoints, keeping only the most recent keep_latest_ckpts.
        keep = int(getattr(args, "keep_latest_ckpts", 1))
        if keep > 0:
            import glob as _glob
            ckpts = sorted(
                _glob.glob(os.path.join(args.out_dir, "step-*.safetensors")),
                key=lambda p: int(re.search(r"step-(\d+)", p).group(1)),
            )
            for old in ckpts[:-keep]:
                try:
                    os.remove(old)
                    old_step = int(re.search(r"step-(\d+)", old).group(1))
                    old_state = os.path.join(args.out_dir, f"train_state_step-{old_step}.pt")
                    if os.path.exists(old_state):
                        os.remove(old_state)
                    logging.info("[save] removed old checkpoint: %s", old)
                except OSError as e:
                    logging.warning("[save] could not remove %s: %s", old, e)


def _maybe_load_optimizer_state(accelerator, optimizer, resume_ckpt: str) -> int:
    if not resume_ckpt:
        return 0
    step = _extract_step_from_path(resume_ckpt)
    state_path = Path(resume_ckpt).with_name(f"train_state_step-{step}.pt")
    if state_path.exists():
        obj = torch.load(str(state_path), map_location="cpu")
        if "optimizer_state_dict" in obj:
            optimizer.load_state_dict(obj["optimizer_state_dict"])
            if accelerator.is_main_process:
                logging.info("[resume] optimizer state loaded: %s", state_path)
        return int(obj.get("step", step))
    return step


@record
def main(default_stage: Optional[str] = None) -> None:
    args = parse_args(default_stage=default_stage)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    if args.batch_size != 1:
        raise ValueError("DiffSynth Wan TI2V training currently expects --batch_size 1.")
    if (args.gen_len + 1 - 1) % 4 != 0:
        raise ValueError(f"gen_len+1 must be 4n+1 for Wan VAE time layout, got {args.gen_len + 1}")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = args.metrics_csv.strip() or os.path.join(args.out_dir, "train_metrics.csv")
    metrics_svg = args.metrics_svg.strip() or os.path.join(args.out_dir, "train_loss.svg")
    progress_json = args.progress_json.strip() or os.path.join(args.out_dir, "progress.json")
    progress_svg = args.progress_svg.strip() or os.path.join(args.out_dir, "train_progress.svg")
    progress_t0 = time.time()
    progress_is_main = [os.environ.get("RANK", "0") in ("", "0")]

    def save_progress(
        phase: str,
        status: str,
        step: int = 0,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        loss_ema_value: Optional[float] = None,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        eta_sec: Optional[float] = None,
        source: str = "",
        anchor_fid: Optional[int] = None,
    ) -> None:
        if not progress_is_main[0]:
            return
        total = max(int(total_steps or args.max_steps), 1)
        current = max(int(step), 0)
        payload: Dict[str, object] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": args.stage,
            "phase": phase,
            "status": status,
            "step": current,
            "total_steps": total,
            "percent": min(max(current / total, 0.0), 1.0),
            "elapsed_sec": time.time() - progress_t0,
            "eta_sec": eta_sec,
            "loss": loss,
            "loss_ema": loss_ema_value,
            "grad_norm": grad_norm,
            "lr": lr,
            "source": source,
            "anchor_fid": anchor_fid,
            "rank": int(os.environ.get("RANK", "0") or 0),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0") or 0),
            "world_size": int(os.environ.get("WORLD_SIZE", "1") or 1),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "out_dir": args.out_dir,
            "metrics_csv": metrics_csv,
            "metrics_svg": metrics_svg,
            "progress_json": progress_json,
            "progress_svg": progress_svg,
        }
        _write_progress_json(progress_json, payload)
        _write_progress_svg(progress_svg, payload)

    save_progress("initializing", "parsed arguments")
    logging.info("[progress] json=%s", progress_json)
    logging.info("[progress] svg=%s", progress_svg)
    logging.info("[init] rank=%s local_rank=%s world_size=%s cuda_visible_devices=%s",
                 os.environ.get("RANK", "0"),
                 os.environ.get("LOCAL_RANK", "0"),
                 os.environ.get("WORLD_SIZE", "1"),
                 os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    _ensure_diffsynth_on_path(args.diffsynth_root)
    save_progress("initializing", "DiffSynth-Studio is on PYTHONPATH")
    from accelerate import Accelerator

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    width, height = _parse_size(args.size)
    model_dir = args.model_dir.strip() or args.ckpt_dir
    target_modules = args.lora_target_modules.strip() or args.lora_keywords.strip() or "q,k,v,o,ffn.0,ffn.2"
    trainable_models = None if args.use_lora else (args.trainable_models.strip() or "dit")
    prompt = args.prompt.strip()
    if args.stage == "stage1" and not prompt:
        prompt = DEFAULT_PROMPT

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset_len = max(args.max_steps * max(world_size, 1) * max(args.gradient_accumulation_steps, 1) * 2, 32)
    strict_anchor_match = not args.non_strict_anchor_match
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    save_progress("dataset", "scanning training windows")
    dataset = WanTi2VWindowDataset(
        stage=args.stage,
        width=width,
        height=height,
        cond_len=args.cond_len,
        gen_len=args.gen_len,
        frame_step=args.frame_step,
        pad_mode=args.pad_mode,
        max_items=dataset_len,
        prompt=prompt,
        train_video_dir=args.train_video_dir,
        train_frames_root=args.train_frames_root,
        dataset_root=args.dataset_root,
        splits=splits,
        frames_dirname=args.frames_dirname,
        frame_embed_dirname=args.frame_embed_dirname,
        sample_stride=args.sample_stride,
        prompt_pattern=args.prompt_pattern,
        prompt_match=args.prompt_match,
        prompt_strict=bool(args.prompt_strict or args.slice_prompt_strict),
        strict_anchor_match=strict_anchor_match,
        replay_video_dir=args.replay_video_dir,
        replay_frames_root=args.replay_frames_root,
        replay_ratio=args.replay_ratio,
        replay_prompt=args.replay_prompt,
        prompt_prefix=args.prompt_prefix,
    )

    save_progress("accelerator", "creating Accelerator")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=_accelerator_precision(args.mixed_precision),
    )
    progress_is_main[0] = bool(accelerator.is_main_process)
    if accelerator.is_main_process:
        logging.info("[accelerator] device=%s num_processes=%s mixed_precision=%s",
                     accelerator.device,
                     accelerator.num_processes,
                     _accelerator_precision(args.mixed_precision))
    save_progress("accelerator", "Accelerator ready")

    resume_lora = args.resume_ckpt.strip() or None
    frozen_lora_ckpt = args.frozen_lora_ckpt.strip() or None
    if accelerator.is_main_process:
        logging.info("[model] loading Wan2.2-TI2V-5B from %s", model_dir)
        if frozen_lora_ckpt:
            logging.info(
                "[model] frozen_lora_ckpt=%s rank=%d alpha=%d (will be fused into base weights)",
                frozen_lora_ckpt, args.frozen_lora_rank, args.frozen_lora_alpha,
            )
    save_progress("model", f"loading Wan2.2-TI2V-5B from {model_dir}")
    model = DiffSynthWanTi2VTrainingModule(
        model_dir=model_dir,
        torch_dtype=_model_dtype(args.mixed_precision),
        device=accelerator.device,
        trainable_models=trainable_models,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=target_modules,
        lora_last_n_blocks=args.lora_last_n_blocks,
        lora_checkpoint=resume_lora,
        frozen_lora_ckpt=frozen_lora_ckpt,
        frozen_lora_rank=args.frozen_lora_rank,
        frozen_lora_alpha=args.frozen_lora_alpha,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        cfg_dropout_p=args.cfg_dropout_p,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tiled=args.tiled,
        tile_size=_parse_pair(args.tile_size),
        tile_stride=_parse_pair(args.tile_stride),
    )
    save_progress("model", "model loaded and LoRA attached")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check LoRA target modules or trainable_models.")
    local_signature = _trainable_param_signature(model)
    if accelerator.is_main_process:
        logging.info(
            "[trainable-signature] local count=%s digest=%s first=%s",
            local_signature["count"],
            local_signature["digest"],
            local_signature["first"][:3],
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: List[Optional[Dict[str, object]]] = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, local_signature)
        digests = [str(x.get("digest")) if x is not None else "" for x in gathered]
        counts = [int(x.get("count")) if x is not None else -1 for x in gathered]
        if accelerator.is_main_process:
            logging.info("[trainable-signature] all ranks counts=%s digests=%s", counts, digests)
        if len(set(digests)) != 1:
            detail = json.dumps(gathered, ensure_ascii=False, indent=2)
            raise RuntimeError(
                "Trainable parameter signatures differ across DDP ranks before accelerator.prepare. "
                "This is usually caused by nondeterministic LoRA module matching or different model variants "
                f"being loaded per rank. Signatures:\n{detail}"
            )
    if accelerator.is_main_process:
        n_all = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in trainable_params)
        logging.info(
            "[cfg] stage=%s model_dir=%s size=%sx%s cond_len=%d gen_len=%d train_frames=%d",
            args.stage,
            model_dir,
            width,
            height,
            args.cond_len,
            args.gen_len,
            args.gen_len + 1,
        )
        logging.info(
            "[cfg] use_lora=%s lora_rank=%d lora_alpha=%d target_modules=%s",
            args.use_lora,
            args.lora_rank,
            args.lora_alpha,
            target_modules,
        )
        logging.info("[trainable] total=%d trainable=%d ratio=%.6f", n_all, n_train, n_train / max(n_all, 1))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.warmup_steps > 0:
        _set_lr(optimizer, 0.0)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
        pin_memory=False,
    )
    save_progress("accelerator", "preparing model, optimizer, and dataloader")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    save_progress("accelerator", "prepare complete")

    start_step = _extract_step_from_path(args.resume_ckpt)
    if args.resume_optim:
        start_step = _maybe_load_optimizer_state(accelerator, optimizer, args.resume_ckpt)
    stop_step = start_step + args.max_steps_add if args.resume_ckpt and args.max_steps_add > 0 else args.max_steps
    if start_step >= stop_step:
        raise ValueError(f"start_step={start_step} >= stop_step={stop_step}")

    target_lr = args.learning_rate
    if args.warmup_steps > 0:
        if start_step >= args.warmup_steps:
            _set_lr(optimizer, target_lr)
        elif start_step > 0:
            _set_lr(optimizer, target_lr * start_step / args.warmup_steps)

    loss_steps: List[int] = []
    loss_vals: List[float] = []
    loss_ema: Optional[float] = None
    if accelerator.is_main_process:
        _ensure_dir_for_file(metrics_csv)
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "loss_ema", "grad_norm", "lr", "elapsed_sec", "source", "anchor_fid"])
        logging.info("[metrics] csv=%s", metrics_csv)
        logging.info("[metrics] svg=%s", metrics_svg)
        logging.info("[progress] json=%s", progress_json)
        logging.info("[progress] svg=%s", progress_svg)

    t0 = time.time()
    global_step = start_step
    data_iter = iter(dataloader)
    progress_bar = _StepProgress(
        total=stop_step,
        initial=start_step,
        disable=(not accelerator.is_main_process) or args.disable_progress_bar,
    )
    save_progress("training", "started", step=global_step, total_steps=stop_step)

    while global_step < stop_step:
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data = next(data_iter)

        model.train()
        with accelerator.accumulate(model):
            loss = model(data)
            accelerator.backward(loss)

            grad_norm_value = 0.0
            if accelerator.sync_gradients:
                if args.grad_clip > 0:
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
                    grad_norm_value = float(grad_norm.detach().float().cpu().item())
                else:
                    total_norm_sq = sum(
                        float(p.grad.detach().float().norm().cpu().item()) ** 2
                        for p in trainable_params
                        if p.grad is not None
                    )
                    grad_norm_value = total_norm_sq ** 0.5

            optimizer.step()
            optimizer.zero_grad()

        if not accelerator.sync_gradients:
            continue

        global_step += 1
        if args.warmup_steps > 0 and global_step <= args.warmup_steps:
            _set_lr(optimizer, target_lr * global_step / args.warmup_steps)
        elif global_step == args.warmup_steps + 1:
            _set_lr(optimizer, target_lr)

        reduced_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
        current_lr = _get_lr(optimizer)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        source = str(meta.get("source", ""))
        anchor_fid = int(meta.get("anchor_fid", -1))

        if accelerator.is_main_process:
            elapsed = time.time() - t0
            loss_scalar = float(reduced_loss)
            if loss_ema is None:
                loss_ema = loss_scalar
            else:
                d = float(args.loss_ema_decay)
                loss_ema = d * loss_ema + (1.0 - d) * loss_scalar
            loss_steps.append(global_step)
            loss_vals.append(loss_scalar)
            done_steps = max(global_step - start_step, 1)
            eta_sec = (elapsed / done_steps) * max(stop_step - global_step, 0)
            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step,
                    f"{loss_scalar:.8f}",
                    f"{loss_ema:.8f}",
                    f"{grad_norm_value:.6f}",
                    f"{current_lr:.2e}",
                    f"{elapsed:.2f}",
                    source,
                    anchor_fid,
                ])
            if args.plot_interval > 0 and (global_step % args.plot_interval == 0 or global_step == stop_step):
                _write_loss_svg(metrics_svg, loss_steps, loss_vals)
            progress_bar.update(global_step, loss=loss_scalar, lr=current_lr)
            if args.progress_interval > 0 and (global_step % args.progress_interval == 0 or global_step == stop_step):
                save_progress(
                    "training",
                    "running" if global_step < stop_step else "finishing",
                    step=global_step,
                    total_steps=stop_step,
                    loss=loss_scalar,
                    loss_ema_value=loss_ema,
                    grad_norm=grad_norm_value,
                    lr=current_lr,
                    eta_sec=eta_sec,
                    source=source,
                    anchor_fid=anchor_fid,
                )
            if global_step % args.log_interval == 0 or global_step == start_step + 1:
                logging.info(
                    "[train] step=%d/%d loss=%.6f ema=%.6f grad_norm=%.4f lr=%.2e source=%s anchor=%s",
                    global_step,
                    stop_step,
                    loss_scalar,
                    loss_ema,
                    grad_norm_value,
                    current_lr,
                    source,
                    anchor_fid,
                )

        if global_step % args.save_steps == 0 or global_step == stop_step:
            _save_checkpoint(accelerator, model, optimizer, args, global_step)
            if accelerator.is_main_process:
                logging.info("[save] checkpoint -> %s", os.path.join(args.out_dir, f"step-{global_step}.safetensors"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        progress_bar.close()
        save_progress("finished", "training finished", step=global_step, total_steps=stop_step)
        logging.info("Training finished.")


if __name__ == "__main__":
    main()
