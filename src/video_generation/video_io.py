#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared video IO utilities for generation/testing scripts."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def ensure_even_dimensions(arr: np.ndarray) -> np.ndarray:
    """Ensure H and W are even for H.264-compatible writing.

    Input shape: [T, H, W, C]
    """
    t, h, w, c = arr.shape
    if h % 2 == 0 and w % 2 == 0:
        return arr

    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    out = np.zeros((t, new_h, new_w, c), dtype=arr.dtype)
    out[:, :h, :w, :] = arr
    return out


@torch.no_grad()
def write_video_opencv(video_tchw_01: torch.Tensor, path: str, fps: int = 8) -> None:
    """Write [T,C,H,W] in [0,1] to mp4 using OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-python.")
    if video_tchw_01.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(video_tchw_01.shape)}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    v = (video_tchw_01.clamp(0, 1) * 255.0).to(torch.uint8)
    v = v.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    v = ensure_even_dimensions(v)

    _, h, w, _ = v.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")

    for frame in v:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


@torch.no_grad()
def read_video_segment_cv2(
    video_path: str,
    start_frame: int,
    num_frames: int,
    target_hw: Tuple[int, int],
    pad_mode: str = "repeat_last",
) -> torch.Tensor:
    """Read a contiguous segment as [T,C,H,W] RGB in [0,1]."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-python.")

    h, w = target_hw
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames = []
    last_rgb = None
    for _ in range(num_frames):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            if pad_mode == "error":
                cap.release()
                raise RuntimeError("Video ended early")
            if pad_mode == "repeat_last" and last_rgb is not None:
                rgb = last_rgb
            else:
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last_rgb = rgb

        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)

    cap.release()
    return torch.stack(frames, dim=0)
