#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal Cholec80 frame dataset for WAN_clean video training.

Expected layout:
  data_root/
    video01/
      frame_000001.jpg
      frame_000002.jpg
      ...
    video02/
      frame_000001.jpg
      ...
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset


class Cholec80VideoSequence(Dataset):
    def __init__(
        self,
        sequence_path: str,
        target_size: Tuple[int, int] = (120, 180),
        max_frames: Optional[int] = None,
    ):
        self.sequence_path = Path(sequence_path)
        self.target_size = target_size
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence not found: {self.sequence_path}")

        files = sorted(self.sequence_path.glob("frame_*.jpg"), key=lambda x: int(x.stem.replace("frame_", "")))
        if not files:
            files = sorted(self.sequence_path.glob("frame_*.png"), key=lambda x: int(x.stem.replace("frame_", "")))

        if max_frames is not None:
            files = files[:max_frames]

        self.image_files = files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.target_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.zeros(1)


class Cholec80(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        target_size: Tuple[int, int] = (120, 180),
        split: str = "train",
        sequences: Optional[List[str]] = None,
        max_frames_per_seq: Optional[int] = None,
    ):
        _ = split
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.dataset_dir}")

        if sequences is None:
            seq_names = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        else:
            seq_names = sequences

        self.video_datasets: List[Cholec80VideoSequence] = []
        for name in seq_names:
            p = self.dataset_dir / name
            if not p.exists():
                continue
            ds = Cholec80VideoSequence(str(p), target_size=target_size, max_frames=max_frames_per_seq)
            if len(ds) > 0:
                self.video_datasets.append(ds)

        if not self.video_datasets:
            raise RuntimeError(f"No non-empty video sequences found under {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.video_datasets)

    def __getitem__(self, idx: int):
        return self.video_datasets[idx]
