#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import torch

# ==== WAN 路径设置 ====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "backbone", "Wan2.1"))
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan  # type: ignore
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES  # type: ignore


def parse_args():
    p = argparse.ArgumentParser("Pre-encode prompt/n_prompt with WAN T5 encoder and save embeddings.")

    p.add_argument("--task", type=str, default="vace-1.3B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--ckpt_dir", type=str, required=True)

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--n_prompt", type=str, default="")

    p.add_argument("--out_path", type=str, default="finetune_ckpts/text_embeds.pt")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


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

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"[encode] task={args.task} size={args.size}")
    logging.info(f"[encode] ckpt_dir={args.ckpt_dir}")
    logging.info(f"[encode] prompt={args.prompt}")
    logging.info(f"[encode] n_prompt={args.n_prompt}")

    # 这里让 T5 在 GPU 上跑：t5_cpu=False
    # 其他模块（DiT / VAE）虽然也会 load，但我们只用 T5 跑一次，40G 显存够用。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = wan.WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,    # 用 GPU:0
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,   # 关键：让 text_encoder 默认在 GPU
    )

    # 若未显式提供 n_prompt，就用官方默认的 sample_neg_prompt
    if not args.n_prompt:
        n_prompt = pipe.sample_neg_prompt
    else:
        n_prompt = args.n_prompt

    logging.info("[encode] running text encoder on GPU...")

    # 保证 text_encoder 在指定 device 上
    pipe.text_encoder.model.to(device)

    ctx = pipe.text_encoder([args.prompt], device)
    ctx_null = pipe.text_encoder([n_prompt], device)

    def _to_cpu_f32_list(xs):
        if isinstance(xs, (list, tuple)):
            return [t.to(device="cpu", dtype=torch.float32) for t in xs]
        return xs.to(device="cpu", dtype=torch.float32)

    ctx = _to_cpu_f32_list(ctx)
    ctx_null = _to_cpu_f32_list(ctx_null)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save({"context": ctx, "context_null": ctx_null}, args.out_path)
    logging.info(f"[encode] saved embeddings -> {args.out_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
