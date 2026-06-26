#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/generation.env"

export CUDA_VISIBLE_DEVICES

python "${ROOT_DIR}/src/video_generation/train_wan_video_continue.py" \
  --data_root "${DATA_ROOT}" \
  --vae_path "${VAE_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --dit_init "${DIT_INIT}" \
  --model_size base \
  --use_checkpoint \
  --video_frames 13 --cond_frames 8 \
  --height 128 --width 192 \
  --epochs 50 \
  --batch_size 8 --accum_steps 8 \
  --lr 1e-4 --min_lr 1e-5 --warmup_steps 200 \
  --weight_decay 0.01 --grad_clip 1.0 \
  --num_steps 1000 --t_max_ratio 1.0 \
  --lambda_x0 0.0 \
  --vis_interval 2500 --save_interval 500 --log_interval 500 \
  --ddim_steps 50 --sample_t_ratio 1.0 --fps 8 \
  --vis_num 8 \
  --amp --amp_dtype bf16
