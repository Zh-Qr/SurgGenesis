#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/generation.env"

export CUDA_VISIBLE_DEVICES

python "${ROOT_DIR}/src/video_generation/generate_vace_compare_gt.py" \
  --task "${TASK}" \
  --size "${SIZE}" \
  --ckpt_dir "${CKPT_DIR}" \
  --video_path "${VIDEO_PATH}" \
  --start_frame "${START_FRAME}" \
  --cond_len "${COND_LEN}" \
  --gen_len "${GEN_LEN}" \
  --future_fill "${FUTURE_FILL}" \
  --prompt "Endoscopic surgical scene, realistic continuous motion, plausible tool and tissue dynamics, steady camera." \
  --n_prompt "" \
  --sample_solver unipc \
  --sample_steps 40 \
  --sample_shift 16.0 \
  --sample_guide_scale 5.0 \
  --context_scale 1.0 \
  --base_seed -1 \
  --offload_model True \
  --save_vs_gt_file "${ROOT_DIR}/outputs/refactor_vs_gt.mp4" \
  --save_pred_file "${ROOT_DIR}/outputs/refactor_pred.mp4" \
  --save_gt_file "${ROOT_DIR}/outputs/refactor_gt.mp4" \
  --save_fps 0 \
  --use_ref_from_cond True \
  --ref_indices "0,5,10,15,20,25,30,35,40" \
  --log_level INFO
