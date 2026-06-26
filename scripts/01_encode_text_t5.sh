#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/01_encode_text_t5.sh
#   bash scripts/01_encode_text_t5.sh --stage stage2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="stage1"
if [[ $# -ge 2 && "$1" == "--stage" ]]; then
  STAGE="$2"
fi

if [[ "$STAGE" == "stage2" ]]; then
  USER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  source "${ROOT_DIR}/configs/stage2_cholecT50.env"
  if [[ -n "${USER_CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES}"
  fi

  python "${ROOT_DIR}/src/encode_text/batch_encode_text_embeds.py" \
    --task "${TASK}" \
    --size "${SIZE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --base_path "${CHOLECT50_VIDEOS_ROOT}" \
    --pattern "${PROMPT_PATTERN}" \
    --datasets "${DATASETS}" \
    --out_dirname "${OUT_DIRNAME}" \
    --batch_size "${ENCODE_BATCH_SIZE:-${BATCH_SIZE:-8}}" \
    --save_null_once \
    --skip_existing
else
  source "${ROOT_DIR}/configs/stage1_cholec80.env"

  python "${ROOT_DIR}/src/encode_text/encode_text_embed.py" \
    --task "${TASK}" \
    --size "${SIZE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --prompt "${PROMPT}" \
    --n_prompt "${N_PROMPT}" \
    --out_path "${TEXT_EMBED_PATH}" \
    --log_level INFO
fi
