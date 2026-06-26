#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/stage2_cholecT50.env"

DATASETS="${LOOKUP_DATASETS:-Training}"
VIDEO_PREFIX="${LOOKUP_VIDEO_PREFIX:-VID}"
ANN_GLOB="${LOOKUP_ANNOTATION_GLOB:-*.json}"
PROMPT_TEMPLATE="${LOOKUP_PROMPT_TEMPLATE:-triplet}"
BATCH_SIZE="${LOOKUP_BATCH_SIZE:-32}"
DEVICE_ID="${LOOKUP_DEVICE_ID:-0}"

if [[ -z "${LOOKUP_LABELS_ROOT:-}" && -n "${CHOLECT50_LABELS_ROOT:-}" ]]; then
  LOOKUP_LABELS_ROOT="${CHOLECT50_LABELS_ROOT}"
fi

if [[ -z "${LOOKUP_BASE_PATH:-}" ]]; then
  LOOKUP_BASE_PATH="${TRAIN_FRAMES_ROOT:-}"
fi

if [[ -z "${LOOKUP_BASE_PATH:-}" ]]; then
  echo "LOOKUP_BASE_PATH is empty. Set LOOKUP_BASE_PATH (or TRAIN_FRAMES_ROOT) in stage2 env." >&2
  exit 1
fi

if [[ -z "${LOOKUP_EMBED_PATH:-}" ]]; then
  echo "LOOKUP_EMBED_PATH is empty in stage2 env." >&2
  exit 1
fi

CMD=(
  python "${ROOT_DIR}/src/encode_text/build_triplet_lookup_bank.py"
  --task "${TASK}"
  --size "${SIZE}"
  --ckpt_dir "${CKPT_DIR}"
  --base_path "${LOOKUP_BASE_PATH}"
  --labels_root "${LOOKUP_LABELS_ROOT:-}"
  --datasets "${DATASETS}"
  --video_prefix "${VIDEO_PREFIX}"
  --annotation_glob "${ANN_GLOB}"
  --prompt_template "${PROMPT_TEMPLATE}"
  --batch_size "${BATCH_SIZE}"
  --device_id "${DEVICE_ID}"
  --out_path "${LOOKUP_EMBED_PATH}"
  --log_level INFO
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
