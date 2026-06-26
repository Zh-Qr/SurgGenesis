#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/stage2_cholecT50.env"

python "${ROOT_DIR}/src/encode_text/prepare_triplet_prompts.py" \
  --labels_root "${CHOLECT50_LABELS_ROOT}" \
  --videos_root "${CHOLECT50_VIDEOS_ROOT}" \
  --output_suffix "_prompts" \
  --future_horizon "${PROMPT_FUTURE_HORIZON:-${GEN_LEN:-60}}"
