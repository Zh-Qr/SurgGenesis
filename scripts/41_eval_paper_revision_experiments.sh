#!/usr/bin/env bash
set -euo pipefail

# Paper-revision experiment launcher.
#
# This runs the extra metrics requested in experiment_guidance/TODO.md for an
# existing three-stage evaluation run:
#   - FVD-style distribution metric
#   - temporal horizon decay curve
#
# Usage:
#   RUN_DIR=outputs/eval_three_stage/quick4_30steps_vbench_sam \
#   bash scripts/41_eval_paper_revision_experiments.sh
#
# Optional external comparison rows:
#   EXTRA_VIDEOS='EndoGen=/path/to/endogen/videos,Surg-World=/path/to/surg_world/videos'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="/mnt/data3/zhongqirui/env/SAM/bin/python"
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/eval_three_stage/quick4_30steps_vbench_sam}"
MANIFEST="${MANIFEST:-${RUN_DIR}/manifest.json}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/paper_revision_metrics}"
SPLITS="${SPLITS:-base,stage1,stage2,stage3}"

FVD_BACKEND="${FVD_BACKEND:-torchvision-r3d18}"
FVD_DEVICE="${FVD_DEVICE:-cuda}"
FVD_NUM_FRAMES="${FVD_NUM_FRAMES:-16}"
FVD_MAX_VIDEOS="${FVD_MAX_VIDEOS:-0}"
FVD_ALLOW_UNTRAINED="${FVD_ALLOW_UNTRAINED:-0}"

HORIZONS="${HORIZONS:-1,2,3,4,5}"
HORIZON_PLOT_METRIC="${HORIZON_PLOT_METRIC:-rmse}"
MAX_CASES="${MAX_CASES:-0}"
MAX_FRAMES_PER_CASE="${MAX_FRAMES_PER_CASE:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${EXTRA_VIDEOS:-}" ]]; then
  IFS=',' read -r -a EXTRA_VIDEO_SPECS <<< "${EXTRA_VIDEOS}"
  for spec in "${EXTRA_VIDEO_SPECS[@]}"; do
    if [[ -n "${spec}" ]]; then
      EXTRA_ARGS+=(--extra_videos "${spec}")
    fi
  done
fi

FVD_ARGS=(
  "${ROOT_DIR}/src/eval/video_distribution_metrics.py"
  --manifest "${MANIFEST}"
  --output_dir "${OUT_DIR}/distribution"
  --labels "${SPLITS}"
  --feature_backend "${FVD_BACKEND}"
  --device "${FVD_DEVICE}"
  --num_frames "${FVD_NUM_FRAMES}"
)

if [[ "${FVD_MAX_VIDEOS}" != "0" ]]; then
  FVD_ARGS+=(--max_videos "${FVD_MAX_VIDEOS}")
fi
if [[ "${FVD_ALLOW_UNTRAINED}" == "1" ]]; then
  FVD_ARGS+=(--allow_untrained)
fi

"${PYTHON}" "${FVD_ARGS[@]}" "${EXTRA_ARGS[@]}"

"${PYTHON}" "${ROOT_DIR}/src/eval/temporal_horizon_analysis.py" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT_DIR}/horizon" \
  --labels "${SPLITS}" \
  --horizons "${HORIZONS}" \
  --plot_metric "${HORIZON_PLOT_METRIC}" \
  --max_cases "${MAX_CASES}" \
  --max_frames_per_case "${MAX_FRAMES_PER_CASE}" \
  --title "Temporal Horizon Decay"

echo "[DONE] paper revision metrics -> ${OUT_DIR}"
