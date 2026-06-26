#!/usr/bin/env bash
set -euo pipefail

# Trustworthy paper-revision evaluation.
#
# This script is intended for paper numbers, not smoke testing. It:
#   1. Generates a fresh Wan2.2-TI2V base/stage1/stage2/stage3 evaluation run.
#   2. Refuses to silently reuse stale outputs unless EVAL_EXISTING=1.
#   3. Checks that stage videos are not byte-identical.
#   4. Runs distribution-level Frechet video distance-style evaluation using
#      Kinetics-pretrained torchvision R3D-18 features.
#   5. Runs temporal horizon analysis and exports publication-style PDF/PNG.
#
# Minimal use:
#   cd /home/xuecheng/WAN_clean
#   bash scripts/42_run_trustworthy_paper_revision_eval.sh
#
# Useful overrides:
#   GPU_ID=2 NUM_CASES=16 NUM_FRAMES=81 bash scripts/42_run_trustworthy_paper_revision_eval.sh
#   EVAL_EXISTING=1 RUN_DIR=outputs/eval_three_stage/<run_name> bash scripts/42_run_trustworthy_paper_revision_eval.sh
#   EXTRA_VIDEOS='EndoGen=/path/to/endogen,Surg-World=/path/to/surg_world' bash scripts/42_run_trustworthy_paper_revision_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_ENV_DIR="/mnt/data3/zhongqirui/env/SAM"
ENV_DIR="${ENV_DIR:-${DEFAULT_ENV_DIR}}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON is not executable: ${PYTHON}" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-paper_revision_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/eval_three_stage/${RUN_NAME}}"
MANIFEST="${RUN_DIR}/manifest.json"
PAPER_OUT_DIR="${PAPER_OUT_DIR:-${RUN_DIR}/paper_revision_metrics}"
PAPER_FIG_DIR="${PAPER_FIG_DIR:-${RUN_DIR}/paper_figures}"

# Evaluation size. 81 generated frames at 15 fps covers roughly 5 seconds after
# the first generated/reference frame alignment.
NUM_CASES="${NUM_CASES:-16}"
MIN_CASES="${MIN_CASES:-8}"
COND_FRAMES="${COND_FRAMES:-41}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-15}"
HORIZONS="${HORIZONS:-1,2,3,4,5}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
LORA_ALPHA="${LORA_ALPHA:-1.5}"
CASE_STRIDE="${CASE_STRIDE:-7}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-2}"

# Formal metric settings. Do not use the pixel backend for paper numbers.
FVD_BACKEND="${FVD_BACKEND:-torchvision-r3d18}"
FVD_DEVICE="${FVD_DEVICE:-cuda}"
FVD_NUM_FRAMES="${FVD_NUM_FRAMES:-16}"
FVD_MAX_VIDEOS="${FVD_MAX_VIDEOS:-0}"
SPLITS_COMMA="${SPLITS_COMMA:-base,stage1,stage2,stage3}"
SPLITS_SPACE="${SPLITS_COMMA//,/ }"

# Optional expensive add-ons from scripts/40_eval_three_stage_wan22ti2v.sh.
RUN_VBENCH="${RUN_VBENCH:-0}"
RUN_MASK_EVAL="${RUN_MASK_EVAL:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"

# Safety switches.
EVAL_EXISTING="${EVAL_EXISTING:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"
ALLOW_SMALL_N="${ALLOW_SMALL_N:-0}"
ALLOW_SHORT_HORIZON="${ALLOW_SHORT_HORIZON:-0}"
CHECK_DUPLICATES="${CHECK_DUPLICATES:-1}"
ALLOW_DUPLICATE_STAGE_OUTPUTS="${ALLOW_DUPLICATE_STAGE_OUTPUTS:-0}"
GENERATOR_ONLY_LABELS="${GENERATOR_ONLY_LABELS:-}"
GENERATOR_SKIP_EXISTING_OUTPUTS="${GENERATOR_SKIP_EXISTING_OUTPUTS:-0}"

log() { echo "[paper-eval $(date +%H:%M:%S)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

require_cmd sha256sum

if [[ "${FVD_BACKEND}" != "torchvision-r3d18" ]]; then
  die "FVD_BACKEND=${FVD_BACKEND} is not allowed for formal paper runs. Use scripts/41... for smoke tests."
fi

if [[ "${NUM_CASES}" -lt "${MIN_CASES}" && "${ALLOW_SMALL_N}" != "1" ]]; then
  die "NUM_CASES=${NUM_CASES} is smaller than MIN_CASES=${MIN_CASES}. Set ALLOW_SMALL_N=1 only for debugging."
fi

"${PYTHON}" - "${NUM_FRAMES}" "${FPS}" "${HORIZONS}" "${ALLOW_SHORT_HORIZON}" <<'PY'
import sys
num_frames = int(sys.argv[1])
fps = float(sys.argv[2])
horizons = [float(x) for x in sys.argv[3].split(",") if x.strip()]
allow = sys.argv[4] == "1"
coverage = max(0, num_frames - 1) / max(fps, 1e-9)
need = max(horizons) if horizons else 0
if coverage + 1e-9 < need and not allow:
    raise SystemExit(
        f"NUM_FRAMES={num_frames} at FPS={fps:g} covers only {coverage:.2f}s, "
        f"but HORIZONS needs {need:.2f}s. Increase NUM_FRAMES or set ALLOW_SHORT_HORIZON=1."
    )
print(f"[check] temporal coverage: {(num_frames - 1) / fps:.2f}s for horizons={horizons}")
PY

if [[ "${EVAL_EXISTING}" == "1" ]]; then
  require_file "${MANIFEST}"
  log "EVAL_EXISTING=1, using existing run: ${RUN_DIR}"
else
  if [[ "${RESUME_GENERATION}" == "1" ]]; then
    [[ -d "${RUN_DIR}" ]] || die "RESUME_GENERATION=1 but RUN_DIR does not exist: ${RUN_DIR}"
    GENERATOR_SKIP_EXISTING_OUTPUTS=1
    log "RESUME_GENERATION=1, reusing existing videos and generating only missing outputs in ${RUN_DIR}"
  fi

  if [[ -e "${MANIFEST}" && "${FORCE_RERUN}" != "1" && "${RESUME_GENERATION}" != "1" ]]; then
    die "Refusing to overwrite/reuse existing manifest: ${MANIFEST}. Use EVAL_EXISTING=1, RESUME_GENERATION=1, or FORCE_RERUN=1."
  fi
  if [[ "${FORCE_RERUN}" == "1" && "${RESUME_GENERATION}" != "1" && -d "${RUN_DIR}" ]]; then
    log "FORCE_RERUN=1, removing old RUN_DIR=${RUN_DIR}"
    rm -rf "${RUN_DIR}"
  fi

  log "Generating videos with Wan2.2-TI2V-5B"
  RUN_DIR="${RUN_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  GPU_ID="${GPU_ID}" \
  NUM_CASES="${NUM_CASES}" \
  COND_FRAMES="${COND_FRAMES}" \
  NUM_FRAMES="${NUM_FRAMES}" \
  FPS="${FPS}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  CFG_SCALE="${CFG_SCALE}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  CASE_STRIDE="${CASE_STRIDE}" \
  SEED="${SEED}" \
  RUN_GENERATION=1 \
  RUN_VBENCH="${RUN_VBENCH}" \
  RUN_MASK_EVAL="${RUN_MASK_EVAL}" \
  RUN_SUMMARY="${RUN_SUMMARY}" \
  SPLITS="${SPLITS_SPACE}" \
  GENERATOR_ONLY_LABELS="${GENERATOR_ONLY_LABELS}" \
  GENERATOR_SKIP_EXISTING_OUTPUTS="${GENERATOR_SKIP_EXISTING_OUTPUTS}" \
  bash "${ROOT_DIR}/scripts/40_eval_three_stage_wan22ti2v.sh"
fi

require_file "${MANIFEST}"
mkdir -p "${PAPER_OUT_DIR}" "${PAPER_FIG_DIR}"

log "Auditing manifest and generated video files"
"${PYTHON}" - "${MANIFEST}" "${MIN_CASES}" "${ALLOW_SMALL_N}" "${SPLITS_COMMA}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
min_cases = int(sys.argv[2])
allow_small = sys.argv[3] == "1"
labels = [x.strip() for x in sys.argv[4].split(",") if x.strip()]
data = json.loads(manifest.read_text(encoding="utf-8"))
cases = data.get("cases", [])
if len(cases) < min_cases and not allow_small:
    raise SystemExit(f"Only {len(cases)} cases in manifest, expected at least {min_cases}.")
missing = []
for case in cases:
    if not Path(case.get("ground_truth_future", "")).is_file():
        missing.append((case.get("case"), "ground_truth_future", case.get("ground_truth_future", "")))
    generated = case.get("generated") or {}
    for label in labels:
        if label not in generated:
            missing.append((case.get("case"), label, "<missing label>"))
        elif not Path(generated[label]).is_file():
            missing.append((case.get("case"), label, generated[label]))
if missing:
    msg = "\n".join(f"{case} {label}: {path}" for case, label, path in missing[:20])
    raise SystemExit(f"Missing generated files:\n{msg}")
print(f"[check] manifest cases={len(cases)} labels={labels}")
PY

if [[ "${CHECK_DUPLICATES}" == "1" ]]; then
  log "Checking duplicate generated videos by SHA256"
  duplicate_report="${PAPER_OUT_DIR}/duplicate_video_audit.txt"
  : > "${duplicate_report}"
  duplicate_count=0
  for case_dir in "${RUN_DIR}"/cases/case_*; do
    [[ -d "${case_dir}" ]] || continue
    declare -A seen=()
    IFS=',' read -r -a labels_array <<< "${SPLITS_COMMA}"
    for label in "${labels_array[@]}"; do
      video="${case_dir}/${label}_generated.mp4"
      [[ -f "${video}" ]] || continue
      digest="$(sha256sum "${video}" | awk '{print $1}')"
      if [[ -n "${seen[${digest}]:-}" ]]; then
        echo "$(basename "${case_dir}") duplicate: ${seen[${digest}]} == ${label} (${digest})" | tee -a "${duplicate_report}"
        duplicate_count=$((duplicate_count + 1))
      else
        seen["${digest}"]="${label}"
      fi
    done
    unset seen
  done
  if [[ "${duplicate_count}" -gt 0 && "${ALLOW_DUPLICATE_STAGE_OUTPUTS}" != "1" ]]; then
    die "Found ${duplicate_count} duplicate generated videos. See ${duplicate_report}. This run is not trustworthy for stage comparisons."
  fi
  log "Duplicate audit written to ${duplicate_report}"
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_VIDEOS:-}" ]]; then
  IFS=',' read -r -a extra_specs <<< "${EXTRA_VIDEOS}"
  for spec in "${extra_specs[@]}"; do
    [[ -n "${spec}" ]] && EXTRA_ARGS+=(--extra_videos "${spec}")
  done
fi

log "Running distribution-level Frechet video distance-style metrics"
FVD_ARGS=(
  "${ROOT_DIR}/src/eval/video_distribution_metrics.py"
  --manifest "${MANIFEST}"
  --output_dir "${PAPER_OUT_DIR}/distribution"
  --labels "${SPLITS_COMMA}"
  --feature_backend "${FVD_BACKEND}"
  --device "${FVD_DEVICE}"
  --num_frames "${FVD_NUM_FRAMES}"
)
if [[ "${FVD_MAX_VIDEOS}" != "0" ]]; then
  FVD_ARGS+=(--max_videos "${FVD_MAX_VIDEOS}")
fi
"${PYTHON}" "${FVD_ARGS[@]}" "${EXTRA_ARGS[@]}"

log "Running temporal horizon analysis"
"${PYTHON}" "${ROOT_DIR}/src/eval/temporal_horizon_analysis.py" \
  --manifest "${MANIFEST}" \
  --output_dir "${PAPER_OUT_DIR}/horizon" \
  --labels "${SPLITS_COMMA}" \
  --horizons "${HORIZONS}" \
  --plot_metric rmse \
  --title "Temporal Horizon Decay"

cp "${PAPER_OUT_DIR}/horizon/horizon_curve.pdf" "${PAPER_FIG_DIR}/fig_horizon.pdf"
cp "${PAPER_OUT_DIR}/horizon/horizon_curve.png" "${PAPER_FIG_DIR}/fig_horizon.png"
cp "${PAPER_OUT_DIR}/distribution/video_distribution_metrics.pdf" "${PAPER_FIG_DIR}/fig_fvd_distribution.pdf"
cp "${PAPER_OUT_DIR}/distribution/video_distribution_metrics.png" "${PAPER_FIG_DIR}/fig_fvd_distribution.png"

REPORT="${PAPER_OUT_DIR}/paper_revision_eval_report.md"
cat > "${REPORT}" <<EOF
# Paper Revision Evaluation Report

- Run directory: \`${RUN_DIR}\`
- Manifest: \`${MANIFEST}\`
- Cases requested: \`${NUM_CASES}\`
- Frames: \`${NUM_FRAMES}\`
- FPS: \`${FPS}\`
- Horizons: \`${HORIZONS}\`
- Splits: \`${SPLITS_COMMA}\`
- Distribution backend: \`${FVD_BACKEND}\`

## Key Outputs

- Distribution metrics: \`${PAPER_OUT_DIR}/distribution/video_distribution_metrics.csv\`
- Distribution figure: \`${PAPER_FIG_DIR}/fig_fvd_distribution.pdf\`
- Horizon summary: \`${PAPER_OUT_DIR}/horizon/horizon_summary.csv\`
- Horizon figure: \`${PAPER_FIG_DIR}/fig_horizon.pdf\`
- Duplicate audit: \`${PAPER_OUT_DIR}/duplicate_video_audit.txt\`

## Notes

This script refuses pixel/smoke features for formal runs and fails on duplicate
stage outputs unless \`ALLOW_DUPLICATE_STAGE_OUTPUTS=1\` is explicitly set.
EOF

log "Done."
log "Report: ${REPORT}"
log "Paper figures: ${PAPER_FIG_DIR}"
