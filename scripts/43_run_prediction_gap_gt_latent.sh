#!/usr/bin/env bash
set -euo pipefail

# Prediction Gap / GT-latent upper-bound experiment.
#
# Formal random-slice generation + latent-gap evaluation:
#   cd /home/xuecheng/WAN_clean
#   NUM_CASES=8 GPU_ID=2 bash scripts/43_run_prediction_gap_gt_latent.sh
#
# Reuse an existing three-stage manifest and only run latent-gap evaluation:
#   MANIFEST=outputs/eval_three_stage/paper_revision_full_20260610_190558/manifest.json \
#   RUN_NAME=pred_gap_existing \
#   ONLY_LABELS=stage3 \
#   bash scripts/43_run_prediction_gap_gt_latent.sh
#
# Debug faster:
#   NUM_CASES=2 NUM_INFERENCE_STEPS=20 ONLY_LABELS=stage3 bash scripts/43_run_prediction_gap_gt_latent.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_ENV_DIR="/mnt/data3/zhongqirui/env/SAM"
ENV_DIR="${ENV_DIR:-${DEFAULT_ENV_DIR}}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON is not executable: ${PYTHON}" >&2
  exit 1
fi

WAN_SAM_ROOT="${WAN_SAM_ROOT:-/home/xuecheng/WAN_SAM}"
MODEL_DIR="${MODEL_DIR:-/mnt/data3/zhongqirui/WAN2_2}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-${ROOT_DIR}/DiffSynth-Studio}"
METADATA_CSV="${METADATA_CSV:-${WAN_SAM_ROOT}/finetune/diffsynth_wan22_ti2v/work/cholec80_wan22_ti2v/metadata_val.csv}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/data3/zhongqirui/Cholec80_slice}"

STAGE1_LORA="${STAGE1_LORA:-${ROOT_DIR}/outputs/stage1/finetune_ckpts/c41_g40_20260524_122831/step-2000.safetensors}"
STAGE2_LORA="${STAGE2_LORA:-${ROOT_DIR}/outputs/stage2/finetune_ckpts/c41_g40_s2_20260524_122831/step-800.safetensors}"
STAGE3_LORA="${STAGE3_LORA:-${ROOT_DIR}/outputs/stage3_track20/finetune_ckpts/c41_g40_20260526_172928/step-500.safetensors}"

GPU_ID="${GPU_ID:-2}"
RUN_NAME="${RUN_NAME:-prediction_gap_gt_latent_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/prediction_gap_gt_latent/${RUN_NAME}}"

NUM_CASES="${NUM_CASES:-8}"
RANDOM_SEED="${RANDOM_SEED:-20260610}"
COND_FRAMES="${COND_FRAMES:-41}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-15}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
LORA_ALPHA="${LORA_ALPHA:-1.5}"
SEED="${SEED:-1}"
HORIZONS="${HORIZONS:-1,2,3,4,5}"

# By default, evaluate stage3 only because this is the final model.  Use
# ONLY_LABELS='base,stage1,stage2,stage3' for the full ablation.
ONLY_LABELS="${ONLY_LABELS:-stage3}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
MANIFEST="${MANIFEST:-}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
DECODE_GT_UPPER_BOUND="${DECODE_GT_UPPER_BOUND:-1}"
SAVE_RECON_VIDEOS="${SAVE_RECON_VIDEOS:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_DIR
export PYTHONPATH="${DIFFSYNTH_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p "${RUN_DIR}"

for path in "${MODEL_DIR}/Wan2.2_VAE.pth" "${METADATA_CSV}" "${STAGE1_LORA}" "${STAGE2_LORA}" "${STAGE3_LORA}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
done

ARGS=(
  "${ROOT_DIR}/src/eval/prediction_gap_gt_latent_experiment.py"
  --output_dir "${RUN_DIR}"
  --model_dir "${MODEL_DIR}"
  --metadata_csv "${METADATA_CSV}"
  --dataset_base_path "${VIDEO_DIR}"
  --stage_lora "stage1=${STAGE1_LORA}"
  --stage_lora "stage2=${STAGE2_LORA}"
  --stage_lora "stage3=${STAGE3_LORA}"
  --num_cases "${NUM_CASES}"
  --random_seed "${RANDOM_SEED}"
  --cond_frames "${COND_FRAMES}"
  --num_frames "${NUM_FRAMES}"
  --fps "${FPS}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --lora_alpha "${LORA_ALPHA}"
  --seed "${SEED}"
  --horizons "${HORIZONS}"
  --only_labels "${ONLY_LABELS}"
)

if [[ "${INCLUDE_BASE}" == "1" ]]; then
  ARGS+=(--include_base)
fi
if [[ "${REUSE_EXISTING}" == "1" ]]; then
  ARGS+=(--reuse_existing --skip_existing_outputs)
fi
if [[ "${DECODE_GT_UPPER_BOUND}" == "1" ]]; then
  ARGS+=(--decode_gt_upper_bound)
else
  ARGS+=(--no-decode_gt_upper_bound)
fi
if [[ "${SAVE_RECON_VIDEOS}" == "1" ]]; then
  ARGS+=(--save_recon_videos)
fi
if [[ -n "${MANIFEST}" ]]; then
  ARGS+=(--manifest "${MANIFEST}" --skip_generation)
fi

echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] ONLY_LABELS=${ONLY_LABELS}"
if [[ -n "${MANIFEST}" ]]; then
  echo "[INFO] eval-only MANIFEST=${MANIFEST}"
else
  echo "[INFO] random generation NUM_CASES=${NUM_CASES}"
fi

"${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${RUN_DIR}/prediction_gap_gt_latent.log"

echo "[DONE] ${RUN_DIR}"
echo "[DONE] Metrics: ${RUN_DIR}/prediction_gap_metrics"
echo "[DONE] Figures: ${RUN_DIR}/prediction_gap_figures"
