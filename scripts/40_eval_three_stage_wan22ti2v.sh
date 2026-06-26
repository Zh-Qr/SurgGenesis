#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAN_SAM_ROOT="${WAN_SAM_ROOT:-/home/xuecheng/WAN_SAM}"
EVAL_ROOT="${EVAL_ROOT:-${WAN_SAM_ROOT}/wan22_ti2v_eval}"
ENV_DIR="${ENV_DIR:-/mnt/data3/zhongqirui/env/SAM}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
MODEL_DIR="${MODEL_DIR:-/mnt/data3/zhongqirui/WAN2_2}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-${ROOT_DIR}/DiffSynth-Studio}"
VBENCH_DIR="${VBENCH_DIR:-${WAN_SAM_ROOT}/benchmark/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/data3/zhongqirui/vbench}"

GPU_ID="${GPU_ID:-2}"
RUN_NAME="${RUN_NAME:-three_stage_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/eval_three_stage/${RUN_NAME}}"

METADATA_CSV="${METADATA_CSV:-${WAN_SAM_ROOT}/finetune/diffsynth_wan22_ti2v/work/cholec80_wan22_ti2v/metadata_val.csv}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/data3/zhongqirui/Cholec80_slice}"

STAGE1_LORA="${STAGE1_LORA:-${ROOT_DIR}/outputs/stage1/finetune_ckpts/c41_g40_20260524_122831/step-2000.safetensors}"
STAGE2_LORA="${STAGE2_LORA:-${ROOT_DIR}/outputs/stage2/finetune_ckpts/c41_g40_s2_20260524_122831/step-800.safetensors}"
STAGE3_LORA="${STAGE3_LORA:-${ROOT_DIR}/outputs/stage3_track20/finetune_ckpts/c41_g40_20260526_172928/step-500.safetensors}"

NUM_CASES="${NUM_CASES:-8}"
COND_FRAMES="${COND_FRAMES:-41}"
NUM_FRAMES="${NUM_FRAMES:-40}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
LORA_ALPHA="${LORA_ALPHA:-1.5}"
CASE_STRIDE="${CASE_STRIDE:-7}"
FPS="${FPS:-15}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-1}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_MASK_EVAL="${RUN_MASK_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
SPLITS="${SPLITS:-base stage1 stage2 stage3}"
DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
GENERATOR_ONLY_LABELS="${GENERATOR_ONLY_LABELS:-}"
GENERATOR_SKIP_EXISTING_OUTPUTS="${GENERATOR_SKIP_EXISTING_OUTPUTS:-0}"

MASK_RESOLUTION="${MASK_RESOLUTION:-1008}"
MASK_FRAME_STRIDE="${MASK_FRAME_STRIDE:-1}"
MASK_MAX_CASES="${MASK_MAX_CASES:-0}"
MASK_MAX_FRAMES_PER_CASE="${MASK_MAX_FRAMES_PER_CASE:-0}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-1}"
MASK_PROMPT="${MASK_PROMPT:-surgical instrument}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/mnt/data3/zhongqirui/sam/sam3.pt}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${WAN_SAM_ROOT}/adapter/outputs/endovis18_tool_sam3_adapter/best_adapter.pt}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_DIR
export WAN_SAM_ROOT
export VBENCH_CACHE_DIR
export PYTHONPATH="${DIFFSYNTH_DIR}:${VBENCH_DIR}:${WAN_SAM_ROOT}/adapter:${WAN_SAM_ROOT}/sam3:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p "${RUN_DIR}"

for path in "${STAGE1_LORA}" "${STAGE2_LORA}" "${STAGE3_LORA}" "${METADATA_CSV}" "${SAM_CHECKPOINT}" "${ADAPTER_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
done

echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] stage1=${STAGE1_LORA}"
echo "[INFO] stage2=${STAGE2_LORA}"
echo "[INFO] stage3=${STAGE3_LORA}"

if [[ "${RUN_GENERATION}" == "1" ]]; then
  generation_args=(
    "${ROOT_DIR}/src/eval/wan22_three_stage_generate.py"
    --model_dir "${MODEL_DIR}" \
    --output_dir "${RUN_DIR}" \
    --metadata_csv "${METADATA_CSV}" \
    --dataset_base_path "${VIDEO_DIR}" \
    --stage_lora "stage1=${STAGE1_LORA}" \
    --stage_lora "stage2=${STAGE2_LORA}" \
    --stage_lora "stage3=${STAGE3_LORA}" \
    --num_cases "${NUM_CASES}" \
    --cond_frames "${COND_FRAMES}" \
    --case_stride "${CASE_STRIDE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --lora_alpha "${LORA_ALPHA}"
  )
  if [[ -n "${GENERATOR_ONLY_LABELS}" ]]; then
    generation_args+=(--only_labels "${GENERATOR_ONLY_LABELS}")
  fi
  if [[ "${GENERATOR_SKIP_EXISTING_OUTPUTS}" == "1" ]]; then
    generation_args+=(--skip_existing_outputs)
  fi
  "${PYTHON}" "${generation_args[@]}" 2>&1 | tee "${RUN_DIR}/generation.log"
fi

if [[ "${RUN_VBENCH}" == "1" ]]; then
  for split in ${SPLITS}; do
    RUN_DIR="${RUN_DIR}" \
    SPLIT="${split}" \
    GPU_ID="${GPU_ID}" \
    VBENCH_DIR="${VBENCH_DIR}" \
    VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
    DIMENSIONS="${DIMENSIONS}" \
    bash "${EVAL_ROOT}/run_vbench.sh" 2>&1 | tee "${RUN_DIR}/vbench_${split}.log"
  done
fi

if [[ "${RUN_MASK_EVAL}" == "1" ]]; then
  "${PYTHON}" "${ROOT_DIR}/src/eval/wan22_three_stage_mask_eval.py" \
    --manifest "${RUN_DIR}/manifest.json" \
    --output_dir "${RUN_DIR}/mask_results" \
    --repo_root "${WAN_SAM_ROOT}" \
    --sam_checkpoint "${SAM_CHECKPOINT}" \
    --adapter_checkpoint "${ADAPTER_CHECKPOINT}" \
    --prompt "${MASK_PROMPT}" \
    --resolution "${MASK_RESOLUTION}" \
    --batch_size "${MASK_BATCH_SIZE}" \
    --frame_stride "${MASK_FRAME_STRIDE}" \
    --max_cases "${MASK_MAX_CASES}" \
    --max_frames_per_case "${MASK_MAX_FRAMES_PER_CASE}" \
    2>&1 | tee "${RUN_DIR}/mask_eval.log"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  "${PYTHON}" "${ROOT_DIR}/src/eval/summarize_three_stage_eval.py" \
    --run_dir "${RUN_DIR}" \
    --splits "$(tr ' ' ',' <<< "${SPLITS}")" \
    2>&1 | tee "${RUN_DIR}/summary.log"
fi

echo "[DONE] ${RUN_DIR}"
