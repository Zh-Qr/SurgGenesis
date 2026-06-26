#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/stage2_cholecT50.env"

OUT_DIR_BASE="${OUT_DIR}"
RUN_STAMP_DEFAULT="$(date +"%Y%m%d_%H%M%S")"
RUN_STAMP="${RUN_STAMP:-${RUN_STAMP_DEFAULT}}"
OUT_DIR="${OUT_DIR_BASE}/c${COND_LEN}_g${GEN_LEN}_${RUN_STAMP}"
mkdir -p "${OUT_DIR}"

if [[ -n "${METRICS_CSV:-}" ]]; then
  METRICS_CSV="${METRICS_CSV/#${OUT_DIR_BASE}/${OUT_DIR}}"
fi
if [[ -n "${METRICS_SVG:-}" ]]; then
  METRICS_SVG="${METRICS_SVG/#${OUT_DIR_BASE}/${OUT_DIR}}"
fi
if [[ -n "${PROGRESS_JSON:-}" ]]; then
  PROGRESS_JSON="${PROGRESS_JSON/#${OUT_DIR_BASE}/${OUT_DIR}}"
fi
if [[ -n "${PROGRESS_SVG:-}" ]]; then
  PROGRESS_SVG="${PROGRESS_SVG/#${OUT_DIR_BASE}/${OUT_DIR}}"
fi

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  else
    NPROC_PER_NODE=1
  fi
fi

VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE_GPU_COUNT="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi
if [[ "${NPROC_PER_NODE}" -gt 1 && "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} is larger than visible GPUs (${VISIBLE_GPU_COUNT}) from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if [[ -n "${TRAIN_FRAMES_ROOT:-}" ]]; then
  SOURCE_ARGS=(--train_frames_root "${TRAIN_FRAMES_ROOT}")
else
  SOURCE_ARGS=(--train_video_dir "${TRAIN_VIDEO_DIR}")
fi

CMD=(
  "${ROOT_DIR}/src/finetune/train_vace_dit_unified.py"
  --stage stage2
  --diffsynth_root "${DIFFSYNTH_ROOT:-${ROOT_DIR}/DiffSynth-Studio}"
  --task "${TASK}"
  --size "${SIZE}"
  --ckpt_dir "${CKPT_DIR}"
  "${SOURCE_ARGS[@]}"
  --prompt_pattern "${PROMPT_PATTERN:-*_prompts.json}"
  --prompt_match "${SLICE_PROMPT_MATCH:-exact}"
  --cond_len "${COND_LEN}"
  --gen_len "${GEN_LEN}"
  --frame_step "${FRAME_STEP:-1}"
  --pad_mode "${PAD_MODE}"
  --batch_size "${BATCH_SIZE:-1}"
  --max_steps "${MAX_STEPS}"
  --lr "${LR}"
  --weight_decay "${WD}"
  --seed "${SEED}"
  --log_interval "${LOG_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --out_dir "${OUT_DIR}"
  --mixed_precision "${MIXED_PRECISION:-bf16}"
  --metrics_csv "${METRICS_CSV}"
  --metrics_svg "${METRICS_SVG}"
  --progress_json "${PROGRESS_JSON:-${OUT_DIR}/progress.json}"
  --progress_svg "${PROGRESS_SVG:-${OUT_DIR}/train_progress.svg}"
  --progress_interval "${PROGRESS_INTERVAL:-1}"
  --plot_interval "${PLOT_INTERVAL:-1}"
  --warmup_steps "${WARMUP_STEPS:-100}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --cfg_dropout_p "${CFG_DROPOUT_P:-0.1}"
  --dataset_num_workers "${DATASET_NUM_WORKERS:-0}"
  --log_level INFO
)

if [[ "${SLICE_PROMPT_STRICT:-1}" == "1" ]]; then
  CMD+=(--prompt_strict)
fi

if [[ "${TILED:-0}" == "1" ]]; then
  CMD+=(--tiled)
fi

if [[ "${DISABLE_PROGRESS_BAR:-0}" == "1" ]]; then
  CMD+=(--disable_progress_bar)
fi

if [[ "${USE_LORA:-1}" == "1" ]]; then
  CMD+=(
    --use_lora
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_modules "${LORA_TARGET_MODULES:-${LORA_KEYWORDS:-q,k,v,o,ffn.0,ffn.2}}"
    --lora_last_n_blocks "${LORA_LAST_N_BLOCKS:-0}"
  )
fi

# Additive LoRA stacking: fuse prior-stage LoRA into base weights.
if [[ -n "${FROZEN_LORA_CKPT:-}" ]]; then
  CMD+=(
    --frozen_lora_ckpt "${FROZEN_LORA_CKPT}"
    --frozen_lora_rank "${FROZEN_LORA_RANK:-32}"
    --frozen_lora_alpha "${FROZEN_LORA_ALPHA:-32}"
  )
fi

# Data replay for catastrophic-forgetting prevention.
if [[ -n "${REPLAY_VIDEO_DIR:-}" ]]; then
  CMD+=(--replay_video_dir "${REPLAY_VIDEO_DIR}")
fi
if [[ -n "${REPLAY_FRAMES_ROOT:-}" ]]; then
  CMD+=(--replay_frames_root "${REPLAY_FRAMES_ROOT}")
fi
if [[ "${REPLAY_RATIO:-0}" != "0" ]]; then
  CMD+=(--replay_ratio "${REPLAY_RATIO}")
fi
if [[ -n "${REPLAY_PROMPT:-}" ]]; then
  CMD+=(--replay_prompt "${REPLAY_PROMPT}")
fi

# Prompt prefix for text-distribution harmonisation.
if [[ -n "${PROMPT_PREFIX:-}" ]]; then
  CMD+=(--prompt_prefix "${PROMPT_PREFIX}")
fi

# Legacy resume (kept for mid-stage warm-starts, not cross-stage transfer).
if [[ -n "${RESUME_CKPT:-}" ]]; then
  CMD+=(--resume_ckpt "${RESUME_CKPT}")
fi
if [[ "${RESUME_OPTIM:-0}" == "1" ]]; then
  CMD+=(--resume_optim)
fi

ACCEL_MP="${MIXED_PRECISION:-bf16}"
if [[ "${ACCEL_MP}" == "fp32" ]]; then ACCEL_MP="no"; fi

if [[ "${NPROC_PER_NODE}" -gt 1 || "${NNODES}" -gt 1 ]]; then
  TOTAL_PROCESSES=$((NPROC_PER_NODE * NNODES))
  LAUNCH=(
    "${ACCELERATE_BIN}" launch
    --num_processes "${TOTAL_PROCESSES}"
    --num_machines "${NNODES}"
    --machine_rank "${NODE_RANK}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --mixed_precision "${ACCEL_MP}"
  )
  if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
    LAUNCH+=(--config_file "${ACCELERATE_CONFIG}")
  fi
else
  LAUNCH=("${PYTHON_BIN}")
fi

printf 'Running command:\n%s %s\n' "${LAUNCH[*]}" "${CMD[*]}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
if [[ -n "${METRICS_CSV:-}" ]]; then echo "[INFO] METRICS_CSV=${METRICS_CSV}"; fi
if [[ -n "${METRICS_SVG:-}" ]]; then echo "[INFO] METRICS_SVG=${METRICS_SVG}"; fi
echo "[INFO] PROGRESS_JSON=${PROGRESS_JSON:-${OUT_DIR}/progress.json}"
echo "[INFO] PROGRESS_SVG=${PROGRESS_SVG:-${OUT_DIR}/train_progress.svg}"

"${LAUNCH[@]}" "${CMD[@]}"
