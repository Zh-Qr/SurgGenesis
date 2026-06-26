#!/usr/bin/env bash
# Unified three-stage LoRA training launcher.
#
# Runs Stage 1 → Stage 2 → Stage 3 sequentially with:
#   • Additive LoRA stacking  (prior-stage delta fused into base before new LoRA is added)
#   • Data replay             (20 % Cholec80 rehearsal in Stage 2/3)
#   • Prompt prefix           ("Endoscopic surgical scene." prepended to Stage 2/3 prompts)
#   • Unified COND_LEN=41 / GEN_LEN=40 across all stages
#
# Usage:
#   bash scripts/50_train_all_stages.sh
#
# Override per-stage GPU assignment before calling, e.g.:
#   S1_GPUS=0,1 S2_GPUS=2,3 S3_GPUS=2,3 bash scripts/50_train_all_stages.sh
#
# To skip a stage (e.g. Stage 1 already done), set the corresponding SKIP_* var:
#   SKIP_STAGE1=1 FROZEN_LORA_S2=<path_to_s1_ckpt> bash scripts/50_train_all_stages.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLOBAL_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"

log() { echo "[unified $(date +%H:%M:%S)] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Cholec80, fixed prompt, rank-32 LoRA, all 30 blocks
# ─────────────────────────────────────────────────────────────────────────────
S1_COND_LEN=41
S1_GEN_LEN=40
S1_MAX_STEPS=2000
S1_OUT_BASE="/home/xuecheng/WAN_clean/outputs/stage1/finetune_ckpts"
S1_STAMP="${GLOBAL_STAMP}"
S1_OUT_DIR="${S1_OUT_BASE}/c${S1_COND_LEN}_g${S1_GEN_LEN}_${S1_STAMP}"
S1_FINAL_CKPT="${S1_OUT_DIR}/step-${S1_MAX_STEPS}.safetensors"

if [[ "${SKIP_STAGE1:-0}" == "1" ]]; then
  log "SKIP_STAGE1=1 — skipping Stage 1"
  # Allow caller to override frozen LoRA path for Stage 2
  S1_FINAL_CKPT="${FROZEN_LORA_S2:-${S1_FINAL_CKPT}}"
  log "Using Stage-1 checkpoint: ${S1_FINAL_CKPT}"
else
  log "===== Stage 1 (Cholec80, rank=32, all blocks) ====="
  CUDA_VISIBLE_DEVICES="${S1_GPUS:-${CUDA_VISIBLE_DEVICES:-2,3}}" \
    RUN_STAMP="${S1_STAMP}" \
    bash "${ROOT_DIR}/scripts/10_train_stage1.sh"
  log "Stage 1 complete → ${S1_FINAL_CKPT}"
fi

if [[ ! -f "${S1_FINAL_CKPT}" ]]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${S1_FINAL_CKPT}" >&2
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — CholecT50 triplet prompts, rank-16 LoRA, last 20 blocks
#            Stage-1 delta fused into base → new Stage-2 LoRA stacked on top
# ─────────────────────────────────────────────────────────────────────────────
S2_COND_LEN=41
S2_GEN_LEN=40
S2_MAX_STEPS=800
S2_OUT_BASE="/home/xuecheng/WAN_clean/outputs/stage2/finetune_ckpts"
S2_STAMP="s2_${GLOBAL_STAMP}"
S2_OUT_DIR="${S2_OUT_BASE}/c${S2_COND_LEN}_g${S2_GEN_LEN}_${S2_STAMP}"
S2_FINAL_CKPT="${S2_OUT_DIR}/step-${S2_MAX_STEPS}.safetensors"

if [[ "${SKIP_STAGE2:-0}" == "1" ]]; then
  log "SKIP_STAGE2=1 — skipping Stage 2"
  S2_FINAL_CKPT="${FROZEN_LORA_S3:-${S2_FINAL_CKPT}}"
  log "Using Stage-2 checkpoint: ${S2_FINAL_CKPT}"
else
  log "===== Stage 2 (CholecT50, rank=16, blocks 10-29, fuse S1) ====="
  CUDA_VISIBLE_DEVICES="${S2_GPUS:-${CUDA_VISIBLE_DEVICES:-2,3}}" \
    RUN_STAMP="${S2_STAMP}" \
    FROZEN_LORA_CKPT="${S1_FINAL_CKPT}" \
    FROZEN_LORA_RANK=32 \
    FROZEN_LORA_ALPHA=32 \
    bash "${ROOT_DIR}/scripts/20_train_stage2.sh"
  log "Stage 2 complete → ${S2_FINAL_CKPT}"
fi

if [[ ! -f "${S2_FINAL_CKPT}" ]]; then
  echo "[ERROR] Stage-2 checkpoint not found: ${S2_FINAL_CKPT}" >&2
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — CholecTrack20 trajectory prompts, rank-8 LoRA, last 10 blocks
#            Stage-2 delta fused into base (which already contains Stage-1)
# ─────────────────────────────────────────────────────────────────────────────
S3_STAMP="s3_${GLOBAL_STAMP}"

log "===== Stage 3 (CholecTrack20, rank=8, blocks 20-29, fuse S2) ====="
CUDA_VISIBLE_DEVICES="${S3_GPUS:-${CUDA_VISIBLE_DEVICES:-1,2}}" \
  RUN_STAMP="${S3_STAMP}" \
  FROZEN_LORA_CKPT="${S2_FINAL_CKPT}" \
  FROZEN_LORA_RANK=16 \
  FROZEN_LORA_ALPHA=16 \
  bash "${ROOT_DIR}/scripts/30_train_stage3_track20.sh"

log "===== All three stages complete (stamp=${GLOBAL_STAMP}) ====="
log "  Stage-1 ckpt : ${S1_FINAL_CKPT}"
log "  Stage-2 ckpt : ${S2_FINAL_CKPT}"
log "  Stage-3 ckpt : outputs/stage3_track20/finetune_ckpts/c41_g40_${S3_STAMP}/step-500.safetensors"
