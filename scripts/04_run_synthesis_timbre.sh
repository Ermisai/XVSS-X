#!/usr/bin/env bash
# ==============================================================================
# Step 4: OmniVoice Speech Synthesis - Timbre-Transferred Mode (XVSS-X-T)
# Zero-shot cross-lingual voice cloning conditioned on source English audio.
# ==============================================================================
set -euo pipefail

TRANSLATED_DIR="${TRANSLATED_DIR:-data/translated}"
OUTPUT_DIR="${OUTPUT_DIR:-data/synthesized}"
LANGS="${LANGS:-all}"
SPLITS="${SPLITS:-test,dev,train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "=================================================================="
echo "  XVSS-X Step 4: OmniVoice Synthesis (XVSS-X-T Timbre)"
echo "  Translated Dir: ${TRANSLATED_DIR}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo "  Languages:      ${LANGS}"
echo "  Splits:         ${SPLITS}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  GPU Device:     ${CUDA_DEV}"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m pipeline.03_synthesize \
    --variant timbre \
    --translated-dir "${TRANSLATED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --languages "${LANGS}" \
    --splits "${SPLITS}" \
    --batch-size "${BATCH_SIZE}"

echo "[Step 4] Timbre synthesis completed. Audio saved in ${OUTPUT_DIR}/"

