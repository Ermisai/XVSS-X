#!/usr/bin/env bash
# ==============================================================================
# Step 3: OmniVoice Speech Synthesis - Canonical Mode (XVSS-X-C)
# Uses 2 fixed reference voices per language (male & female, selected by gender).
# ==============================================================================
set -euo pipefail

TRANSLATED_DIR="${TRANSLATED_DIR:-data/translated}"
OUTPUT_DIR="${OUTPUT_DIR:-data/synthesized}"
LANGS="${LANGS:-all}"
SPLITS="${SPLITS:-test,dev,train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
REF_MALE="${REF_MALE:-assets/pt-male.wav}"
REF_FEMALE="${REF_FEMALE:-assets/pt-female.wav}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "=================================================================="
echo "  XVSS-X Step 3: OmniVoice Synthesis (XVSS-X-C Canonical)"
echo "  Translated Dir: ${TRANSLATED_DIR}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo "  Languages:      ${LANGS}"
echo "  Splits:         ${SPLITS}"
echo "  Male Voice:     ${REF_MALE}"
echo "  Female Voice:   ${REF_FEMALE}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  GPU Device:     ${CUDA_DEV}"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m pipeline.03_synthesize \
    --variant canonical \
    --translated-dir "${TRANSLATED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --languages "${LANGS}" \
    --splits "${SPLITS}" \
    --ref-male "${REF_MALE}" \
    --ref-female "${REF_FEMALE}" \
    --batch-size "${BATCH_SIZE}"

echo "[Step 3] Canonical synthesis completed. Audio saved in ${OUTPUT_DIR}/"

