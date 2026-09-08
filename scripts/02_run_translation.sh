#!/usr/bin/env bash
# ==============================================================================
# Step 2: Multilingual NLLB Text Translation (EN -> 28 Languages)
# ==============================================================================
set -euo pipefail

MANIFEST_DIR="${MANIFEST_DIR:-data/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-data/translated}"
LANGS="${LANGS:-all}"
SPLITS="${SPLITS:-test,dev,train}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "=================================================================="
echo "  XVSS-X Step 2: NLLB Multilingual Translation"
echo "  Manifest Dir: ${MANIFEST_DIR}"
echo "  Output Dir:   ${OUTPUT_DIR}"
echo "  Languages:    ${LANGS}"
echo "  Splits:       ${SPLITS}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  GPU Device:   ${CUDA_DEV}"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m pipeline.02_translate \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --languages "${LANGS}" \
    --splits "${SPLITS}" \
    --batch-size "${BATCH_SIZE}"

echo "[Step 2] Completed successfully. Translations saved in ${OUTPUT_DIR}/"

