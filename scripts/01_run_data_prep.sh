#!/usr/bin/env bash
# ==============================================================================
# Step 1: Prepare Source Audio & Splits
# Aligns Common Voice v17 with CVSS (CV v4) metadata.
# ==============================================================================
set -euo pipefail

USE_OFFICIAL_MANIFESTS="${USE_OFFICIAL_MANIFESTS:-false}"
OFFICIAL_METADATA_DIR="${OFFICIAL_METADATA_DIR:-metadata}"
CV4_DIR="${CV4_DIR:-/raid/aluno_lucasgris/Projetos/s2s_translation/kaggle_cache/datasets/vedant2022/common-voice-dataset-version-4/versions/1}"
CV17_DIR="${CV17_DIR:-/raid/aluno_lucasgris/Projetos/s2s_translation/common_voice_17_en}"
OUTPUT_DIR="${OUTPUT_DIR:-data/manifests}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "  XVSS-X Step 1: Source Data Preparation"
echo "  Use Official Manifests: ${USE_OFFICIAL_MANIFESTS}"
if [ "${USE_OFFICIAL_MANIFESTS}" = "true" ]; then
    echo "  Metadata Dir:           ${OFFICIAL_METADATA_DIR}"
    echo "  CV17 Audio:             ${CV17_DIR}"
else
    echo "  CV4 Metadata:           ${CV4_DIR}"
    echo "  CV17 Audio:             ${CV17_DIR}"
fi
echo "  Output Dir:             ${OUTPUT_DIR}"
echo "  Seed:                   ${SEED}"
echo "=================================================================="

if [ "${USE_OFFICIAL_MANIFESTS}" = "true" ]; then
    ${PYTHON:-python3} -m pipeline.01_prepare_source \
        --use-official-manifests \
        --official-metadata-dir "${OFFICIAL_METADATA_DIR}" \
        --cv17-dir "${CV17_DIR}" \
        --output-dir "${OUTPUT_DIR}"
else
    ${PYTHON:-python3} -m pipeline.01_prepare_source \
        --cv4-dir "${CV4_DIR}" \
        --cv17-dir "${CV17_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --seed "${SEED}"
fi

echo "[Step 1] Completed successfully. Manifests saved in ${OUTPUT_DIR}/"


