#!/usr/bin/env bash
# ==============================================================================
# Step 5: Full Evaluation Benchmark
# Runs dev sampling (200/lang), ASR-BLEU (Whisper large-v3), UTMOS, ECAPA-TDNN,
# and generates summary Markdown & LaTeX tables matching the paper.
# ==============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/synthesized}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SEED="${SEED:-42}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "=================================================================="
echo "  XVSS-X Step 5: Full Quality Evaluation Suite"
echo "  Data Dir:    ${DATA_DIR}"
echo "  Output Dir:  ${OUTPUT_DIR}"
echo "  Sample Size: ${SAMPLE_SIZE} utterances/lang"
echo "  Seed:        ${SEED}"
echo "  GPU Device:  ${CUDA_DEV}"
echo "=================================================================="

# 1. Sample Dev set
echo "[1/4] Selecting 200 stratified dev samples per language..."
${PYTHON:-python3} -m evaluation.sample_dev --data-dir "${DATA_DIR}" --variant canonical --sample-size "${SAMPLE_SIZE}" --seed "${SEED}"
${PYTHON:-python3} -m evaluation.sample_dev --data-dir "${DATA_DIR}" --variant timbre    --sample-size "${SAMPLE_SIZE}" --seed "${SEED}"

# 2. Canonical Evaluation (ASR-BLEU & UTMOS)
echo "[2/4] Evaluating Canonical Variant (XVSS-X-C)..."
CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m evaluation.evaluate_asr_bleu \
    --samples-file evaluation/samples/eval_samples_all_canonical.json \
    --output-dir "${OUTPUT_DIR}/canonical"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m evaluation.evaluate_utmos \
    --samples-file evaluation/samples/eval_samples_all_canonical.json \
    --output-dir "${OUTPUT_DIR}/canonical"

# 3. Timbre Evaluation (ASR-BLEU, UTMOS, Speaker Similarity)
echo "[3/4] Evaluating Timbre Variant (XVSS-X-T)..."
CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m evaluation.evaluate_asr_bleu \
    --samples-file evaluation/samples/eval_samples_all_timbre.json \
    --output-dir "${OUTPUT_DIR}/timbre"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m evaluation.evaluate_utmos \
    --samples-file evaluation/samples/eval_samples_all_timbre.json \
    --output-dir "${OUTPUT_DIR}/timbre"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" ${PYTHON:-python3} -m evaluation.evaluate_speaker_sim \
    --samples-file evaluation/samples/eval_samples_all_timbre.json \
    --output-dir "${OUTPUT_DIR}/timbre"

# 4. Generate Tables
echo "[4/4] Generating Paper Tables (Table 2 & Table 4)..."
${PYTHON:-python3} -m evaluation.generate_tables \
    --results-dir "${OUTPUT_DIR}" \
    --output-file "evaluation/evaluation_report.md"

echo "=================================================================="
echo "  Evaluation completed! Report saved at evaluation/evaluation_report.md"
echo "=================================================================="

