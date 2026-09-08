#!/usr/bin/env bash
# ==============================================================================
# Setup Environment for XVSS-X Dataset Pipeline and Evaluation Suite
# ==============================================================================
set -euo pipefail

ENV_NAME="xvss-x"
PYTHON_VERSION="3.10"

echo "=================================================================="
echo "  Setting up XVSS-X Environment (Python ${PYTHON_VERSION})"
echo "=================================================================="

# Check if conda is available, otherwise suggest virtualenv
if command -v conda &>/dev/null; then
    echo "[1/4] Using Conda to create environment '${ENV_NAME}'..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  - Environment '${ENV_NAME}' already exists. Activating..."
    else
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "[1/4] Conda not found in PATH. Creating Python virtual environment in .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo "[2/4] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch>=2.1.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "[4/4] Installing OmniVoice TTS engine..."
if [ ! -d "OmniVoice" ]; then
    echo "  - Cloning OmniVoice from https://github.com/k2-fsa/OmniVoice..."
    git clone https://github.com/k2-fsa/OmniVoice.git
fi
pip install -e ./OmniVoice

echo "=================================================================="
echo "  Setup Complete! Activate your environment with:"
echo "    conda activate ${ENV_NAME}  (or: source .venv/bin/activate)"
echo "=================================================================="

