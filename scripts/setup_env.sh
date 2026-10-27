#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere: bash scripts/setup_env.sh
# Requirements: Linux, Conda, Git, and a compatible NVIDIA driver for GPU use.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="obtopomem"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not on PATH. Initialize Conda, then retry." >&2
    exit 1
fi

# Do not delete or recreate an existing environment.
if conda env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'; then
    echo "Using existing Conda environment: $ENV_NAME"
else
    echo "Creating Conda environment: $ENV_NAME"
    conda env create -f "$ROOT/environment.yml"
fi

run_env() {
    conda run --no-capture-output -n "$ENV_NAME" "$@"
}

# An existing environment must also use Python 3.11.
run_env python -c 'import sys; assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"'

# Restore pinned upstream submodules when cloning from GitHub.
if [[ ! -f "$ROOT/third_party/omniglue/pyproject.toml" || \
      ! -d "$ROOT/third_party/segment-anything/segment_anything" ]]; then
    echo "Initializing Git submodules..."
    git submodule update --init --recursive
fi

# TensorFlow is installed from official PyPI with NVIDIA CUDA runtime packages.
run_env python -m pip install --upgrade pip
run_env python -m pip install 'tensorflow[and-cuda]==2.20.0'

# PyTorch CUDA 12.6 wheels must come from the official PyTorch wheel index.
run_env python -m pip install \
    'torch==2.13.0' 'torchvision==0.28.0' \
    --index-url https://download.pytorch.org/whl/cu126

# Install declared ObTopoMem dependencies from pyproject.toml and install the
# local source package in editable mode. Existing framework wheels satisfy
# the pinned version requirements.
run_env python -m pip install -e "$ROOT"

# OmniGlue and SAM are pinned Git submodules. Install their source packages
# without allowing their own metadata to replace our verified framework stack.
run_env python -m pip install --no-deps \
    -e "$ROOT/third_party/omniglue" \
    -e "$ROOT/third_party/segment-anything"

run_env python -m pip check

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg is required. Install it with: sudo apt install ffmpeg" >&2
    exit 1
fi

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep "h264_nvenc" >/dev/null; then
    echo "FFmpeg must support the h264_nvenc encoder." >&2
    exit 1
fi

echo
echo "Setup complete. Activate with: conda activate $ENV_NAME"
echo "Model checkpoints are separate; place them under models/ (gitignored)."
echo "Note: the GPU requires an appropriate host NVIDIA driver."
