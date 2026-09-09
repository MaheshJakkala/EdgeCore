#!/bin/bash
# Master benchmark runner for Upgrade 2
# Runs all three benchmarks and generates comparison

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${SCRIPT_DIR}/common"
PYTORCH_DIR="${SCRIPT_DIR}/pytorch"
LLAMACPP_DIR="${SCRIPT_DIR}/llamacpp"
EDGECORE_DIR="${SCRIPT_DIR}/edgecore"
RESULTS_DIR="${SCRIPT_DIR}/results/upgrade2"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "UPGRADE 2 BENCHMARK SUITE"
echo "=========================================="
echo "Model: Qwen2.5-0.5B-Instruct"
echo "Hardware: Intel i3-1115G4"
echo "Precision: FP16"
echo "Batch: 1, Context: 2048, Generation: 32 tokens"
echo "Threads: 1, 2, 4"
echo "Repetitions: 10"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Python for PyTorch
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if transformers is available
if ! python3 -c "import transformers, torch" 2>/dev/null; then
    echo "Warning: PyTorch/transformers not available. PyTorch benchmark will be skipped."
    SKIP_PYTORCH=1
else
    SKIP_PYTORCH=0
fi

# Check llama.cpp
LLAMA_CLI="${SCRIPT_DIR}/../../build/llama.cpp/llama-cli"
if [[ ! -f "${LLAMA_CLI}" ]]; then
    echo "Warning: llama.cpp not built at ${LLAMA_CLI}. llama.cpp benchmark will be skipped."
    SKIP_LLAMACPP=1
else
    SKIP_LLAMACPP=0
fi

# Check EdgeCore
EDGECORE_RUNTIME="${SCRIPT_DIR}/../../build/edgecore-runtime"
if [[ ! -f "${EDGECORE_RUNTIME}" ]]; then
    echo "Warning: EdgeCore not built at ${EDGECORE_RUNTIME}. EdgeCore benchmark will be skipped."
    SKIP_EDGECORE=1
else
    SKIP_EDGECORE=0
fi

# Check models
PYTORCH_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LLAMACPP_MODEL="${SCRIPT_DIR}/../../models/baselines/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-f16.gguf"
EDGECORE_MODEL="${SCRIPT_DIR}/../../models/qwen2-fp16.ecm"

if [[ ! -f "${LLAMACPP_MODEL}" ]]; then
    echo "Warning: llama.cpp model not found at ${LLAMACPP_MODEL}. llama.cpp benchmark will be skipped."
    SKIP_LLAMACPP=1
fi

if [[ ! -f "${EDGECORE_MODEL}" ]]; then
    echo "Warning: EdgeCore model not found at ${EDGECORE_MODEL}. EdgeCore benchmark will be skipped."
    SKIP_EDGECORE=1
fi

echo ""

# Run PyTorch benchmark
if [[ ${SKIP_PYTORCH} -eq 0 ]]; then
    echo "=========================================="
    echo "Running PyTorch FP16 baseline..."
    echo "=========================================="
    cd "${PYTORCH_DIR}"
    python3 run_pytorch.py
    echo ""
else
    echo "Skipping PyTorch benchmark (prerequisites not met)"
    echo ""
fi

# Run llama.cpp benchmark
if [[ ${SKIP_LLAMACPP} -eq 0 ]]; then
    echo "=========================================="
    echo "Running llama.cpp FP16 baseline..."
    echo "=========================================="
    cd "${LLAMACPP_DIR}"
    bash run_llamacpp.sh
    echo ""
else
    echo "Skipping llama.cpp benchmark (prerequisites not met)"
    echo ""
fi

# Run EdgeCore benchmark
if [[ ${SKIP_EDGECORE} -eq 0 ]]; then
    echo "=========================================="
    echo "Running EdgeCore FP16 baseline..."
    echo "=========================================="
    cd "${EDGECORE_DIR}"
    bash run_edgecore.sh
    echo ""
else
    echo "Skipping EdgeCore benchmark (prerequisites not met)"
    echo ""
fi

# Generate comparison table
echo "=========================================="
echo "Generating comparison table..."
echo "=========================================="
cd "${COMMON_DIR}"
python3 compare_results.py

echo ""
echo "=========================================="
echo "UPGRADE 2 BENCHMARK COMPLETE"
echo "=========================================="
echo "Results in: ${RESULTS_DIR}"