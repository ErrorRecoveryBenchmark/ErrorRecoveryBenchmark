#!/bin/bash
# Download the RecoverBench dataset and seed demos
#
# Usage:
#   bash scripts/download_data.sh [/path/to/data/dir]
#
# This script downloads:
#   1. Seed demos from MimicGen's HuggingFace repo (required for data generation)
#   2. RecoverBench evaluation data from HuggingFace (error scenes + recovery demos)
#
# After downloading, set:
#   export BENCHMARK_DATA=/path/to/data/dir

set -euo pipefail

DEST_DIR="${1:-./data}"
mkdir -p "$DEST_DIR"

echo "=========================================="
echo " RecoverBench Data Download"
echo " Destination: $DEST_DIR"
echo "=========================================="

# ─── 1. Download seed demos from MimicGen HuggingFace ───
SEED_DIR="$DEST_DIR/seed_demos"
mkdir -p "$SEED_DIR"

TASKS=(pick_place stack stack_three coffee threading three_piece_assembly)
HF_REPO="amandlek/mimicgen_datasets"

echo ""
echo "[1/2] Downloading seed demos from HuggingFace ($HF_REPO)..."

if command -v huggingface-cli &>/dev/null; then
    for task in "${TASKS[@]}"; do
        if [ -f "$SEED_DIR/${task}.hdf5" ]; then
            echo "  Skip: ${task}.hdf5 (already exists)"
        else
            echo "  Downloading: source/${task}.hdf5 -> $SEED_DIR/${task}.hdf5"
            huggingface-cli download "$HF_REPO" "source/${task}.hdf5" \
                --repo-type dataset \
                --local-dir "$SEED_DIR" \
                --local-dir-use-symlinks False 2>/dev/null
            # Move from nested path to flat
            if [ -f "$SEED_DIR/source/${task}.hdf5" ]; then
                mv "$SEED_DIR/source/${task}.hdf5" "$SEED_DIR/${task}.hdf5"
            fi
        fi
    done
    rmdir "$SEED_DIR/source" 2>/dev/null || true
else
    echo "  WARNING: huggingface-cli not found. Install with: pip install huggingface_hub"
    echo "  Manual download: huggingface-cli download $HF_REPO --include 'source/*.hdf5' --repo-type dataset --local-dir $SEED_DIR"
fi

# ─── 2. Download RecoverBench evaluation data ───
echo ""
echo "[2/2] Downloading RecoverBench evaluation data..."

# TODO: Replace ANONYMOUS with actual HuggingFace repo after de-anonymization
HF_RECOVERBENCH="ANONYMOUS/RecoverBench"

echo "  NOTE: Dataset repo not yet public. After release, run:"
echo "    huggingface-cli download $HF_RECOVERBENCH --repo-type dataset --local-dir $DEST_DIR"
echo ""
echo "  Or download manually from the HuggingFace dataset page."

# ─── Done ───
echo ""
echo "=========================================="
echo " Download complete."
echo ""
echo " Set environment variable:"
echo "   export BENCHMARK_DATA=$DEST_DIR"
echo ""
echo " Seed demos: $SEED_DIR/"
echo "=========================================="
