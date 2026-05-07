#!/bin/bash
# Train BC-RNN baseline for one task on one GPU (20 epochs, mixed-2000 data).
#
# Usage:
#   bash train_bc_rnn_baseline.sh <task> <gpu_id>
#   e.g.  bash train_bc_rnn_baseline.sh stack 0

set -euo pipefail

TASK="${1:?usage: $0 <task> <gpu_id>}"
GPU_ID="${2:?usage: $0 <task> <gpu_id>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONDA_DIR="${CONDA_DIR:-${CONDA_BASE}}"
CONFIG_DIR="${BC_RNN_CONFIG_DIR:-${BENCHMARK_ROOT}/configs/bc_rnn}"
LOG_DIR="$PROJECT_DIR/error_benchmark/outputs/logs/bc_rnn_baseline"

mkdir -p "$LOG_DIR"

CONFIG="$CONFIG_DIR/bc_rnn_${TASK}.json"
[ -f "$CONFIG" ] || { echo "config missing: $CONFIG" >&2; exit 1; }

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate mimicgen_env

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TASK}_${TS}.log"

echo "[train] task=$TASK gpu=$GPU_ID config=$CONFIG"
echo "[train] log=$LOG_FILE"

python -m robomimic.scripts.train --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
