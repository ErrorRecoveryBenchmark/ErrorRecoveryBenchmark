#!/bin/bash
# Train a BC-RNN baseline for one task on one GPU with the configs used for Table 2
# (error_benchmark/configs/bc_rnn/bc_rnn_<task>_<arm>.json: 50 epochs x 500 steps, batch 16).
# 用论文 Table 2 的配置训练单个任务的 BC-RNN（50 epoch × 500 步，batch 16）。
#
# Usage:
#   bash train_bc_rnn_baseline.sh <task> <gpu_id> [nominal|recovery] [dataset.hdf5]
#   e.g.  bash train_bc_rnn_baseline.sh stack 0 recovery datasets/bc_rnn/stack_recovery.hdf5
#
# nominal: MimicGen nominal demonstrations; recovery: the same data plus the task's
# trajectories from recovery_training_success_from_error.tar (EVALUATION.md, "Training").

set -euo pipefail

TASK="${1:?usage: $0 <task> <gpu_id> [nominal|recovery] [dataset.hdf5]}"
GPU_ID="${2:?usage: $0 <task> <gpu_id> [nominal|recovery] [dataset.hdf5]}"
ARM="${3:-nominal}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONFIG_DIR="${BC_RNN_CONFIG_DIR:-$PROJECT_DIR/error_benchmark/configs/bc_rnn}"
LOG_DIR="$PROJECT_DIR/error_benchmark/outputs/logs/bc_rnn_baseline"

mkdir -p "$LOG_DIR"

CONFIG="$CONFIG_DIR/bc_rnn_${TASK}_${ARM}.json"
[ -f "$CONFIG" ] || { echo "config missing: $CONFIG" >&2; exit 1; }
DATASET="${4:-$PROJECT_DIR/datasets/bc_rnn/${TASK}_${ARM}.hdf5}"
[ -f "$DATASET" ] || { echo "dataset missing: $DATASET" >&2; exit 1; }

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate mimicgen_env

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TASK}_${ARM}_${TS}.log"

echo "[train] task=$TASK arm=$ARM gpu=$GPU_ID config=$CONFIG dataset=$DATASET"
echo "[train] log=$LOG_FILE"

cd "$PROJECT_DIR"
python -m robomimic.scripts.train --config "$CONFIG" --dataset "$DATASET" 2>&1 | tee "$LOG_FILE"
