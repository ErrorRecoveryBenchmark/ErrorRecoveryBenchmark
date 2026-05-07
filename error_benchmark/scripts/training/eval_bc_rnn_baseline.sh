#!/bin/bash
# Eval one BC-RNN baseline task on one GPU.
# Usage: bash eval_bc_rnn_baseline.sh <task> <gpu_id> [<scenes_root>] [<scenes_limit>] [<num_clean>] [<max_steps>] [<scenes_seed>]

set -euo pipefail

TASK="${1:?usage: $0 <task> <gpu_id>}"
GPU_ID="${2:?usage: $0 <task> <gpu_id>}"
SCENES_ROOT="${3:-}"
SCENES_LIMIT="${4:-0}"
NUM_CLEAN="${5:-50}"
MAX_STEPS="${6:-500}"
SCENES_SEED="${7:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONDA_DIR="${CONDA_DIR:-${CONDA_BASE}}"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate mimicgen_env

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# After CUDA_VISIBLE_DEVICES remaps to a single visible GPU, EGL must use local device 0.
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"

ARGS=(--task "$TASK" --gpu 0 --num_clean "$NUM_CLEAN" --max_steps "$MAX_STEPS")
[ -n "$SCENES_ROOT" ] && ARGS+=(--scenes_root "$SCENES_ROOT")
[ "$SCENES_LIMIT" -gt 0 ] && ARGS+=(--scenes_limit "$SCENES_LIMIT")
ARGS+=(--scenes_seed "$SCENES_SEED")

echo "[eval] task=$TASK gpu=$GPU_ID scenes_limit=$SCENES_LIMIT scenes_seed=$SCENES_SEED num_clean=$NUM_CLEAN"
python "$PROJECT_DIR/error_benchmark/scripts/training/eval_bc_rnn_error_scenes.py" "${ARGS[@]}"
