#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 base -> normal MimicGen merged-data LoRA finetune.
#
# Usage:
#   bash error_benchmark/scripts/training/launch_pi05_normal_finetune.sh status
#   bash error_benchmark/scripts/training/launch_pi05_normal_finetune.sh prepare
#   bash error_benchmark/scripts/training/launch_pi05_normal_finetune.sh train
#   bash error_benchmark/scripts/training/launch_pi05_normal_finetune.sh all
#
# Optional:
#   PI05_GPUS="0 1 2 3 4 5" bash ... train
#   PI05_TASKS="coffee stack" bash ... train
#   FORCE_RETRAIN=1 bash ... train

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_EXE="${CONDA_EXE:-${CONDA_BASE}/bin/conda}"
PI05_BASE_CKPT="${PI05_BASE_CKPT:-WARNING_EXTERNAL_zhaoganlong/openpi_cache/openpi-assets/checkpoints/pi05_base/params}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_CACHE}/lerobot}"

PIPELINE_SCRIPT="$PROJECT_DIR/error_benchmark/scripts/training/train_pi05_merged.py"
LOG_DIR="$PROJECT_DIR/error_benchmark/outputs/logs/pi05_normal_finetune"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_TASKS=(pick_place coffee stack stack_three threading three_piece_assembly)
MODE="${1:-status}"
shift || true

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
elif [[ -n "${PI05_TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS=($PI05_TASKS)
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

if [[ -n "${PI05_GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  GPUS=($PI05_GPUS)
else
  GPUS=(0 1 2 3 4 5)
fi

# Existing normal baseline checkpoints use exp_name=merged. Keep this default so
# reruns resume the existing Pi0.5-base-finetuned normal models instead of
# creating a parallel directory.
EXP_NAME="${PI05_EXP_NAME:-merged}"
TARGET_STEP="${PI05_TARGET_STEP:-29999}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
XLA_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
WANDB_FLAG="${PI05_WANDB_FLAG:---wandb-enabled}"

mkdir -p "$LOG_DIR"

log() {
  echo "[pi05-normal] $*"
}

die() {
  echo "[pi05-normal] ERROR: $*" >&2
  exit 1
}

validate_task() {
  local task="$1"
  local ok=1
  for known in "${DEFAULT_TASKS[@]}"; do
    if [[ "$task" == "$known" ]]; then
      ok=0
      break
    fi
  done
  [[ "$ok" -eq 0 ]] || die "unknown task '$task'. Available: ${DEFAULT_TASKS[*]}"
}

config_name() {
  local task="$1"
  echo "pi05_benchmark_${task}_merged_finetune"
}

dataset_dir() {
  local task="$1"
  echo "$HF_LEROBOT_HOME/benchmark/mimicgen_${task}_merged"
}

checkpoint_dir() {
  local task="$1"
  echo "$OPENPI_DIR/checkpoints/$(config_name "$task")/$EXP_NAME"
}

latest_step() {
  local ckpt_dir="$1"
  [[ -d "$ckpt_dir" ]] || return 0
  find "$ckpt_dir" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

check_common_paths() {
  [[ -d "$PROJECT_DIR" ]] || die "project dir not found: $PROJECT_DIR"
  [[ -d "$OPENPI_DIR" ]] || die "openpi dir not found: $OPENPI_DIR"
  [[ -x "$CONDA_EXE" ]] || die "conda not executable: $CONDA_EXE"
  [[ -f "$PIPELINE_SCRIPT" ]] || die "pipeline script not found: $PIPELINE_SCRIPT"
  [[ -e "$PI05_BASE_CKPT" || -d "$PI05_BASE_CKPT" ]] || die "Pi0.5 base checkpoint not found: $PI05_BASE_CKPT"
  for task in "${TASKS[@]}"; do
    validate_task "$task"
  done
}

print_status() {
  check_common_paths
  log "project: $PROJECT_DIR"
  log "openpi: $OPENPI_DIR"
  log "base checkpoint: $PI05_BASE_CKPT"
  log "lerobot home: $HF_LEROBOT_HOME"
  log "exp_name: $EXP_NAME"
  log "target step: $TARGET_STEP"
  echo
  printf "%-24s %-9s %-12s %s\n" "task" "dataset" "latest_step" "checkpoint_dir"
  for task in "${TASKS[@]}"; do
    local data_status="missing"
    local data_dir
    local ckpt_dir
    local step
    data_dir="$(dataset_dir "$task")"
    ckpt_dir="$(checkpoint_dir "$task")"
    if [[ -d "$data_dir" ]]; then
      data_status="ok"
    fi
    step="$(latest_step "$ckpt_dir" || true)"
    if [[ -z "$step" ]]; then
      step="-"
    fi
    printf "%-24s %-9s %-12s %s\n" "$task" "$data_status" "$step" "$ckpt_dir"
  done
}

prepare_data() {
  check_common_paths
  log "converting normal merged LeRobot datasets where missing"
  "$CONDA_EXE" run -n openpi05 python "$PIPELINE_SCRIPT" convert-data --tasks "${TASKS[@]}" \
    2>&1 | tee "$LOG_DIR/prepare_convert_${TIMESTAMP}.log"

  log "computing norm stats for Pi0.5 normal finetune configs"
  "$CONDA_EXE" run -n openpi05 python "$PIPELINE_SCRIPT" compute-norms --tasks "${TASKS[@]}" \
    2>&1 | tee "$LOG_DIR/prepare_norms_${TIMESTAMP}.log"
}

select_train_tasks() {
  TRAIN_TASKS=()
  for task in "${TASKS[@]}"; do
    local data_dir
    local ckpt_dir
    local step
    data_dir="$(dataset_dir "$task")"
    ckpt_dir="$(checkpoint_dir "$task")"
    [[ -d "$data_dir" ]] || die "dataset missing for $task: $data_dir. Run mode 'prepare' first."

    step="$(latest_step "$ckpt_dir" || true)"
    if [[ "$FORCE_RETRAIN" == "1" ]]; then
      TRAIN_TASKS+=("$task")
    elif [[ -z "$step" ]]; then
      TRAIN_TASKS+=("$task")
    elif [[ "$step" -lt "$TARGET_STEP" ]]; then
      TRAIN_TASKS+=("$task")
    else
      log "skip $task: latest step $step >= target $TARGET_STEP"
    fi
  done
}

train_local() {
  check_common_paths
  select_train_tasks

  if [[ "${#TRAIN_TASKS[@]}" -eq 0 ]]; then
    log "no tasks need training"
    return 0
  fi
  [[ "${#GPUS[@]}" -ge "${#TRAIN_TASKS[@]}" ]] || die "need ${#TRAIN_TASKS[@]} GPUs, got ${#GPUS[@]}: ${GPUS[*]}"

  log "launching ${#TRAIN_TASKS[@]} Pi0.5 normal finetune jobs locally"
  log "tasks: ${TRAIN_TASKS[*]}"
  log "gpus: ${GPUS[*]}"
  log "logs: $LOG_DIR"

  pids=()
  names=()
  cd "$OPENPI_DIR"
  for i in "${!TRAIN_TASKS[@]}"; do
    task="${TRAIN_TASKS[$i]}"
    gpu="${GPUS[$i]}"
    cfg="$(config_name "$task")"
    task_log="$LOG_DIR/train_${task}_${TIMESTAMP}.log"

    if [[ "$FORCE_RETRAIN" == "1" ]]; then
      run_mode_flag="--overwrite"
      log "starting $task on GPU $gpu with overwrite"
    else
      run_mode_flag="--resume"
      log "starting/resuming $task on GPU $gpu"
    fi

    (
      export HF_LEROBOT_HOME
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
      "$CONDA_EXE" run -n openpi05 python scripts/train.py "$cfg" \
        --exp-name="$EXP_NAME" \
        "$run_mode_flag" \
        $WANDB_FLAG \
        2>&1 | tee "$task_log"
    ) &
    pids+=("$!")
    names+=("$task")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "${names[$i]} finished"
    else
      log "${names[$i]} failed; see $LOG_DIR/train_${names[$i]}_${TIMESTAMP}.log"
      failed=1
    fi
  done

  print_status
  [[ "$failed" -eq 0 ]] || die "one or more Pi0.5 finetune jobs failed"
}

case "$MODE" in
  status)
    print_status
    ;;
  prepare)
    prepare_data
    print_status
    ;;
  train)
    train_local
    ;;
  all)
    prepare_data
    train_local
    ;;
  *)
    die "unknown mode '$MODE'. Use: status, prepare, train, all"
    ;;
esac
