#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 base -> merged+error-training-data LoRA finetune.
#
# Usage:
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh status
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh convert  # combine existing LeRobot repos
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh norms
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh smoke
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh train
#   bash error_benchmark/scripts/training/launch_pi05_error_training_finetune.sh all
#
# Optional:
#   PI05_GPUS="0 1 2 3 5 6" bash ... all
#   PI05_TASKS="pick_place stack" bash ... all
#   PI05_EXP_NAME=merged_error_from_base_v2 bash ... train
#   PI05_RUN_MODE=--resume bash ... train

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_EXE="${CONDA_EXE:-${CONDA_BASE}/bin/conda}"
MIMICGEN_PYTHON="${MIMICGEN_PYTHON:-${CONDA_BASE}/envs/mimicgen_env/bin/python}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${CONDA_BASE}/envs/openpi05/bin/python}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_CACHE}/lerobot}"
PI05_BASE_CKPT="${PI05_BASE_CKPT:-WARNING_EXTERNAL_zhaoganlong/openpi_cache/openpi-assets/checkpoints/pi05_base/params}"

LOG_DIR="${PI05_LOG_DIR:-$PROJECT_DIR/error_benchmark/outputs/logs/pi05_merged_error_from_base_v2}"
REPORT_ROOT="${PI05_REPORT_ROOT:-$PROJECT_DIR/error_benchmark/outputs/pi05_merged_error_training_lerobot_reports}"
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

MERGED_DATASET_SUFFIX="${PI05_MERGED_DATASET_SUFFIX:-merged}"
ERROR_DATASET_SUFFIX="${PI05_ERROR_DATASET_SUFFIX:-error_training}"
DATASET_SUFFIX="${PI05_DATASET_SUFFIX:-merged_error_training}"
EXP_NAME="${PI05_EXP_NAME:-merged_error_from_base_v2}"
NUM_TRAIN_STEPS="${PI05_NUM_TRAIN_STEPS:-20000}"
TARGET_STEP="${PI05_TARGET_STEP:-$((NUM_TRAIN_STEPS - 1))}"
SAVE_INTERVAL="${PI05_SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${PI05_KEEP_PERIOD:-1000}"
FSDP_DEVICES="${PI05_FSDP_DEVICES:-1}"
RUN_MODE="${PI05_RUN_MODE:---overwrite}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_NORMS="${FORCE_NORMS:-0}"
CONVERT_OVERWRITE="${PI05_CONVERT_OVERWRITE:-0}"
CONVERT_WORKERS="${PI05_CONVERT_WORKERS:-24}"
CONVERT_MIN_ACTION_LEN="${PI05_CONVERT_MIN_ACTION_LEN:-30}"
XLA_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
if [[ "${PI05_WANDB_FLAG+x}" == "x" ]]; then
  WANDB_FLAG="$PI05_WANDB_FLAG"
else
  WANDB_FLAG="--wandb-enabled"
fi
ASSETS_BASE_DIR="${PI05_ASSETS_BASE_DIR:-$OPENPI_DIR/assets_merged_error_from_base_v2}"
MAX_NORM_FRAMES="${PI05_MAX_NORM_FRAMES:-}"
NORM_NUM_WORKERS="${PI05_NORM_NUM_WORKERS:-2}"
REQUIRE_SMOKE="${PI05_REQUIRE_SMOKE:-1}"
SMOKE_EXP_NAME="${PI05_SMOKE_EXP_NAME:-merged_error_from_base_v2_smoke_${TIMESTAMP}}"

# Match zhaoganlong/Motion-based-Self-Reflection-Framework/deps/openpi
# add_finetune_config_lr(..._lr_v2): Pi0.5 LoRA, batch 64, 5e-5 cosine
# decay to 5e-6, save/keep every 1000, EMA off in the base config.
BATCH_SIZE="${PI05_BATCH_SIZE:-64}"
LR_WARMUP_STEPS="${PI05_LR_WARMUP_STEPS:-1000}"
LR_PEAK="${PI05_LR_PEAK:-5e-5}"
LR_DECAY_STEPS="${PI05_LR_DECAY_STEPS:-35000}"
LR_DECAY_LR="${PI05_LR_DECAY_LR:-5e-6}"
OPT_ACCUM_STEPS="${PI05_OPT_ACCUM_STEPS:-1}"

mkdir -p "$LOG_DIR" "$REPORT_ROOT"

log() {
  echo "[pi05-error-training] $*"
}

die() {
  echo "[pi05-error-training] ERROR: $*" >&2
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

merged_repo_id() {
  local task="$1"
  echo "benchmark/mimicgen_${task}_${MERGED_DATASET_SUFFIX}"
}

source_error_repo_id() {
  local task="$1"
  echo "benchmark/mimicgen_${task}_${ERROR_DATASET_SUFFIX}"
}

training_repo_id() {
  local task="$1"
  echo "benchmark/mimicgen_${task}_${DATASET_SUFFIX}"
}

# Kept as a compatibility alias for reports/logs produced by earlier versions
# of this launcher.
error_training_repo_id() {
  training_repo_id "$1"
}

merged_dataset_dir() {
  local task="$1"
  echo "$HF_LEROBOT_HOME/$(merged_repo_id "$task")"
}

source_error_dataset_dir() {
  local task="$1"
  echo "$HF_LEROBOT_HOME/$(source_error_repo_id "$task")"
}

dataset_dir() {
  local task="$1"
  echo "$HF_LEROBOT_HOME/$(training_repo_id "$task")"
}

validation_report() {
  local task="$1"
  echo "$REPORT_ROOT/${task}_validation_report.json"
}

smoke_report() {
  local task="$1"
  echo "$REPORT_ROOT/${task}_openpi_smoke.json"
}

json_status() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing"
    return 0
  fi
  "$MIMICGEN_PYTHON" - "$path" <<'PY'
import json
import sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get("status", "unknown"))
except Exception as exc:
    print(f"bad_json:{type(exc).__name__}")
PY
}

norm_stats_dir() {
  local task="$1"
  echo "$ASSETS_BASE_DIR/$(config_name "$task")/$(error_training_repo_id "$task")"
}

norm_stats_file() {
  local task="$1"
  echo "$(norm_stats_dir "$task")/norm_stats.json"
}

target_checkpoint_dir() {
  local task="$1"
  echo "$OPENPI_DIR/checkpoints/$(config_name "$task")/$EXP_NAME"
}

init_params_path() {
  local task="$1"
  validate_task "$task"
  echo "$PI05_BASE_CKPT"
}

latest_step() {
  local ckpt_dir="$1"
  [[ -d "$ckpt_dir" ]] || return 0
  find "$ckpt_dir" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}


running_train_task() {
  local task="$1"
  pgrep -af "python scripts/train.py $(config_name "$task").*--exp-name=$EXP_NAME" >/dev/null 2>&1
}

check_common_paths() {
  [[ -d "$PROJECT_DIR" ]] || die "project dir not found: $PROJECT_DIR"
  [[ -d "$OPENPI_DIR" ]] || die "openpi dir not found: $OPENPI_DIR"
  [[ -x "$CONDA_EXE" ]] || die "conda not executable: $CONDA_EXE"
  [[ -x "$MIMICGEN_PYTHON" ]] || die "mimicgen python not executable: $MIMICGEN_PYTHON"
  [[ -x "$OPENPI_PYTHON" ]] || die "openpi python not executable: $OPENPI_PYTHON"
  [[ -f "$OPENPI_DIR/scripts/train.py" ]] || die "train script not found: $OPENPI_DIR/scripts/train.py"
  [[ -f "$OPENPI_DIR/scripts/compute_norm_stats.py" ]] || die "norm script not found: $OPENPI_DIR/scripts/compute_norm_stats.py"
  [[ -f "$PROJECT_DIR/error_benchmark/scripts/conversion/4f_combine_pi05_merged_error_lerobot.py" ]] \
    || die "combined LeRobot converter not found"
  [[ -d "$PI05_BASE_CKPT" || -f "$PI05_BASE_CKPT" ]] || die "Pi0.5 base checkpoint params not found: $PI05_BASE_CKPT"

  [[ -f "$PROJECT_DIR/error_benchmark/scripts/training/compute_pi05_error_training_norms.py" ]] \
    || die "norm helper not found"

  case "$EXP_NAME" in
    merged|merged_lr_original|recovery_from_clean20k_v2_10k|error_training_from_clean20k_v2_10k)
      [[ "${ALLOW_PROTECTED_EXP:-0}" == "1" ]] || die "refusing protected exp_name '$EXP_NAME'; choose a new PI05_EXP_NAME"
      ;;
  esac

  case "$RUN_MODE" in
    --overwrite|--resume) ;;
    *) die "PI05_RUN_MODE must be --overwrite or --resume, got '$RUN_MODE'" ;;
  esac

  for task in "${TASKS[@]}"; do
    validate_task "$task"
  done
}

status_for_task() {
  local task="$1"
  local gpu="${2:-}"
  local merged_status="missing"
  local error_status="missing"
  local data_status="missing"
  local val_status
  local norm_status="missing"
  local init_status="missing"
  local latest="-"
  local target_dir

  [[ -d "$(merged_dataset_dir "$task")" ]] && merged_status="ok"
  [[ -d "$(source_error_dataset_dir "$task")" ]] && error_status="ok"
  [[ -d "$(dataset_dir "$task")" ]] && data_status="ok"
  val_status="$(json_status "$(validation_report "$task")")"
  [[ -f "$(norm_stats_file "$task")" ]] && norm_status="ok"
  [[ -d "$(init_params_path "$task")" || -f "$(init_params_path "$task")" ]] && init_status="ok"

  target_dir="$(target_checkpoint_dir "$task")"
  latest="$(latest_step "$target_dir" || true)"
  [[ -n "$latest" ]] || latest="-"

  printf "%-24s %-5s %-6s %-6s %-8s %-10s %-8s %-8s %-11s %s\n" \
    "$task" "${gpu:-"-"}" "$merged_status" "$error_status" "$data_status" "$val_status" "$norm_status" "$init_status" "$latest" "$target_dir"
}

print_status() {
  check_common_paths
  log "project: $PROJECT_DIR"
  log "openpi: $OPENPI_DIR"
  log "lerobot home: $HF_LEROBOT_HOME"
  log "source repo suffixes: merged=$MERGED_DATASET_SUFFIX error=$ERROR_DATASET_SUFFIX"
  log "training repo suffix: $DATASET_SUFFIX"
  log "exp_name: $EXP_NAME"
  log "base checkpoint: $PI05_BASE_CKPT"
  log "num_train_steps: $NUM_TRAIN_STEPS target checkpoint step: $TARGET_STEP"
  log "zhaoganlong lr_v2 defaults: batch=$BATCH_SIZE warmup=$LR_WARMUP_STEPS peak_lr=$LR_PEAK decay_steps=$LR_DECAY_STEPS decay_lr=$LR_DECAY_LR accumulation=$OPT_ACCUM_STEPS"
  log "run mode: $RUN_MODE require_smoke=$REQUIRE_SMOKE"
  log "assets base: $ASSETS_BASE_DIR"
  log "reports: $REPORT_ROOT"
  echo
  printf "%-24s %-5s %-6s %-6s %-8s %-10s %-8s %-8s %-11s %s\n" \
    "task" "gpu" "merged" "error" "dataset" "validate" "norms" "init" "latest" "checkpoint_dir"
  for i in "${!TASKS[@]}"; do
    status_for_task "${TASKS[$i]}" "${GPUS[$i]:-}"
  done
}

check_dataset_and_validation() {
  local missing=0
  for task in "${TASKS[@]}"; do
    if [[ ! -d "$(dataset_dir "$task")" ]]; then
      log "missing combined dataset for $task: $(dataset_dir "$task")"
      missing=1
    fi
    if [[ "$(json_status "$(validation_report "$task")")" != "PASS" ]]; then
      log "validation report is not PASS for $task: $(validation_report "$task")"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "dataset validation gate failed"
}

check_norms_and_init() {
  local missing=0
  for task in "${TASKS[@]}"; do
    if [[ ! -f "$(norm_stats_file "$task")" ]]; then
      log "missing norm stats for $task: $(norm_stats_file "$task")"
      missing=1
    fi
    if [[ ! -d "$(init_params_path "$task")" && ! -f "$(init_params_path "$task")" ]]; then
      log "missing init params for $task: $(init_params_path "$task")"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "norm/init gate failed"
}

check_smoke_gate() {
  [[ "$REQUIRE_SMOKE" == "1" ]] || return 0
  local missing=0
  for task in "${TASKS[@]}"; do
    if [[ "$(json_status "$(smoke_report "$task")")" != "PASS" ]]; then
      log "OpenPI smoke report is not PASS for $task: $(smoke_report "$task")"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "OpenPI smoke gate failed; run mode 'smoke' or set PI05_REQUIRE_SMOKE=0"
}

check_train_inputs() {
  check_dataset_and_validation
  check_norms_and_init
  check_smoke_gate
}

convert_one() {
  local task="$1"
  local gpu="$2"
  local log_file="$LOG_DIR/combine_${task}_${TIMESTAMP}.log"
  local overwrite_flag=()

  [[ -d "$(merged_dataset_dir "$task")" ]] || die "merged source dataset missing for $task: $(merged_dataset_dir "$task")"
  [[ -d "$(source_error_dataset_dir "$task")" ]] || die "error-training source dataset missing for $task: $(source_error_dataset_dir "$task")"
  if [[ -d "$(dataset_dir "$task")" && "$(json_status "$(validation_report "$task")")" == "PASS" && "$FORCE_CONVERT" != "1" ]]; then
    log "skip combine $task: combined dataset exists and validation report PASS"
    return 0
  fi
  if [[ -d "$(dataset_dir "$task")" ]]; then
    [[ "$CONVERT_OVERWRITE" == "1" || "$FORCE_CONVERT" == "1" ]] \
      || die "combined dataset exists for $task; set PI05_CONVERT_OVERWRITE=1 or FORCE_CONVERT=1"
    overwrite_flag=(--overwrite)
  fi

  log "combining $task on slot $gpu: $(merged_repo_id "$task") + $(source_error_repo_id "$task") -> $(training_repo_id "$task")"
  (
    cd "$PROJECT_DIR"
    export HF_LEROBOT_HOME
    "$OPENPI_PYTHON" error_benchmark/scripts/conversion/4f_combine_pi05_merged_error_lerobot.py \
      --task "$task" \
      --merged-repo-suffix "$MERGED_DATASET_SUFFIX" \
      --error-repo-suffix "$ERROR_DATASET_SUFFIX" \
      --output-repo-suffix "$DATASET_SUFFIX" \
      --report-root "$REPORT_ROOT" \
      "${overwrite_flag[@]}"
  ) 2>&1 | tee "$log_file"
}

convert_all() {
  check_common_paths

  log "launching ${#TASKS[@]} combined-dataset jobs"
  pids=()
  names=()
  for i in "${!TASKS[@]}"; do
    convert_one "${TASKS[$i]}" "${GPUS[$i]:-$i}" &
    pids+=("$!")
    names+=("${TASKS[$i]}")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "combine ${names[$i]} finished"
    else
      log "combine ${names[$i]} failed; see $LOG_DIR/combine_${names[$i]}_${TIMESTAMP}.log"
      failed=1
    fi
  done
  print_status
  [[ "$failed" -eq 0 ]] || die "one or more combined-dataset jobs failed"
}

compute_norms_one() {
  local task="$1"
  local cfg repo log_file
  cfg="$(config_name "$task")"
  repo="$(error_training_repo_id "$task")"
  log_file="$LOG_DIR/norms_${task}_${TIMESTAMP}.log"

  [[ -d "$(dataset_dir "$task")" ]] || die "dataset missing for $task: $(dataset_dir "$task")"
  [[ "$(json_status "$(validation_report "$task")")" == "PASS" ]] || die "validation report is not PASS for $task"

  if [[ -f "$(norm_stats_file "$task")" && "$FORCE_NORMS" != "1" ]]; then
    log "skip norm stats for $task: $(norm_stats_file "$task") exists"
    return 0
  fi

  log "computing norm stats for $task repo=$repo"
  (
    cd "$OPENPI_DIR"
    export HF_LEROBOT_HOME
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    "$OPENPI_PYTHON" "$PROJECT_DIR/error_benchmark/scripts/training/compute_pi05_error_training_norms.py" \
      --config-name "$cfg" \
      --repo-id "$repo" \
      --assets-base-dir "$ASSETS_BASE_DIR" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NORM_NUM_WORKERS" \
      ${MAX_NORM_FRAMES:+--max-frames "$MAX_NORM_FRAMES"}
  ) 2>&1 | tee "$log_file"
}

compute_norms() {
  check_common_paths
  check_dataset_and_validation
  for task in "${TASKS[@]}"; do
    compute_norms_one "$task"
  done
  print_status
}

select_train_tasks() {
  TRAIN_TASKS=()
  for task in "${TASKS[@]}"; do
    local step
    if running_train_task "$task"; then
      log "skip $task: train process already running for exp $EXP_NAME"
      continue
    fi
    step="$(latest_step "$(target_checkpoint_dir "$task")" || true)"
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

run_train_jobs() {
  local mode_label="$1"
  local exp_name="$2"
  local run_mode="$3"
  local num_steps="$4"
  local save_interval="$5"
  local keep_period="$6"
  shift 6
  local run_tasks=("$@")

  [[ "${#run_tasks[@]}" -gt 0 ]] || return 0
  [[ "${#GPUS[@]}" -ge "${#run_tasks[@]}" ]] || die "need ${#run_tasks[@]} GPUs, got ${#GPUS[@]}: ${GPUS[*]}"

  log "launching ${#run_tasks[@]} Pi0.5 $mode_label jobs"
  log "tasks: ${run_tasks[*]}"
  log "gpus: ${GPUS[*]}"
  log "logs: $LOG_DIR"

  pids=()
  names=()
  cd "$OPENPI_DIR"
  for i in "${!run_tasks[@]}"; do
    local task gpu cfg repo init_params task_log
    task="${run_tasks[$i]}"
    gpu="${GPUS[$i]}"
    cfg="$(config_name "$task")"
    repo="$(error_training_repo_id "$task")"
    init_params="$(init_params_path "$task")"
    task_log="$LOG_DIR/${mode_label}_${task}_${TIMESTAMP}.log"

    log "starting $mode_label $task on GPU $gpu from $init_params"
    (
      export HF_LEROBOT_HOME
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
      "$CONDA_EXE" run -n openpi05 python scripts/train.py "$cfg" \
        --exp-name="$exp_name" \
        "$run_mode" \
        --num-train-steps="$num_steps" \
        --batch-size="$BATCH_SIZE" \
        --save-interval="$save_interval" \
        --keep-period="$keep_period" \
        --fsdp-devices="$FSDP_DEVICES" \
        --lr-schedule.warmup-steps="$LR_WARMUP_STEPS" \
        --lr-schedule.peak-lr="$LR_PEAK" \
        --lr-schedule.decay-steps="$LR_DECAY_STEPS" \
        --lr-schedule.decay-lr="$LR_DECAY_LR" \
        --optimizer.accumulation-steps="$OPT_ACCUM_STEPS" \
        --assets-base-dir="$ASSETS_BASE_DIR" \
        --data.repo-id="$repo" \
        --weight-loader.params-path="$init_params" \
        $WANDB_FLAG \
        2>&1 | tee "$task_log"
    ) &
    pids+=("$!")
    names+=("$task")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "$mode_label ${names[$i]} finished"
      if [[ "$mode_label" == "smoke" ]]; then
        printf '{"status":"PASS","task":"%s","repo_id":"%s","exp_name":"%s","log":"%s","finished_at":"%s"}\n' \
          "${names[$i]}" "$(error_training_repo_id "${names[$i]}")" "$exp_name" "$LOG_DIR/${mode_label}_${names[$i]}_${TIMESTAMP}.log" "$(date +%Y-%m-%dT%H:%M:%S)" \
          > "$(smoke_report "${names[$i]}")"
      fi
    else
      log "$mode_label ${names[$i]} failed; see $LOG_DIR/${mode_label}_${names[$i]}_${TIMESTAMP}.log"
      failed=1
      if [[ "$mode_label" == "smoke" ]]; then
        printf '{"status":"FAIL","task":"%s","repo_id":"%s","exp_name":"%s","log":"%s","finished_at":"%s"}\n' \
          "${names[$i]}" "$(error_training_repo_id "${names[$i]}")" "$exp_name" "$LOG_DIR/${mode_label}_${names[$i]}_${TIMESTAMP}.log" "$(date +%Y-%m-%dT%H:%M:%S)" \
          > "$(smoke_report "${names[$i]}")"
      fi
    fi
  done

  print_status
  [[ "$failed" -eq 0 ]] || die "one or more Pi0.5 $mode_label jobs failed"
}

smoke_local() {
  check_common_paths
  check_dataset_and_validation
  check_norms_and_init
  run_train_jobs "smoke" "$SMOKE_EXP_NAME" "--overwrite" "1" "1" "1" "${TASKS[@]}"
}

train_local() {
  check_common_paths
  check_train_inputs
  select_train_tasks

  if [[ "${#TRAIN_TASKS[@]}" -eq 0 ]]; then
    log "no tasks need training"
    return 0
  fi
  run_train_jobs "train" "$EXP_NAME" "$RUN_MODE" "$NUM_TRAIN_STEPS" "$SAVE_INTERVAL" "$KEEP_PERIOD" "${TRAIN_TASKS[@]}"
}

case "$MODE" in
  status)
    print_status
    ;;
  convert|combine)
    convert_all
    ;;
  norms)
    compute_norms
    ;;
  smoke)
    smoke_local
    ;;
  train)
    train_local
    ;;
  all)
    convert_all
    compute_norms
    smoke_local
    train_local
    ;;
  *)
    die "unknown mode '$MODE'. Use: status, convert, combine, norms, smoke, train, all"
    ;;
esac
