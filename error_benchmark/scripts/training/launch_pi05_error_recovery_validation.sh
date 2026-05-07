#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 error recovery validation on v5 validation scenes.
# Uses eval_pi05_error_scenes.py with --skip_clean and explicit
# error_benchmark/outputs/v5/<task>/scenes, locked to PI05_STEP by default.
# Set SCENES_PER_GROUP=0 for full validation; default is 100 per GROUP_BY value.

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_DIR="${CONDA_DIR:-${CONDA_BASE}}"
MIMICGEN_PYTHON="${MIMICGEN_PYTHON:-$CONDA_DIR/envs/mimicgen_env/bin/python}"
EVAL_SCRIPT_SINGLE="$PROJECT_DIR/error_benchmark/scripts/training/eval_pi05_error_scenes.py"
EVAL_SCRIPT_MULTI="$PROJECT_DIR/error_benchmark/scripts/training/eval_pi05_error_scenes_multi.py"

PI05_STEP="${PI05_STEP:-25000}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-$NUM_WORKERS}"
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-20}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-600}"
WORKER_MODE="${WORKER_MODE:-round_robin}"
GROUP_BY="${GROUP_BY:-subtype_id}"
SCENES_PER_GROUP="${SCENES_PER_GROUP:-100}"
RESUME="${RESUME:-1}"
SCENES_ROOT="${SCENES_ROOT:-$PROJECT_DIR/error_benchmark/outputs/v5}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$SCENES_PER_GROUP" =~ ^[1-9][0-9]*$ ]]; then
    OUTPUT_DIR="$PROJECT_DIR/error_benchmark/outputs/pi05_error_recovery_validation_${SCENES_PER_GROUP}per_${GROUP_BY}"
  else
    OUTPUT_DIR="$PROJECT_DIR/error_benchmark/outputs/pi05_error_recovery_validation"
  fi
fi
if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ "$SCENES_PER_GROUP" =~ ^[1-9][0-9]*$ ]]; then
    LOG_DIR="$PROJECT_DIR/error_benchmark/outputs/logs/pi05_error_recovery_validation_${SCENES_PER_GROUP}per_${GROUP_BY}"
  else
    LOG_DIR="$PROJECT_DIR/error_benchmark/outputs/logs/pi05_error_recovery_validation"
  fi
fi
BASE_PORT="${BASE_PORT:-5560}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${1:-status}"

DEFAULT_TASKS=(pick_place coffee stack stack_three threading three_piece_assembly)
if [[ -n "${PI05_TASKS:-}" ]]; then
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

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

log() { echo "[pi05-error-val] $*"; }
die() { echo "[pi05-error-val] ERROR: $*" >&2; exit 1; }

scene_limit_enabled() {
  [[ "$SCENES_PER_GROUP" =~ ^[1-9][0-9]*$ ]]
}

validate_options() {
  [[ "$SCENES_PER_GROUP" =~ ^[0-9]+$ ]] || die "SCENES_PER_GROUP must be a non-negative integer, got '$SCENES_PER_GROUP'"
  case "$GROUP_BY" in
    subtype_id|error_name|degree) ;;
    *) die "GROUP_BY must be one of: subtype_id, error_name, degree" ;;
  esac
}

validate_task() {
  local task="$1"
  local ok=1
  for known in "${DEFAULT_TASKS[@]}"; do
    if [[ "$task" == "$known" ]]; then ok=0; break; fi
  done
  [[ "$ok" -eq 0 ]] || die "unknown task '$task'. Available: ${DEFAULT_TASKS[*]}"
}

finetune_config() { echo "pi05_benchmark_${1}_merged_finetune"; }
checkpoint_path() { echo "$OPENPI_DIR/checkpoints/$(finetune_config "$1")/merged/$PI05_STEP"; }
scenes_dir() { echo "$SCENES_ROOT/$1/scenes"; }

check_paths() {
  validate_options
  [[ -d "$PROJECT_DIR" ]] || die "project dir not found: $PROJECT_DIR"
  [[ -d "$OPENPI_DIR" ]] || die "openpi dir not found: $OPENPI_DIR"
  [[ -x "$MIMICGEN_PYTHON" ]] || die "mimicgen python not executable: $MIMICGEN_PYTHON"
  [[ -f "$EVAL_SCRIPT_SINGLE" ]] || die "eval script not found: $EVAL_SCRIPT_SINGLE"
  [[ -f "$EVAL_SCRIPT_MULTI" ]] || die "eval script not found: $EVAL_SCRIPT_MULTI"
  [[ -d "$SCENES_ROOT" ]] || die "scenes root not found: $SCENES_ROOT"
  for task in "${TASKS[@]}"; do
    validate_task "$task"
    [[ -d "$(checkpoint_path "$task")" ]] || die "checkpoint missing for $task: $(checkpoint_path "$task")"
    [[ -d "$(scenes_dir "$task")" ]] || die "scenes dir missing for $task: $(scenes_dir "$task")"
  done
}

scene_count() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.json' | wc -l
}

selected_scene_count() {
  local dir="$1"
  if ! scene_limit_enabled; then
    scene_count "$dir"
    return
  fi
  "$MIMICGEN_PYTHON" - "$dir" "$GROUP_BY" "$SCENES_PER_GROUP" <<'PY'
import sys
from pathlib import Path

scenes_dir = Path(sys.argv[1])
group_by = sys.argv[2]
limit = int(sys.argv[3])
counts = {}
selected = 0

for path in sorted(scenes_dir.glob("*.json")):
    stem = path.stem
    group_value = None
    if stem.startswith("v5_"):
        parts = stem[3:].rsplit("_", 2)
        if len(parts) == 3 and parts[1].startswith("D"):
            error_name, degree, _ = parts
            subtype_id = f"{error_name}_{degree}"
            group_value = {
                "subtype_id": subtype_id,
                "error_name": error_name,
                "degree": degree,
            }[group_by]
    if group_value is None:
        group_value = "unknown"

    current = counts.get(group_value, 0)
    if current < limit:
        counts[group_value] = current + 1
        selected += 1

print(selected)
PY
}

print_status() {
  check_paths
  log "project: $PROJECT_DIR"
  log "openpi: $OPENPI_DIR"
  log "step: $PI05_STEP"
  log "workers per task: $NUM_WORKERS"
  log "worker mode: $WORKER_MODE"
  log "group by: $GROUP_BY"
  log "scenes per group: $SCENES_PER_GROUP"
  log "resume: $RESUME"
  log "scenes root: $SCENES_ROOT"
  log "output: $OUTPUT_DIR"
  echo
  printf "%-24s %-10s %-18s %s\n" "task" "step" "selected/total" "checkpoint"
  for task in "${TASKS[@]}"; do
    local total selected
    total="$(scene_count "$(scenes_dir "$task")")"
    selected="$(selected_scene_count "$(scenes_dir "$task")")"
    printf "%-24s %-10s %-18s %s\n" "$task" "$PI05_STEP" "$selected/$total" "$(checkpoint_path "$task")"
  done
}

run_one() {
  local task="$1"
  local gpu="$2"
  local port="$3"
  local log_file="$LOG_DIR/${task}_${TIMESTAMP}.log"
  local total selected
  total="$(scene_count "$(scenes_dir "$task")")"
  selected="$(selected_scene_count "$(scenes_dir "$task")")"
  log "starting $task on GPU $gpu port $port workers=$NUM_WORKERS mode=$WORKER_MODE group_by=$GROUP_BY scenes=$selected/$total scenes_per_group=$SCENES_PER_GROUP"
  (
    cd "$PROJECT_DIR"
    export CONDA_DIR="$CONDA_DIR"
    export OPENPI_DIR="$OPENPI_DIR"
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID="$gpu"
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
    if [[ "$NUM_WORKERS" -gt 1 || "$WORKER_MODE" != "round_robin" ]] || scene_limit_enabled; then
      args=(
        "$EVAL_SCRIPT_MULTI"
        --task "$task" \
        --gpu "$gpu" \
        --port "$port" \
        --num_workers "$NUM_WORKERS" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --batch_timeout_ms "$BATCH_TIMEOUT_MS" \
        --inference_timeout "$INFERENCE_TIMEOUT" \
        --worker_mode "$WORKER_MODE" \
        --group_by "$GROUP_BY" \
        --max_steps "$MAX_STEPS" \
        --checkpoint "$(checkpoint_path "$task")" \
        --scenes_dir "$(scenes_dir "$task")" \
        --output_dir "$OUTPUT_DIR"
      )
      if scene_limit_enabled; then
        args+=(--limit_per_group "$SCENES_PER_GROUP")
      fi
      if [[ "$RESUME" == "0" || "$RESUME" == "false" || "$RESUME" == "False" || "$RESUME" == "no" ]]; then
        args+=(--no_resume)
      else
        args+=(--resume)
      fi
      "$MIMICGEN_PYTHON" "${args[@]}"
    else
      "$MIMICGEN_PYTHON" "$EVAL_SCRIPT_SINGLE" \
        --task "$task" \
        --gpu "$gpu" \
        --port "$port" \
        --skip_clean \
        --max_steps "$MAX_STEPS" \
        --checkpoint "$(checkpoint_path "$task")" \
        --scenes_dir "$(scenes_dir "$task")" \
        --output_dir "$OUTPUT_DIR"
    fi
  ) > "$log_file" 2>&1
}

run_all() {
  check_paths
  [[ "${#GPUS[@]}" -ge "${#TASKS[@]}" ]] || die "need ${#TASKS[@]} GPUs, got ${#GPUS[@]}: ${GPUS[*]}"
  log "launching ${#TASKS[@]} error recovery validation jobs"
  log "workers per task: $NUM_WORKERS"
  log "worker mode: $WORKER_MODE"
  log "group by: $GROUP_BY"
  log "scenes per group: $SCENES_PER_GROUP"
  log "resume: $RESUME"
  log "logs: $LOG_DIR"
  pids=(); names=()
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    run_one "$task" "$gpu" "$port" &
    pids+=("$!"); names+=("$task")
    sleep 3
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "${names[$i]} done"
    else
      log "${names[$i]} failed; see $LOG_DIR/${names[$i]}_${TIMESTAMP}.log"
      failed=1
    fi
  done
  print_summary
  [[ "$failed" -eq 0 ]] || die "one or more validation jobs failed"
}

print_summary() {
  echo
  printf "%-24s %-10s %-12s %-12s %s\n" "task" "step" "scenes" "RSR" "succ/total"
  for task in "${TASKS[@]}"; do
    local file="$OUTPUT_DIR/${task}.json"
    if [[ -f "$file" ]]; then
      "$MIMICGEN_PYTHON" - "$task" "$file" <<'PY'
import json, sys
task, file = sys.argv[1:3]
d = json.load(open(file))
err = d.get("error_scenes", {})
step = d.get("checkpoint_step", "?")
print(f"{task:<24} {str(step):<10} {err.get('total', 0):<12} {err.get('overall_sr', 0):<12.1%} {err.get('successes', 0)}/{err.get('total', 0)}")
PY
    else
      printf "%-24s %-10s %-12s %-12s %s\n" "$task" "$PI05_STEP" "$(scene_count "$(scenes_dir "$task")")" "-" "missing"
    fi
  done
}

case "$MODE" in
  status) print_status ;;
  all) run_all ;;
  summary) check_paths; print_summary ;;
  *) die "unknown mode '$MODE'. Use: status, all, summary" ;;
esac
