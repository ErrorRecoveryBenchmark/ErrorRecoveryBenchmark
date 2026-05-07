#!/usr/bin/env bash
set -Eeuo pipefail

# Unified validation launcher for the benchmark matrix.
#
# This script only runs validation. It assumes training scripts/checkpoints are
# produced elsewhere and discovers checkpoints from the conventions below.
#
# Modes:
#   status       Print checkpoint resolution for selected tasks.
#   pi05-clean   Run MimicGen rollout SR for PI0.5.
#   pi05-error   Run error scene validation SR for PI0.5.
#   bcrnn        Run BCRNN clean + error validation.
#   all          Run pi05-clean, pi05-error, then bcrnn.
#   summary      Aggregate existing validation outputs into a markdown table.
#
# Examples:
#   bash error_benchmark/scripts/training/run_validation_matrix.sh status
#   VERSIONS=v1 TASKS="coffee threading pick_place" PI05_GPUS="5 6 7" \
#     bash error_benchmark/scripts/training/run_validation_matrix.sh pi05-clean
#   VERSIONS="v1 v2" MODEL_FAMILIES="pi05 bcrnn" \
#     bash error_benchmark/scripts/training/run_validation_matrix.sh summary

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_DIR="${CONDA_DIR:-${CONDA_BASE}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$CONDA_DIR/envs/openpi05/bin/python}"
MIMICGEN_PYTHON="${MIMICGEN_PYTHON:-$CONDA_DIR/envs/mimicgen_env/bin/python}"

PI05_SERVE_POLICY="${PI05_SERVE_POLICY:-$PROJECT_DIR/error_benchmark/scripts/training/start_openpi_serve_policy_safe.py}"
PI05_MIMICGEN_EVAL="${PI05_MIMICGEN_EVAL:-$PROJECT_DIR/Phoenix/evaluation/evaluate_mimicgen.py}"
PI05_ERROR_EVAL="${PI05_ERROR_EVAL:-$PROJECT_DIR/error_benchmark/scripts/training/eval_pi05_error_scenes_multi.py}"
BCRNN_EVAL="${BCRNN_EVAL:-$PROJECT_DIR/error_benchmark/scripts/training/eval_bc_rnn_error_scenes.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$PROJECT_DIR/error_benchmark/scripts/training/summarize_validation_matrix.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/error_benchmark/outputs/validation_matrix}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/error_benchmark/outputs/logs/validation_matrix}"
SCENES_ROOT="${SCENES_ROOT:-$PROJECT_DIR/error_benchmark/outputs/v5}"

DEFAULT_TASKS=(pick_place coffee stack stack_three threading three_piece_assembly)
if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARRAY=($TASKS)
else
  TASK_ARRAY=("${DEFAULT_TASKS[@]}")
fi

if [[ -n "${VERSIONS:-}" ]]; then
  # shellcheck disable=SC2206
  VERSION_ARRAY=($VERSIONS)
else
  VERSION_ARRAY=(v1)
fi

if [[ -n "${PI05_GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  PI05_GPU_ARRAY=($PI05_GPUS)
else
  PI05_GPU_ARRAY=(0 1 2 3 4 5)
fi

if [[ -n "${BCRNN_GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  BCRNN_GPU_ARRAY=($BCRNN_GPUS)
else
  BCRNN_GPU_ARRAY=(0 1 2 3 4 5)
fi

MODE="${1:-status}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Default PI0.5 checkpoint target is "20k". Exact 20000 is preferred; if a
# run uses 0-based final naming, 19999 is accepted only for target 20000.
PI05_STEP_TARGET="${PI05_STEP_TARGET:-20000}"
PI05_V2_EXP_NAME="${PI05_V2_EXP_NAME:-merged_error_from_base_v2}"
PI05_ALLOW_NEAREST="${PI05_ALLOW_NEAREST:-0}"

NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
PI05_CLEAN_BASE_PORT="${PI05_CLEAN_BASE_PORT:-8000}"
PI05_ERROR_BASE_PORT="${PI05_ERROR_BASE_PORT:-5560}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"

SCENES_PER_GROUP="${SCENES_PER_GROUP:-10}"
GROUP_BY="${GROUP_BY:-subtype_id}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-$NUM_WORKERS}"
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-20}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-600}"
WORKER_MODE="${WORKER_MODE:-round_robin}"
RESUME="${RESUME:-1}"
SCENES_SEED="${SCENES_SEED:-0}"

BCRNN_CHECKPOINT_DIR="${BCRNN_CHECKPOINT_DIR:-${BENCHMARK_ROOT}/checkpoints}"
BCRNN_V1_EXP_CANDIDATES="${BCRNN_V1_EXP_CANDIDATES:-bc_rnn_{task}_mixed_2000_100ep bc_rnn_{task}_mixed_2000}"
BCRNN_V2_EXP_CANDIDATES="${BCRNN_V2_EXP_CANDIDATES:-bc_rnn_{task}_error_v2 bc_rnn_{task}_error_ft_v2 bc_rnn_{task}_recovery_v2 bc_rnn_{task}_recovery_ft_v2}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() { echo "[validation-matrix] $*"; }
die() { echo "[validation-matrix] ERROR: $*" >&2; exit 1; }

upper_key() {
  printf "%s" "$1" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_'
}

env_value() {
  local name="$1"
  printf "%s" "${!name:-}"
}

validate_task() {
  local task="$1"
  local known
  for known in "${DEFAULT_TASKS[@]}"; do
    [[ "$task" == "$known" ]] && return 0
  done
  die "unknown task '$task'. Available: ${DEFAULT_TASKS[*]}"
}

validate_version() {
  case "$1" in
    v1|v2) ;;
    *) die "unknown version '$1'. Use v1 or v2" ;;
  esac
}

check_common_paths() {
  [[ -d "$PROJECT_DIR" ]] || die "project dir not found: $PROJECT_DIR"
  [[ -d "$OPENPI_DIR" ]] || die "openpi dir not found: $OPENPI_DIR"
  [[ -x "$OPENPI_PYTHON" ]] || die "openpi python not executable: $OPENPI_PYTHON"
  [[ -x "$MIMICGEN_PYTHON" ]] || die "mimicgen python not executable: $MIMICGEN_PYTHON"
  [[ -f "$PI05_SERVE_POLICY" ]] || die "PI0.5 serve wrapper not found: $PI05_SERVE_POLICY"
  [[ -f "$PI05_MIMICGEN_EVAL" ]] || die "MimicGen eval script not found: $PI05_MIMICGEN_EVAL"
  [[ -f "$PI05_ERROR_EVAL" ]] || die "PI0.5 error eval script not found: $PI05_ERROR_EVAL"
  [[ -f "$BCRNN_EVAL" ]] || die "BCRNN eval script not found: $BCRNN_EVAL"
  [[ -d "$SCENES_ROOT" ]] || die "scenes root not found: $SCENES_ROOT"
  local task version
  for task in "${TASK_ARRAY[@]}"; do validate_task "$task"; done
  for version in "${VERSION_ARRAY[@]}"; do validate_version "$version"; done
}

pi05_finetune_config() {
  echo "pi05_benchmark_${1}_merged_finetune"
}

pi05_inference_config() {
  echo "pi05_benchmark_${1}_merged_inference"
}

pi05_eval_name() {
  case "$1" in
    pick_place) echo "PickPlace_D0" ;;
    coffee) echo "Coffee_D0" ;;
    stack) echo "Stack_D0" ;;
    stack_three) echo "StackThree_D0" ;;
    threading) echo "Threading_D0" ;;
    three_piece_assembly) echo "ThreePieceAssembly_D0" ;;
    *) return 1 ;;
  esac
}

pi05_exp_name() {
  local task="$1"
  local version="$2"
  local task_key version_key override
  task_key="$(upper_key "$task")"
  version_key="$(upper_key "$version")"

  override="$(env_value "PI05_${version_key}_EXP_${task_key}")"
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  override="$(env_value "PI05_${version_key}_EXP")"
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi

  if [[ "$version" == "v2" ]]; then
    echo "$PI05_V2_EXP_NAME"
    return
  fi

  case "$task" in
    pick_place|coffee|threading) echo "merged_lr_original" ;;
    *) echo "merged" ;;
  esac
}

pi05_step_target() {
  local task="$1"
  local version="$2"
  local task_key version_key override
  task_key="$(upper_key "$task")"
  version_key="$(upper_key "$version")"

  override="$(env_value "PI05_${version_key}_STEP_${task_key}")"
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  override="$(env_value "PI05_${version_key}_STEP")"
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  echo "$PI05_STEP_TARGET"
}

resolve_step_dir() {
  local root="$1"
  local target="$2"
  [[ -d "$root" ]] || return 1

  if [[ -n "$target" ]]; then
    if [[ -d "$root/$target" ]]; then
      echo "$target"
      return 0
    fi
    if [[ "$target" == "20000" && -d "$root/19999" ]]; then
      echo "19999"
      return 0
    fi
    if [[ "$PI05_ALLOW_NEAREST" == "1" ]]; then
      find "$root" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
        | grep -E '^[0-9]+$' \
        | awk -v target="$target" '$1 <= target' \
        | sort -n \
        | tail -1
      return 0
    fi
    return 1
  fi

  find "$root" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

pi05_checkpoint_dir() {
  local task="$1"
  local version="$2"
  local exp root step target
  exp="$(pi05_exp_name "$task" "$version")"
  root="$OPENPI_DIR/checkpoints/$(pi05_finetune_config "$task")/$exp"
  target="$(pi05_step_target "$task" "$version")"
  step="$(resolve_step_dir "$root" "$target" || true)"
  [[ -n "$step" ]] || return 1
  echo "$root/$step"
}

latest_bcrnn_checkpoint_in_root() {
  local root="$1"
  [[ -d "$root" ]] || return 1
  find "$root" -path '*/models/model_epoch_*.pth' -type f 2>/dev/null \
    | sed -E 's/.*model_epoch_([0-9]+)\.pth$/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

bcrnn_checkpoint_path() {
  local task="$1"
  local version="$2"
  local task_key version_key override candidates pattern exp_name root ckpt
  task_key="$(upper_key "$task")"
  version_key="$(upper_key "$version")"

  override="$(env_value "BCRNN_${version_key}_CHECKPOINT_${task_key}")"
  if [[ -n "$override" ]]; then
    [[ -f "$override" ]] || return 1
    echo "$override"
    return
  fi

  if [[ "$version" == "v1" ]]; then
    candidates="$BCRNN_V1_EXP_CANDIDATES"
  else
    candidates="$BCRNN_V2_EXP_CANDIDATES"
  fi

  for pattern in $candidates; do
    exp_name="${pattern//\{task/$task}"
    exp_name="${exp_name//\}/}"
    root="$BCRNN_CHECKPOINT_DIR/$exp_name"
    ckpt="$(latest_bcrnn_checkpoint_in_root "$root" || true)"
    if [[ -n "$ckpt" ]]; then
      echo "$ckpt"
      return
    fi
  done
  return 1
}

scene_limit_for_task() {
  local task="$1"
  local scenes_dir="$SCENES_ROOT/$task/scenes"
  if [[ "$SCENES_PER_GROUP" == "0" ]]; then
    echo "0"
    return
  fi
  "$MIMICGEN_PYTHON" - "$scenes_dir" "$GROUP_BY" "$SCENES_PER_GROUP" <<'PY'
import json
import sys
from pathlib import Path

scenes_dir = Path(sys.argv[1])
group_by = sys.argv[2]
limit = int(sys.argv[3])
counts = {}
selected = 0

for path in sorted(scenes_dir.glob("*.json")):
    group_value = None
    stem = path.stem
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
        try:
            meta = json.load(open(path))
            labels = meta.get("labels", {})
            spec = meta.get("error_spec", {})
            error_name = labels.get("error_name", spec.get("error_name", "unknown"))
            degree = labels.get("degree", spec.get("degree", "unknown"))
            subtype_id = labels.get("subtype_id", f"{error_name}_{degree}")
            group_value = {
                "subtype_id": subtype_id,
                "error_name": error_name,
                "degree": degree,
            }[group_by]
        except Exception:
            group_value = "unknown"
    current = counts.get(group_value, 0)
    if current < limit:
        counts[group_value] = current + 1
        selected += 1
print(selected)
PY
}

wait_for_port() {
  local port="$1"
  local timeout="${2:-300}"
  local waited=0
  while ! "$MIMICGEN_PYTHON" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", port))
s.close()
PY
  do
    sleep 5
    waited=$((waited + 5))
    [[ "$waited" -lt "$timeout" ]] || return 1
  done
}

port_is_open() {
  local port="$1"
  "$MIMICGEN_PYTHON" -c 'import socket, sys; s = socket.socket(); s.settimeout(1); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$port" >/dev/null 2>&1
}

ensure_port_free() {
  local port="$1"
  local context="$2"
  if port_is_open "$port"; then
    die "port $port is already in use before starting $context"
  fi
}

stop_process_group() {
  local pid="$1"
  kill -- -"$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

warmup_pi05_server() {
  local port="$1"
  CUDA_VISIBLE_DEVICES="" "$MIMICGEN_PYTHON" - "$port" <<'PY'
import sys
import numpy as np
from openpi_client import websocket_client_policy as wcp

port = int(sys.argv[1])
client = wcp.WebsocketClientPolicy("localhost", port)
client.infer({
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "test",
})
PY
}

print_status() {
  check_common_paths
  log "output root: $OUTPUT_ROOT"
  log "PI0.5 default target step: $PI05_STEP_TARGET (20000 can resolve to 19999)"
  log "error validation: scenes_per_group=$SCENES_PER_GROUP group_by=$GROUP_BY"
  echo
  printf "%-24s %-8s %-8s %-20s %-8s %s\n" "task" "model" "version" "exp" "step" "checkpoint"
  local task version ckpt exp step bckpt
  for task in "${TASK_ARRAY[@]}"; do
    for version in "${VERSION_ARRAY[@]}"; do
      exp="$(pi05_exp_name "$task" "$version")"
      ckpt="$(pi05_checkpoint_dir "$task" "$version" || true)"
      step="-"
      [[ -n "$ckpt" ]] && step="$(basename "$ckpt")"
      printf "%-24s %-8s %-8s %-20s %-8s %s\n" "$task" "pi05" "$version" "$exp" "$step" "${ckpt:-missing}"
      bckpt="$(bcrnn_checkpoint_path "$task" "$version" || true)"
      printf "%-24s %-8s %-8s %-20s %-8s %s\n" "$task" "bcrnn" "$version" "-" "-" "${bckpt:-missing}"
    done
  done
}

run_pi05_clean_one() {
  local task="$1"
  local version="$2"
  local gpu="$3"
  local port="$4"
  local eval_task cfg ckpt step exp output_dir result_file meta_file server_log eval_log server_pid eval_exit
  local seed_args=()

  ckpt="$(pi05_checkpoint_dir "$task" "$version" || true)"
  [[ -n "$ckpt" ]] || die "missing PI0.5 checkpoint for task=$task version=$version"
  step="$(basename "$ckpt")"
  exp="$(pi05_exp_name "$task" "$version")"
  eval_task="$(pi05_eval_name "$task")"
  cfg="$(pi05_inference_config "$task")"
  output_dir="$OUTPUT_ROOT/pi05_${version}/mimicgen_sr"
  mkdir -p "$output_dir" "$LOG_ROOT/pi05_${version}"

  result_file="$output_dir/results_${task}_${exp}_step${step}.json"
  meta_file="$output_dir/meta_${task}_${exp}_step${step}.json"
  server_log="$LOG_ROOT/pi05_${version}/server_clean_${task}_${TIMESTAMP}.log"
  eval_log="$LOG_ROOT/pi05_${version}/eval_clean_${task}_${TIMESTAMP}.log"
  if [[ -n "${MIMICGEN_EVAL_SEED:-}" ]]; then
    seed_args=(--args.seed "$MIMICGEN_EVAL_SEED")
  fi

  log "PI0.5 clean: task=$task version=$version exp=$exp step=$step gpu=$gpu port=$port seed=${MIMICGEN_EVAL_SEED:-default}"
  ensure_port_free "$port" "PI0.5 clean $task $version"
  (
    cd "$OPENPI_DIR"
    exec setsid env CUDA_VISIBLE_DEVICES="$gpu" OPENPI_DIR="$OPENPI_DIR" "$OPENPI_PYTHON" "$PI05_SERVE_POLICY" \
      --port "$port" \
      policy:checkpoint \
      --policy.config "$cfg" \
      --policy.dir "$ckpt"
  ) > "$server_log" 2>&1 &
  server_pid=$!

  if ! wait_for_port "$port" "$SERVER_TIMEOUT"; then
    stop_process_group "$server_pid"
    die "PI0.5 server failed for $task; see $server_log"
  fi

  warmup_pi05_server "$port" >> "$server_log" 2>&1 || true

  set +e
  (
    cd "$PROJECT_DIR"
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID=0
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
    "$MIMICGEN_PYTHON" "$PI05_MIMICGEN_EVAL" \
      --args.host localhost \
      --args.port "$port" \
      --args.tasks "$eval_task" \
      --args.num-trials-per-task "$NUM_TRIALS" \
      --args.results-out-path "$result_file" \
      --args.video-out-path "$output_dir/videos/$task" \
      --args.no-save-videos \
      "${seed_args[@]}"
  ) > "$eval_log" 2>&1
  eval_exit=$?
  set -e

  stop_process_group "$server_pid"

  [[ "$eval_exit" -eq 0 ]] || die "PI0.5 clean eval failed for $task; see $eval_log"

  "$MIMICGEN_PYTHON" - "$meta_file" "$task" "$version" "$exp" "$step" "$ckpt" "$result_file" "$eval_log" <<'PY'
import json
import sys
path, task, version, exp, step, ckpt, result, log = sys.argv[1:9]
json.dump({
    "task": task,
    "model_family": "pi05",
    "version": version,
    "checkpoint_exp": exp,
    "checkpoint_step": step,
    "checkpoint": ckpt,
    "result_file": result,
    "eval_log": log,
}, open(path, "w"), indent=2)
PY
}

run_pi05_error_one() {
  local task="$1"
  local version="$2"
  local gpu="$3"
  local port="$4"
  local ckpt step exp output_dir scenes_dir resume_flag limit_args

  ckpt="$(pi05_checkpoint_dir "$task" "$version" || true)"
  [[ -n "$ckpt" ]] || die "missing PI0.5 checkpoint for task=$task version=$version"
  step="$(basename "$ckpt")"
  exp="$(pi05_exp_name "$task" "$version")"
  output_dir="$OUTPUT_ROOT/pi05_${version}/error_scenes_${SCENES_PER_GROUP}per_${GROUP_BY}"
  scenes_dir="$SCENES_ROOT/$task/scenes"
  [[ -d "$scenes_dir" ]] || die "missing scenes dir: $scenes_dir"
  mkdir -p "$output_dir" "$LOG_ROOT/pi05_${version}"

  if [[ "$RESUME" == "0" || "$RESUME" == "false" || "$RESUME" == "False" ]]; then
    resume_flag="--no_resume"
  else
    resume_flag="--resume"
  fi

  limit_args=()
  if [[ "$SCENES_PER_GROUP" != "0" ]]; then
    limit_args=(--limit_per_group "$SCENES_PER_GROUP")
  fi

  log "PI0.5 error: task=$task version=$version exp=$exp step=$step gpu=$gpu port=$port scenes_per_group=$SCENES_PER_GROUP"
  ensure_port_free "$port" "PI0.5 error $task $version"
  (
    cd "$PROJECT_DIR"
    export CONDA_DIR
    export OPENPI_DIR
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID=0
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
    "$MIMICGEN_PYTHON" "$PI05_ERROR_EVAL" \
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
      --checkpoint "$ckpt" \
      --scenes_dir "$scenes_dir" \
      --output_dir "$output_dir" \
      "${limit_args[@]}" \
      "$resume_flag"
  ) > "$LOG_ROOT/pi05_${version}/eval_error_${task}_${TIMESTAMP}.log" 2>&1
}

run_bcrnn_one() {
  local task="$1"
  local version="$2"
  local gpu="$3"
  local ckpt output_dir scenes_limit

  ckpt="$(bcrnn_checkpoint_path "$task" "$version" || true)"
  [[ -n "$ckpt" ]] || die "missing BCRNN checkpoint for task=$task version=$version"
  scenes_limit="$(scene_limit_for_task "$task")"
  output_dir="$OUTPUT_ROOT/bcrnn_${version}/combined_${SCENES_PER_GROUP}per_${GROUP_BY}"
  mkdir -p "$output_dir" "$LOG_ROOT/bcrnn_${version}"

  log "BCRNN validation: task=$task version=$version gpu=$gpu scenes_limit=$scenes_limit ckpt=$ckpt"
  (
    cd "$PROJECT_DIR"
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID=0
    "$MIMICGEN_PYTHON" "$BCRNN_EVAL" \
      --task "$task" \
      --gpu 0 \
      --num_clean "$NUM_TRIALS" \
      --max_steps "$MAX_STEPS" \
      --output_dir "$output_dir" \
      --scenes_root "$SCENES_ROOT" \
      --scenes_limit "$scenes_limit" \
      --scenes_seed "$SCENES_SEED" \
      --checkpoint "$ckpt"
  ) > "$LOG_ROOT/bcrnn_${version}/eval_${task}_${TIMESTAMP}.log" 2>&1
}

run_parallel_tasks() {
  local family="$1"
  local eval_kind="$2"
  local version="$3"
  local -n gpus_ref="$4"
  local base_port="${5:-0}"
  local pids=()
  local names=()
  local i task gpu port

  [[ "${#gpus_ref[@]}" -ge "${#TASK_ARRAY[@]}" ]] || die "need ${#TASK_ARRAY[@]} GPUs for ${TASK_ARRAY[*]}, got ${gpus_ref[*]}"

  for i in "${!TASK_ARRAY[@]}"; do
    task="${TASK_ARRAY[$i]}"
    gpu="${gpus_ref[$i]}"
    port=$((base_port + i))
    case "$family:$eval_kind" in
      pi05:clean) run_pi05_clean_one "$task" "$version" "$gpu" "$port" & ;;
      pi05:error) run_pi05_error_one "$task" "$version" "$gpu" "$port" & ;;
      bcrnn:combined) run_bcrnn_one "$task" "$version" "$gpu" & ;;
      *) die "unknown run target $family:$eval_kind" ;;
    esac
    pids+=("$!")
    names+=("$task")
    sleep 3
  done

  local failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "$family $eval_kind $version ${names[$i]} done"
    else
      log "$family $eval_kind $version ${names[$i]} failed"
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || return 1
}

run_summary() {
  check_common_paths
  "$MIMICGEN_PYTHON" "$SUMMARY_SCRIPT" \
    --output-root "$OUTPUT_ROOT" \
    --tasks "${TASK_ARRAY[@]}" \
    --versions "${VERSION_ARRAY[@]}" \
    --scenes-per-group "$SCENES_PER_GROUP" \
    --group-by "$GROUP_BY"
}

check_common_paths

case "$MODE" in
  status)
    print_status
    ;;
  pi05-clean)
    for version in "${VERSION_ARRAY[@]}"; do
      run_parallel_tasks pi05 clean "$version" PI05_GPU_ARRAY "$PI05_CLEAN_BASE_PORT"
    done
    run_summary
    ;;
  pi05-error)
    for version in "${VERSION_ARRAY[@]}"; do
      run_parallel_tasks pi05 error "$version" PI05_GPU_ARRAY "$PI05_ERROR_BASE_PORT"
    done
    run_summary
    ;;
  bcrnn)
    for version in "${VERSION_ARRAY[@]}"; do
      run_parallel_tasks bcrnn combined "$version" BCRNN_GPU_ARRAY 0
    done
    run_summary
    ;;
  all)
    for version in "${VERSION_ARRAY[@]}"; do
      run_parallel_tasks pi05 clean "$version" PI05_GPU_ARRAY "$PI05_CLEAN_BASE_PORT"
      run_parallel_tasks pi05 error "$version" PI05_GPU_ARRAY "$PI05_ERROR_BASE_PORT"
      run_parallel_tasks bcrnn combined "$version" BCRNN_GPU_ARRAY 0
    done
    run_summary
    ;;
  summary)
    run_summary
    ;;
  *)
    die "unknown mode '$MODE'. Use: status, pi05-clean, pi05-error, bcrnn, all, summary"
    ;;
esac
