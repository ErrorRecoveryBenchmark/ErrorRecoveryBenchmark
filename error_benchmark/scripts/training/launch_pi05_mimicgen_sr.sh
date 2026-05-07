#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 Normal SR using the MimicGen evaluation protocol.
#
# This is different from eval_pi05_error_scenes.py --skip_error:
# it runs Phoenix/evaluation/evaluate_mimicgen.py directly on MimicGen
# environments and reports success_rate/successes/episodes.
#
# Usage:
#   bash error_benchmark/scripts/training/launch_pi05_mimicgen_sr.sh status
#   bash error_benchmark/scripts/training/launch_pi05_mimicgen_sr.sh smoke
#   bash error_benchmark/scripts/training/launch_pi05_mimicgen_sr.sh all
#
# Optional:
#   PI05_TASKS="coffee stack" NUM_TRIALS=50 PI05_GPUS="0 1" bash ... all

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_DIR="${CONDA_DIR:-${CONDA_BASE}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$CONDA_DIR/envs/openpi05/bin/python}"
MIMICGEN_PYTHON="${MIMICGEN_PYTHON:-$CONDA_DIR/envs/mimicgen_env/bin/python}"
EVAL_SCRIPT="${MIMICGEN_EVAL_SCRIPT:-$PROJECT_DIR/Phoenix/evaluation/evaluate_mimicgen.py}"
OPENPI_SERVE_POLICY="${OPENPI_SERVE_POLICY:-$PROJECT_DIR/error_benchmark/scripts/training/start_openpi_serve_policy_safe.py}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/error_benchmark/outputs/pi05_mimicgen_sr}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/error_benchmark/outputs/logs/pi05_mimicgen_sr}"
VIDEO_DIR="${VIDEO_DIR:-$OUTPUT_DIR/videos}"
NUM_TRIALS_WAS_SET="${NUM_TRIALS+x}"
NUM_TRIALS="${NUM_TRIALS:-50}"
BASE_PORT="${BASE_PORT:-8000}"
PI05_STEP="${PI05_STEP:-}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
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

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$VIDEO_DIR"

log() {
  echo "[pi05-mimicgen-sr] $*"
}

die() {
  echo "[pi05-mimicgen-sr] ERROR: $*" >&2
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

finetune_config() {
  echo "pi05_benchmark_${1}_merged_finetune"
}

inference_config() {
  echo "pi05_benchmark_${1}_merged_inference"
}

eval_name() {
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

checkpoint_root() {
  echo "$OPENPI_DIR/checkpoints/$(finetune_config "$1")/merged"
}

latest_checkpoint() {
  local root
  root="$(checkpoint_root "$1")"
  [[ -d "$root" ]] || return 0
  if [[ -n "$PI05_STEP" ]]; then
    if [[ -d "$root/$PI05_STEP" ]]; then
      echo "$PI05_STEP"
    fi
    return 0
  fi
  find "$root" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

check_paths() {
  [[ -d "$PROJECT_DIR" ]] || die "project dir not found: $PROJECT_DIR"
  [[ -d "$OPENPI_DIR" ]] || die "openpi dir not found: $OPENPI_DIR"
  [[ -x "$OPENPI_PYTHON" ]] || die "openpi05 python not executable: $OPENPI_PYTHON"
  [[ -x "$MIMICGEN_PYTHON" ]] || die "mimicgen_env python not executable: $MIMICGEN_PYTHON"
  [[ -f "$EVAL_SCRIPT" ]] || die "MimicGen eval script not found: $EVAL_SCRIPT"
  [[ -f "$OPENPI_SERVE_POLICY" ]] || die "OpenPI serve policy wrapper not found: $OPENPI_SERVE_POLICY"
  for task in "${TASKS[@]}"; do
    validate_task "$task"
  done
}

print_status() {
  check_paths
  log "project: $PROJECT_DIR"
  log "openpi: $OPENPI_DIR"
  log "eval script: $EVAL_SCRIPT"
  log "output: $OUTPUT_DIR"
  log "trials/task: $NUM_TRIALS"
  log "server timeout: $SERVER_TIMEOUT sec"
  if [[ -n "$PI05_STEP" ]]; then
    log "forced checkpoint step: $PI05_STEP"
  fi
  echo
  printf "%-24s %-24s %-12s %s\n" "task" "mimicgen_eval_task" "latest_step" "checkpoint_root"
  for task in "${TASKS[@]}"; do
    local step root
    step="$(latest_checkpoint "$task" || true)"
    root="$(checkpoint_root "$task")"
    [[ -n "$step" ]] || step="-"
    printf "%-24s %-24s %-12s %s\n" "$task" "$(eval_name "$task")" "$step" "$root"
  done
}

wait_for_port() {
  local port="$1"
  local timeout="${2:-300}"
  local waited=0
  while ! python3 - "$port" <<'PY' >/dev/null 2>&1
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

warmup_server() {
  local port="$1"
  CUDA_VISIBLE_DEVICES="" "$MIMICGEN_PYTHON" - "$port" <<'PY'
import sys
import numpy as np
from openpi_client import websocket_client_policy as wcp
port = int(sys.argv[1])
client = wcp.WebsocketClientPolicy("localhost", port)
dummy = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "test",
}
client.infer(dummy)
print(f"warmup done on port {port}")
PY
}

run_one() {
  local task="$1"
  local gpu="$2"
  local port="$3"
  local eval_task cfg step ckpt_dir result_file server_log eval_log server_pid eval_exit

  validate_task "$task"
  eval_task="$(eval_name "$task")"
  cfg="$(inference_config "$task")"
  step="$(latest_checkpoint "$task" || true)"
  [[ -n "$step" ]] || die "no checkpoint step found for $task under $(checkpoint_root "$task")"
  ckpt_dir="$(checkpoint_root "$task")/$step"
  result_file="$OUTPUT_DIR/results_${task}_step${step}.json"
  server_log="$LOG_DIR/server_${task}_${TIMESTAMP}.log"
  eval_log="$LOG_DIR/eval_${task}_${TIMESTAMP}.log"

  log "starting $task ($eval_task), step=$step, gpu=$gpu, port=$port"

  (
    cd "$OPENPI_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" OPENPI_DIR="$OPENPI_DIR" "$OPENPI_PYTHON" "$OPENPI_SERVE_POLICY" \
      --port "$port" \
      policy:checkpoint \
      --policy.config "$cfg" \
      --policy.dir "$ckpt_dir"
  ) > "$server_log" 2>&1 &
  server_pid=$!

  if ! wait_for_port "$port" "$SERVER_TIMEOUT"; then
    log "$task server did not become ready; see $server_log"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi

  log "$task server ready; warming up"
  if ! warmup_server "$port" >> "$server_log" 2>&1; then
    log "$task warmup failed; continuing to eval"
  fi

  (
    cd "$PROJECT_DIR"
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID="$gpu"
    "$MIMICGEN_PYTHON" "$EVAL_SCRIPT" \
      --args.host localhost \
      --args.port "$port" \
      --args.tasks "$eval_task" \
      --args.num-trials-per-task "$NUM_TRIALS" \
      --args.results-out-path "$result_file" \
      --args.video-out-path "$VIDEO_DIR/$task" \
      --args.no-save-videos
  ) > "$eval_log" 2>&1
  eval_exit=$?

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true

  if [[ "$eval_exit" -ne 0 ]]; then
    log "$task eval failed; see $eval_log"
    return "$eval_exit"
  fi

  if [[ -f "$result_file" ]]; then
    "$MIMICGEN_PYTHON" - "$result_file" "$eval_task" <<'PY'
import json
import sys
path, task = sys.argv[1], sys.argv[2]
data = json.load(open(path))
res = data.get(task, {})
print(f"[pi05-mimicgen-sr] {task}: SR={res.get('success_rate', 0):.1%} ({res.get('successes', 0)}/{res.get('episodes', 0)})")
PY
  else
    log "$task finished but result file missing: $result_file"
    return 1
  fi
}

run_all() {
  check_paths
  [[ "${#GPUS[@]}" -ge "${#TASKS[@]}" ]] || die "need ${#TASKS[@]} GPUs, got ${#GPUS[@]}: ${GPUS[*]}"

  export PYTHONPATH="$PROJECT_DIR/shared/mimicgen_workspace/robosuite:$PROJECT_DIR/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true

  pids=()
  names=()
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    run_one "$task" "$gpu" "$port" &
    pids+=("$!")
    names+=("$task")
    sleep 5
  done

  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "${names[$i]} done"
    else
      log "${names[$i]} failed"
      failed=1
    fi
  done

  print_summary
  [[ "$failed" -eq 0 ]] || die "one or more SR jobs failed"
}

print_summary() {
  echo
  printf "%-24s %-24s %-12s %-12s %s\n" "task" "mimicgen_eval_task" "step" "SR" "succ/episodes"
  for task in "${TASKS[@]}"; do
    local eval_task step file
    eval_task="$(eval_name "$task")"
    step="$(latest_checkpoint "$task" || true)"
    file="$OUTPUT_DIR/results_${task}_step${step}.json"
    if [[ -f "$file" ]]; then
      "$MIMICGEN_PYTHON" - "$task" "$eval_task" "$step" "$file" <<'PY'
import json
import sys
task, eval_task, step, file = sys.argv[1:5]
data = json.load(open(file))
res = data.get(eval_task, {})
print(f"{task:<24} {eval_task:<24} {step:<12} {res.get('success_rate', 0):<12.1%} {res.get('successes', 0)}/{res.get('episodes', 0)}")
PY
    else
      printf "%-24s %-24s %-12s %-12s %s\n" "$task" "$eval_task" "${step:-"-"}" "-" "missing"
    fi
  done
}

case "$MODE" in
  status)
    print_status
    ;;
  smoke)
    if [[ -z "$NUM_TRIALS_WAS_SET" ]]; then
      NUM_TRIALS=1
    fi
    if [[ -z "${PI05_TASKS:-}" ]]; then
      TASKS=(stack)
    fi
    print_status
    run_all
    ;;
  all)
    run_all
    ;;
  summary)
    check_paths
    print_summary
    ;;
  *)
    die "unknown mode '$MODE'. Use: status, smoke, all, summary"
    ;;
esac
