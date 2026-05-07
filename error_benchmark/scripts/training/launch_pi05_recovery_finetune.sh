#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 clean-20k -> recovery-data LoRA finetune.
#
# Usage:
#   bash error_benchmark/scripts/training/launch_pi05_recovery_finetune.sh status
#   bash error_benchmark/scripts/training/launch_pi05_recovery_finetune.sh norms
#   bash error_benchmark/scripts/training/launch_pi05_recovery_finetune.sh train
#   bash error_benchmark/scripts/training/launch_pi05_recovery_finetune.sh all
#
# Optional:
#   PI05_GPUS="0 1 2 3 4 5" bash ... train
#   PI05_TASKS="pick_place stack" bash ... train
#   PI05_EXP_NAME=recovery_from_clean20k_v2_10k bash ... train
#   PI05_RUN_MODE=--resume bash ... train

PROJECT_DIR="${ERROR_RECOVERY_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
OPENPI_DIR="${OPENPI_DIR:-${BENCHMARK_ROOT}/shared_deps/openpi}"
CONDA_EXE="${CONDA_EXE:-${CONDA_BASE}/bin/conda}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_CACHE}/lerobot}"

LOG_DIR="${PI05_LOG_DIR:-$PROJECT_DIR/error_benchmark/outputs/logs/pi05_recovery_finetune}"
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

EXP_NAME="${PI05_EXP_NAME:-recovery_from_clean20k_v2_10k}"
NUM_TRAIN_STEPS="${PI05_NUM_TRAIN_STEPS:-10000}"
TARGET_STEP="${PI05_TARGET_STEP:-$((NUM_TRAIN_STEPS - 1))}"
SAVE_INTERVAL="${PI05_SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${PI05_KEEP_PERIOD:-1000}"
FSDP_DEVICES="${PI05_FSDP_DEVICES:-1}"
RUN_MODE="${PI05_RUN_MODE:---overwrite}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
XLA_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
WANDB_FLAG="${PI05_WANDB_FLAG:---wandb-enabled}"
ASSETS_BASE_DIR="${PI05_ASSETS_BASE_DIR:-$OPENPI_DIR/assets_recovery_from_clean20k}"
MAX_NORM_FRAMES="${PI05_MAX_NORM_FRAMES:-}"

# Match zhaoganlong/Motion-based-Self-Reflection-Framework/deps/openpi
# add_finetune_config_lr(..._lr_v2): Pi0.5 LoRA, batch 64, 5e-5 cosine
# decay to 5e-6, save/keep every 1000, EMA off in the base config.
BATCH_SIZE="${PI05_BATCH_SIZE:-64}"
LR_WARMUP_STEPS="${PI05_LR_WARMUP_STEPS:-1000}"
LR_PEAK="${PI05_LR_PEAK:-5e-5}"
LR_DECAY_STEPS="${PI05_LR_DECAY_STEPS:-35000}"
LR_DECAY_LR="${PI05_LR_DECAY_LR:-5e-6}"
OPT_ACCUM_STEPS="${PI05_OPT_ACCUM_STEPS:-1}"

mkdir -p "$LOG_DIR"

log() {
  echo "[pi05-recovery] $*"
}

die() {
  echo "[pi05-recovery] ERROR: $*" >&2
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

recovery_repo_id() {
  local task="$1"
  echo "benchmark/mimicgen_${task}_recovery_merged"
}

dataset_dir() {
  local task="$1"
  echo "$HF_LEROBOT_HOME/$(recovery_repo_id "$task")"
}

norm_stats_dir() {
  local task="$1"
  echo "$ASSETS_BASE_DIR/$(config_name "$task")/$(recovery_repo_id "$task")"
}

target_checkpoint_dir() {
  local task="$1"
  echo "$OPENPI_DIR/checkpoints/$(config_name "$task")/$EXP_NAME"
}

init_checkpoint_dir() {
  local task="$1"
  case "$task" in
    pick_place|coffee|threading)
      echo "$OPENPI_DIR/checkpoints/$(config_name "$task")/merged_lr_original/19999"
      ;;
    stack|stack_three|three_piece_assembly)
      echo "$OPENPI_DIR/checkpoints/$(config_name "$task")/merged/20000"
      ;;
    *)
      die "unknown task '$task'"
      ;;
  esac
}

init_params_path() {
  local task="$1"
  echo "$(init_checkpoint_dir "$task")/params"
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
  [[ -f "$OPENPI_DIR/scripts/train.py" ]] || die "train script not found: $OPENPI_DIR/scripts/train.py"
  [[ -f "$OPENPI_DIR/scripts/compute_norm_stats.py" ]] || die "norm script not found: $OPENPI_DIR/scripts/compute_norm_stats.py"

  case "$EXP_NAME" in
    merged|merged_lr_original)
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
  local data_status="missing"
  local norm_status="missing"
  local init_status="missing"
  local latest="-"
  local target_dir

  [[ -d "$(dataset_dir "$task")" ]] && data_status="ok"
  [[ -d "$(norm_stats_dir "$task")" ]] && norm_status="ok"
  [[ -d "$(init_params_path "$task")" ]] && init_status="ok"

  target_dir="$(target_checkpoint_dir "$task")"
  latest="$(latest_step "$target_dir" || true)"
  [[ -n "$latest" ]] || latest="-"

  printf "%-24s %-5s %-8s %-8s %-8s %-11s %s\n" \
    "$task" "${gpu:-"-"}" "$data_status" "$norm_status" "$init_status" "$latest" "$target_dir"
}

print_status() {
  check_common_paths
  log "project: $PROJECT_DIR"
  log "openpi: $OPENPI_DIR"
  log "lerobot home: $HF_LEROBOT_HOME"
  log "exp_name: $EXP_NAME"
  log "num_train_steps: $NUM_TRAIN_STEPS target checkpoint step: $TARGET_STEP"
  log "zhaoganlong lr_v2 defaults: batch=$BATCH_SIZE warmup=$LR_WARMUP_STEPS peak_lr=$LR_PEAK decay_steps=$LR_DECAY_STEPS decay_lr=$LR_DECAY_LR accumulation=$OPT_ACCUM_STEPS"
  log "run mode: $RUN_MODE"
  log "assets base: $ASSETS_BASE_DIR"
  echo
  printf "%-24s %-5s %-8s %-8s %-8s %-11s %s\n" \
    "task" "gpu" "dataset" "norms" "init" "latest" "checkpoint_dir"
  for i in "${!TASKS[@]}"; do
    status_for_task "${TASKS[$i]}" "${GPUS[$i]:-}"
  done
}

check_train_inputs() {
  local missing=0
  for task in "${TASKS[@]}"; do
    if [[ ! -d "$(dataset_dir "$task")" ]]; then
      log "missing recovery dataset for $task: $(dataset_dir "$task")"
      missing=1
    fi
    if [[ ! -d "$(norm_stats_dir "$task")" ]]; then
      log "missing recovery norm stats for $task: $(norm_stats_dir "$task")"
      missing=1
    fi
    if [[ ! -d "$(init_params_path "$task")" ]]; then
      log "missing init params for $task: $(init_params_path "$task")"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "required training inputs are missing"
}

compute_norms_one() {
  local task="$1"
  local cfg repo log_file
  cfg="$(config_name "$task")"
  repo="$(recovery_repo_id "$task")"
  log_file="$LOG_DIR/norms_${task}_${TIMESTAMP}.log"

  [[ -d "$(dataset_dir "$task")" ]] || die "dataset missing for $task: $(dataset_dir "$task")"

  log "computing norm stats for $task repo=$repo"
  (
    cd "$OPENPI_DIR"
    export HF_LEROBOT_HOME
    "$CONDA_EXE" run -n openpi05 python - "$cfg" "$repo" "$ASSETS_BASE_DIR" "$BATCH_SIZE" "${MAX_NORM_FRAMES:-}" <<'PY'
import dataclasses
import sys

import numpy as np
import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.shared import normalize
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


cfg_name, repo_id, assets_base_dir, batch_size_arg, max_frames_arg = sys.argv[1:6]
batch_size = int(batch_size_arg)
max_frames = int(max_frames_arg) if max_frames_arg else None

cfg = _config.get_config(cfg_name)
cfg = dataclasses.replace(
    cfg,
    assets_base_dir=assets_base_dir,
    batch_size=batch_size,
    data=dataclasses.replace(cfg.data, repo_id=repo_id),
)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
dataset = _data_loader.TransformedDataset(
    dataset,
    [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ],
)

if max_frames is not None and max_frames < len(dataset):
    num_batches = max_frames // cfg.batch_size
    shuffle = True
else:
    num_batches = len(dataset) // cfg.batch_size
    shuffle = False

data_iter = _data_loader.TorchDataLoader(
    dataset,
    local_batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    shuffle=shuffle,
    num_batches=num_batches,
)

stats = {key: normalize.RunningStats() for key in ("state", "actions")}
for batch in tqdm.tqdm(data_iter, total=num_batches, desc=f"Computing stats for {repo_id}"):
    for key in stats:
        stats[key].update(np.asarray(batch[key]))

norm_stats = {key: value.get_statistics() for key, value in stats.items()}
output_path = cfg.assets_dirs / data_config.asset_id
print(f"Writing stats to: {output_path}")
normalize.save(output_path, norm_stats)
PY
  ) 2>&1 | tee "$log_file"
}

compute_norms() {
  check_common_paths
  for task in "${TASKS[@]}"; do
    compute_norms_one "$task"
  done
  print_status
}

select_train_tasks() {
  TRAIN_TASKS=()
  for task in "${TASKS[@]}"; do
    local step
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

train_local() {
  check_common_paths
  check_train_inputs
  select_train_tasks

  if [[ "${#TRAIN_TASKS[@]}" -eq 0 ]]; then
    log "no tasks need training"
    return 0
  fi
  [[ "${#GPUS[@]}" -ge "${#TRAIN_TASKS[@]}" ]] || die "need ${#TRAIN_TASKS[@]} GPUs, got ${#GPUS[@]}: ${GPUS[*]}"

  log "launching ${#TRAIN_TASKS[@]} Pi0.5 recovery finetune jobs"
  log "tasks: ${TRAIN_TASKS[*]}"
  log "gpus: ${GPUS[*]}"
  log "logs: $LOG_DIR"

  pids=()
  names=()
  cd "$OPENPI_DIR"
  for i in "${!TRAIN_TASKS[@]}"; do
    local task gpu cfg repo init_params task_log
    task="${TRAIN_TASKS[$i]}"
    gpu="${GPUS[$i]}"
    cfg="$(config_name "$task")"
    repo="$(recovery_repo_id "$task")"
    init_params="$(init_params_path "$task")"
    task_log="$LOG_DIR/train_${task}_${TIMESTAMP}.log"

    log "starting $task on GPU $gpu from $init_params"
    (
      export HF_LEROBOT_HOME
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
      "$CONDA_EXE" run -n openpi05 python scripts/train.py "$cfg" \
        --exp-name="$EXP_NAME" \
        "$RUN_MODE" \
        --num-train-steps="$NUM_TRAIN_STEPS" \
        --batch-size="$BATCH_SIZE" \
        --save-interval="$SAVE_INTERVAL" \
        --keep-period="$KEEP_PERIOD" \
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
      log "${names[$i]} finished"
    else
      log "${names[$i]} failed; see $LOG_DIR/train_${names[$i]}_${TIMESTAMP}.log"
      failed=1
    fi
  done

  print_status
  [[ "$failed" -eq 0 ]] || die "one or more Pi0.5 recovery finetune jobs failed"
}

case "$MODE" in
  status)
    print_status
    ;;
  norms)
    compute_norms
    ;;
  train)
    train_local
    ;;
  all)
    compute_norms
    train_local
    ;;
  *)
    die "unknown mode '$MODE'. Use: status, norms, train, all"
    ;;
esac
