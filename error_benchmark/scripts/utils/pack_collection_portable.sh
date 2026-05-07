#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Pack Recovery Collection Portable Directory
# ═══════════════════════════════════════════════════════════════
# Creates a self-contained directory that can be scp'd to a machine
# with a display + SpaceMouse for recovery demo collection.
#
# Tasks are derived from error_benchmark/configs/task_registry.yaml.
#
# Usage:
#   bash error_benchmark/scripts/utils/pack_collection_portable.sh
#
# Output:
#   ~/recovery_collection_portable/
#   ~/recovery_collection_portable.tar.gz

set -euo pipefail

# Resolve project root (this script is at error_benchmark/scripts/utils/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

PYTHON="${CONDA_PREFIX:-${CONDA_DIR:-$HOME/miniconda3}/envs/mimicgen_env}/bin/python"
SOURCE_REGISTRY="$PROJECT_ROOT/error_benchmark/configs/task_registry.yaml"

if [ ! -f "$SOURCE_REGISTRY" ]; then
    echo "ERROR: Task registry not found: $SOURCE_REGISTRY"
    exit 1
fi

mapfile -t TASKS < <("$PYTHON" - "$SOURCE_REGISTRY" <<'PY'
import sys
from pathlib import Path

import yaml

registry_path = Path(sys.argv[1])
with registry_path.open() as f:
    registry = yaml.safe_load(f) or {}

tasks = registry.get("tasks") or {}
if not tasks:
    raise SystemExit(f"ERROR: No tasks found in {registry_path}")

for task_name in tasks:
    print(task_name)
PY
)

if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "ERROR: No tasks discovered from $SOURCE_REGISTRY"
    exit 1
fi

echo "Registry tasks: ${TASKS[*]}"

echo "Validating registry tasks..."
"$PYTHON" - "$PROJECT_ROOT" "$SOURCE_REGISTRY" <<'PY'
import sys
from pathlib import Path

import yaml

project_root = Path(sys.argv[1])
registry_path = Path(sys.argv[2])

with registry_path.open() as f:
    registry = yaml.safe_load(f) or {}

tasks = registry.get("tasks") or {}
errors = []

for task_name, info in tasks.items():
    task_config = info.get("task_config")
    dataset_path = info.get("dataset_path")
    scenes_dir = project_root / "error_benchmark" / "outputs" / "v5_training" / task_name / "scenes"

    if not task_config:
        errors.append(f"- {task_name}: missing task_config in {registry_path}")
    else:
        task_config_path = Path(task_config)
        if not task_config_path.is_absolute():
            task_config_path = project_root / task_config_path
        if not task_config_path.is_file():
            errors.append(f"- {task_name}: task_config not found at {task_config_path}")

    if not dataset_path:
        errors.append(f"- {task_name}: missing dataset_path in {registry_path}")
    else:
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.is_absolute():
            dataset_path_obj = project_root / dataset_path_obj
        if not dataset_path_obj.is_file():
            errors.append(f"- {task_name}: dataset_path not found at {dataset_path_obj}")

    if not scenes_dir.is_dir():
        errors.append(f"- {task_name}: scenes directory not found at {scenes_dir}")
    else:
        if not any(scenes_dir.glob("*.json")):
            errors.append(f"- {task_name}: no scene JSON files found in {scenes_dir}")
        if not any(scenes_dir.glob("*.npz")):
            errors.append(f"- {task_name}: no scene NPZ files found in {scenes_dir}")

if errors:
    print("ERROR: portable pack validation failed:", file=sys.stderr)
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

print(f"Validated {len(tasks)} task(s) from {registry_path}")
PY

PACK_DIR="$HOME/recovery_collection_portable"
echo "Pack target:  $PACK_DIR"

# Clean previous pack
if [ -d "$PACK_DIR" ]; then
    echo "Removing previous pack directory..."
    rm -rf "$PACK_DIR"
fi

mkdir -p "$PACK_DIR"

# ─── 1. error_benchmark code ───
echo "Copying error_benchmark code..."
mkdir -p "$PACK_DIR/error_benchmark"
cp "$PROJECT_ROOT/error_benchmark/__init__.py" "$PACK_DIR/error_benchmark/"

# framework/
rsync -a --include='*.py' --exclude='__pycache__' \
    "$PROJECT_ROOT/error_benchmark/framework/" \
    "$PACK_DIR/error_benchmark/framework/"

# scripts/
mkdir -p "$PACK_DIR/error_benchmark/scripts/collection"
mkdir -p "$PACK_DIR/error_benchmark/scripts/utils"
mkdir -p "$PACK_DIR/error_benchmark/scripts/augmentation"
cp "$PROJECT_ROOT/error_benchmark/scripts/__init__.py" "$PACK_DIR/error_benchmark/scripts/" 2>/dev/null || touch "$PACK_DIR/error_benchmark/scripts/__init__.py"
cp "$PROJECT_ROOT/error_benchmark/scripts/collection/2_collect_recovery_demos.py" "$PACK_DIR/error_benchmark/scripts/collection/"
touch "$PACK_DIR/error_benchmark/scripts/collection/__init__.py"
cp "$PROJECT_ROOT/error_benchmark/scripts/utils/script_utils.py" "$PACK_DIR/error_benchmark/scripts/utils/"
cp "$PROJECT_ROOT/error_benchmark/scripts/utils/__init__.py" "$PACK_DIR/error_benchmark/scripts/utils/" 2>/dev/null || touch "$PACK_DIR/error_benchmark/scripts/utils/__init__.py"
cp "$PROJECT_ROOT/error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py" "$PACK_DIR/error_benchmark/scripts/augmentation/"
cp "$PROJECT_ROOT/error_benchmark/scripts/augmentation/__init__.py" "$PACK_DIR/error_benchmark/scripts/augmentation/" 2>/dev/null || touch "$PACK_DIR/error_benchmark/scripts/augmentation/__init__.py"

# ─── 1b. configs (derived from registry tasks) ───
echo "Copying configs..."
mkdir -p "$PACK_DIR/error_benchmark/configs/tasks"
cp "$PROJECT_ROOT/error_benchmark/configs/recovery_collection.yaml" "$PACK_DIR/error_benchmark/configs/"
cp "$PROJECT_ROOT/error_benchmark/configs/logging.yaml" "$PACK_DIR/error_benchmark/configs/" 2>/dev/null || true
"$PYTHON" - "$PROJECT_ROOT" "$PACK_DIR" "$SOURCE_REGISTRY" <<'PY'
import shutil
import sys
from pathlib import Path

import yaml

project_root = Path(sys.argv[1])
pack_dir = Path(sys.argv[2])
registry_path = Path(sys.argv[3])

with registry_path.open() as f:
    registry = yaml.safe_load(f) or {}

tasks = registry.get("tasks") or {}
portable_registry = {"tasks": {}}

for task_name, info in tasks.items():
    portable_info = dict(info)

    task_config_path = Path(info["task_config"])
    if not task_config_path.is_absolute():
        task_config_path = project_root / task_config_path

    portable_task_config = Path("error_benchmark/configs/tasks") / task_config_path.name
    shutil.copy2(task_config_path, pack_dir / portable_task_config)

    portable_info["task_config"] = portable_task_config.as_posix()
    portable_info["dataset_path"] = f"data/{task_name}.hdf5"
    portable_registry["tasks"][task_name] = portable_info

registry_out = pack_dir / "error_benchmark" / "configs" / "task_registry.yaml"
with registry_out.open("w") as f:
    f.write("# Portable task registry (relative paths for recovery collection)\n")
    yaml.safe_dump(portable_registry, f, sort_keys=False)

print(f"Wrote portable registry: {registry_out}")
PY

# ─── 2. v5_training error scenes (derived from registry tasks) ───
echo "Copying v5_training error scenes..."
TOTAL_SCENES=0
for task in "${TASKS[@]}"; do
    SCENES_SRC="$PROJECT_ROOT/error_benchmark/outputs/v5_training/$task/scenes"
    SCENES_DST="$PACK_DIR/error_benchmark/outputs/v5_training/$task/scenes"
    mkdir -p "$SCENES_DST"
    cp "$SCENES_SRC"/*.json "$SCENES_DST/"
    cp "$SCENES_SRC"/*.npz "$SCENES_DST/"
    SCENE_COUNT=$(find "$SCENES_DST" -maxdepth 1 -type f -name '*.json' | wc -l)
    TOTAL_SCENES=$((TOTAL_SCENES + SCENE_COUNT))
    echo "  $task: $SCENE_COUNT scenes"
done
echo "  Total: $TOTAL_SCENES scenes"

# ─── 2b. MimicGen prepared source datasets ───
echo "Copying prepared MimicGen source datasets..."
PREPARED_SRC_DIR="$PROJECT_ROOT/error_benchmark/outputs/mimicgen_success/prepared_source"
PREPARED_DST_DIR="$PACK_DIR/error_benchmark/outputs/mimicgen_success/prepared_source"
if [ -d "$PREPARED_SRC_DIR" ]; then
    mkdir -p "$PREPARED_DST_DIR"
    cp "$PREPARED_SRC_DIR"/*.hdf5 "$PREPARED_DST_DIR/" 2>/dev/null || true
    ls -lh "$PREPARED_DST_DIR" 2>/dev/null || true
else
    echo "  WARNING: Missing prepared source dir: $PREPARED_SRC_DIR"
fi

# ─── 3. robosuite version lock (installed locally, not bundled) ───
echo "Writing robosuite version lock..."
ROBOSUITE_COMMIT=$(git -C "$PROJECT_ROOT/shared/mimicgen_workspace/robosuite" rev-parse HEAD)
cat > "$PACK_DIR/ROBOSUITE_VERSION_LOCK.txt" <<EOF
robosuite_version=1.4.1
robosuite_commit=${ROBOSUITE_COMMIT}
robosuite_install_url=git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_COMMIT}#egg=robosuite
EOF

# ─── 4. mimicgen ───
echo "Copying mimicgen (filtered)..."
MIMICGEN_SRC="$PROJECT_ROOT/shared/mimicgen_workspace/mimicgen"
MIMICGEN_DST="$PACK_DIR/shared/mimicgen_workspace/mimicgen"
mkdir -p "$MIMICGEN_DST"

rsync -a \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='docs/' \
    --exclude='datasets/' \
    --exclude='*.egg-info' \
    "$MIMICGEN_SRC/" "$MIMICGEN_DST/"

# ─── 5. HDF5 stub datasets ───
echo "Creating HDF5 stub datasets..."
mkdir -p "$PACK_DIR/data"
"$PYTHON" "$PROJECT_ROOT/error_benchmark/scripts/utils/create_hdf5_stubs.py" \
    --registry "$SOURCE_REGISTRY" \
    --output "$PACK_DIR/data/" \
    --tasks "${TASKS[@]}"

# ─── 6. Top-level scripts ───
echo "Copying top-level scripts..."
cp "$PROJECT_ROOT/error_benchmark/scripts/collection/run_collection.sh" "$PACK_DIR/run_collection.sh"
chmod +x "$PACK_DIR/run_collection.sh"

# setup_env.sh (reuse existing or copy from scripts/collection/)
if [ -f "$PROJECT_ROOT/error_benchmark/scripts/collection/setup_env.sh" ]; then
    cp "$PROJECT_ROOT/error_benchmark/scripts/collection/setup_env.sh" "$PACK_DIR/setup_env.sh"
else
    # Copy from the old pack location if it exists
    OLD_SETUP="$HOME/recovery_collection_portable/setup_env.sh"
    if [ -f "$OLD_SETUP" ]; then
        cp "$OLD_SETUP" "$PACK_DIR/setup_env.sh"
    fi
fi
chmod +x "$PACK_DIR/setup_env.sh" 2>/dev/null || true

# TUTORIAL.md
cp "$PROJECT_ROOT/error_benchmark/scripts/collection/TUTORIAL.md" "$PACK_DIR/TUTORIAL.md"

# ─── 7. Output directory placeholder ───
mkdir -p "$PACK_DIR/outputs/recovery/demos"

# ─── Summary ───
echo ""
echo "═══════════════════════════════════════════════════════"
echo "Pack complete: $PACK_DIR"
echo "═══════════════════════════════════════════════════════"
du -sh "$PACK_DIR"
echo ""
echo "Tasks: ${TASKS[*]}"
echo ""
echo "Contents:"
du -sh "$PACK_DIR"/* 2>/dev/null | sort -rh
echo ""

# ─── 8. Create tar.gz ───
echo "Creating tar.gz archive..."
TAR_PATH="$HOME/recovery_collection_portable.tar.gz"
cd "$(dirname "$PACK_DIR")"
tar czf "$TAR_PATH" "$(basename "$PACK_DIR")"
echo ""
echo "Archive: $TAR_PATH"
ls -lh "$TAR_PATH"
echo ""
echo "Next steps:"
echo "  1. scp $TAR_PATH user@target:/path/"
echo "  2. On target: tar xzf recovery_collection_portable.tar.gz"
echo "  3. cd recovery_collection_portable && bash setup_env.sh"
echo "  4. bash run_collection.sh <task> <subtype> [num_demos]"
