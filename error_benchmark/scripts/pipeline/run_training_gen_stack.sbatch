#!/bin/bash
#SBATCH --job-name=v5_train_stack
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=${BENCHMARK_ROOT}/error_benchmark/outputs/v5_training_v2/slurm_%j.log

PROJECT_ROOT="${BENCHMARK_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

# Conda — adjust to your environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mimicgen_env

# MuJoCo EGL rendering
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# PYTHONPATH for local robosuite/mimicgen
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/shared/mimicgen_workspace/robosuite:$PROJECT_ROOT/shared/mimicgen_workspace/mimicgen:$PYTHONPATH"

echo "=== v5 Training Scene Generation (stack) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "Start: $(date)"
echo ""

# Create output dir
mkdir -p "$PROJECT_ROOT/error_benchmark/outputs/v5_training_v2/stack/scenes"

# Run generation
python error_benchmark/scripts/pipeline/generate_training_scenes.py \
    --task stack \
    --scenes_per_subtype 10 \
    --output_dir error_benchmark/outputs/v5_training_v2 \
    --seed 42

echo ""
echo "=== Generation Complete ==="
echo "End: $(date)"

# Quick summary
echo ""
echo "=== Output Summary ==="
SCENES_DIR="$PROJECT_ROOT/error_benchmark/outputs/v5_training_v2/stack/scenes"
echo "Total scene files: $(ls "$SCENES_DIR"/*.npz 2>/dev/null | wc -l)"
echo ""

# Check coverage from meta.json
python -c "
import json, glob
from collections import defaultdict

meta_path = '$PROJECT_ROOT/error_benchmark/outputs/v5_training_v2/stack/meta.json'
try:
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'Total scenes: {meta[\"total_scenes\"]}')
    print(f'Coverage gaps: {len(meta.get(\"coverage_gaps\", []))}')
    for gap in meta.get('coverage_gaps', []):
        print(f'  GAP: {gap}')
    print()
    print('Per-object coverage:')
    cov = defaultdict(set)
    for s in meta.get('scenes', []):
        obj = s.get('target_object', '')
        err = s.get('error_name', '')
        if obj and err:
            cov[obj].add(err)
    for obj, errors in sorted(cov.items()):
        print(f'  {obj}: {len(errors)} error types')
except Exception as e:
    print(f'Could not read meta.json: {e}')
"
