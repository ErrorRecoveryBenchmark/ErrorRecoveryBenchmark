#!/bin/bash
# Prepare MimicGen-generated datasets with datagen_info
# This runs prepare_src_dataset.py to extract eef_pose, object_poses, target_pose, etc.

set -e

export PATH="${CONDA_BASE}/bin:$PATH"
source activate mimicgen_env

export PYTHONPATH="$(pwd)/shared/mimicgen_workspace/robosuite:$(pwd)/shared/mimicgen_workspace/mimicgen:$(pwd):${PYTHONPATH:-}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5
export MUJOCO_EGL_DEVICE_ID=0

PROJECT_ROOT="${BENCHMARK_ROOT}"
OUTPUT_DIR="${PROJECT_ROOT}/data/mimicgen_prepared"

mkdir -p "${OUTPUT_DIR}"

# Task configs: dataset_path -> (env_interface_name, output_filename)
declare -A TASKS
TASKS["stack"]="${BENCHMARK_DATA}/mimicgen_prepared/stack_d0.hdf5:MG_Stack:stack_d0_prepared.hdf5"
TASKS["pick_place"]="${BENCHMARK_DATA}/mimicgen_prepared/pick_place_d0.hdf5:MG_PickPlace:pick_place_d0_prepared.hdf5"
TASKS["coffee"]="${BENCHMARK_DATA}/mimicgen_prepared/coffee_d0.hdf5:MG_Coffee:coffee_d0_prepared.hdf5"
TASKS["threading"]="${BENCHMARK_DATA}/mimicgen_prepared/threading_d0.hdf5:MG_Threading:threading_d0_prepared.hdf5"
TASKS["stack_three"]="${BENCHMARK_DATA}/mimicgen_prepared/stack_three_d0.hdf5:MG_StackThree:stack_three_d0_prepared.hdf5"
TASKS["three_piece_assembly"]="${BENCHMARK_DATA}/mimicgen_prepared/three_piece_assembly_d0.hdf5:MG_ThreePieceAssembly:three_piece_assembly_d0_prepared.hdf5"

echo "=== Starting prepare_src_dataset.py batch run ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

for task in "${!TASKS[@]}"; do
    IFS=':' read -r dataset_path env_interface output_file <<< "${TASKS[$task]}"

    output_path="${OUTPUT_DIR}/${output_file}"

    if [[ -f "${output_path}" ]]; then
        echo "[SKIP] ${task}: output already exists at ${output_path}"
        continue
    fi

    echo "[RUN] ${task}: preparing ${dataset_path} -> ${output_path}"
    echo "  env_interface: ${env_interface}"
    echo "  Started at: $(date)"

    python shared/mimicgen_workspace/mimicgen/mimicgen/scripts/prepare_src_dataset.py \
        --dataset "${dataset_path}" \
        --env_interface "${env_interface}" \
        --env_interface_type robosuite \
        --output "${output_path}" \
        2>&1 | tee "${OUTPUT_DIR}/${task}_prepare.log"

    echo "  Finished at: $(date)"
    echo ""
done

echo "=== Batch preparation complete ==="
echo "Prepared datasets:"
ls -la "${OUTPUT_DIR}"/*.hdf5 2>/dev/null || echo "No HDF5 files found"