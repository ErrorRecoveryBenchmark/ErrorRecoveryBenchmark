# Error Recovery Benchmark

A robotics benchmark for collecting and augmenting **human recovery demonstrations** in MuJoCo-simulated manipulation tasks. A human operator teleoperates a Sawyer robot arm using a SpaceMouse, recovering from injected error states (e.g., dropped objects, misaligned grasps, collisions) and completing the original task.

## [Supplimentary Files](https://github.com/ErrorRecoveryBenchmark/ErrorRecoveryBenchmark/blob/main/Supplimentary_files.pdf)

## Overview

The benchmark pipeline has three stages:

1. **Error scene generation** (v5 pipeline) — inject errors into clean trajectories to produce error scenes
2. **Recovery demo collection** — human teleoperates from error scenes, producing recovery demonstrations
3. **MimicGen augmentation** — augment human demos via scene-configuration warping, cross-degree, and cross-subtype transfer

### Error Taxonomy

12 Error Skills x 2 Degrees (D0/D1) = 24 subtypes, grouped into 5 Recovery Behavior Groups (RBGs):

| RBG | Recovery Primitive | Error Skills |
|-----|-------------------|-------------|
| A (Re-grasp) | retract, re_orient, re_grasp, re_lift, re_transport, re_place | grasp_misalignment, grasp_wrong_pose |
| B (Retrieve) | navigate_to_object, re_grasp, re_lift, re_transport, re_place | drop_in_transit, drop_at_wrong_place |
| C (Retract) | retract, navigate_to_object | collision_holding/empty/eef_object/self |
| D (Redirect) | release, navigate_to_object, re_grasp, re_lift, re_transport, re_place | wrong_object |
| E (Realign) | correct_position, resume_task | trajectory_regression, stuck_no_progress, position_error |

### Tasks

6 manipulation tasks in total: pick_place, stack, coffee, threading, stack_three, three_piece_assembly

## Quick Start

### Prerequisites

- Python 3.10+
- MuJoCo 2.3.2
- Conda
- SpaceMouse (for recovery demo collection only)

### Environment Setup

```bash
# Option 1: Automated setup (Linux, creates conda env + installs all deps)
bash setup_env.sh

# Option 2: Manual setup
conda create -n error_recovery python=3.10 -y
conda activate error_recovery
pip install -e .

# Install MuJoCo
pip install mujoco==2.3.2

# Install robosuite and mimicgen from vendored submodules
git submodule update --init --recursive
pip install -e shared/mimicgen_workspace/robosuite
pip install -e shared/mimicgen_workspace/mimicgen
pip install -e shared/mimicgen_workspace/robosuite-task-zoo
```

### Set Environment Variables

```bash
export MUJOCO_GL=egl                  # For headless GPU rendering (use "glfw" for windowed display)
export PYTHONPATH="$(pwd):$(pwd)/shared/mimicgen_workspace/robosuite:$(pwd)/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
```

### Download Data

The full dataset (~30 GB) includes error scenes, human recovery demos, and MimicGen-augmented demos.

```bash
bash scripts/download_data.sh
```

See [examples/](examples/) for small sample data that lets you test the pipeline without the full dataset.

## Usage

### V5 Error Scene Pipeline

```bash
# Step 0: Collect clean trajectories
python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py \
    --config error_benchmark/configs/benchmark_v5.yaml \
    --task pick_place --label_phases

# Step 0b: Scan injection opportunities
python error_benchmark/scripts/pipeline/0b_scan_opportunities.py \
    --config error_benchmark/configs/benchmark_v5.yaml \
    --task pick_place

# Step 0c: Schedule injections to fill quotas
python error_benchmark/scripts/pipeline/0c_schedule_injections.py \
    --config error_benchmark/configs/benchmark_v5.yaml \
    --task pick_place

# Step 1: Execute injections to produce error scenes
python error_benchmark/scripts/pipeline/1_v5_execute_injections.py \
    --config error_benchmark/configs/benchmark_v5.yaml \
    --task pick_place

# Or run all steps for all tasks:
python error_benchmark/scripts/pipeline/run_v5_all_tasks.py \
    --config error_benchmark/configs/benchmark_v5.yaml
```

### Recovery Demo Collection

```bash
# Collect recovery demos for a single subtype
python error_benchmark/scripts/collection/2_collect_recovery_demos.py \
    --task pick_place \
    --subtype grasp_misalignment_D0 \
    --num_demos 8 \
    --scenes_dir error_benchmark/outputs/v5_training/pick_place/scenes \
    --device spacemouse

# Or collect all tasks/subtypes with the master script:
bash collect_all.sh
```

### Recovery Augmentation & Training Data

```bash
# Stage 3: MimicGen augmentation
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python \
    error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
    --task stack --target_per_subtype 100

# Stage 4A: Generate MCM training data
python error_benchmark/scripts/conversion/4a_generate_mcm_training_data.py \
    --task stack --config error_benchmark/configs/recovery_collection.yaml

# Stage 4B: Generate diffusion training data
python error_benchmark/scripts/conversion/4b_generate_diffusion_training_data.py \
    --task stack --config error_benchmark/configs/recovery_collection.yaml

# Stage 4C: Convert to BC-RNN HDF5 format
MUJOCO_GL=egl python error_benchmark/scripts/conversion/4c_convert_recovery_to_bc_rnn_hdf5.py \
    --task stack
```

### Training & Evaluation

```bash
# BC-RNN baseline training
bash error_benchmark/scripts/training/train_bc_rnn_baseline.sh

# Pi0.5 LoRA finetuning
python error_benchmark/scripts/training/train_pi05_merged.py --task stack

# Evaluate on error scenes
python error_benchmark/scripts/training/eval_pi05_error_scenes.py \
    --task stack --checkpoint <path_to_checkpoint>
```

## Architecture

### Core Framework (`error_benchmark/framework/`)

- `core.py` — Central dataclasses: ErrorSpec, ErrorScene, CleanTrajectory, etc.
- `env_wrapper.py` — Abstracts robosuite internals; all modules interact through this wrapper
- `error_taxonomy_v5.py` — ErrorSkillName enum, ErrorDegree enum, skill definitions
- `error_skills/` — One module per error skill (e01-e12)
- `recovery_types.py` — RecoveryDemo, RecoverySubtask, subtype-to-RBG mapping
- `recovery_segmenter.py` — Segments recovery demos into MimicGen-compatible subtask sequences
- `recovery_mimicgen.py` — Bridges recovery demos to MimicGen augmentation

### Pipeline Modules

- `clean_trajectory_collector.py` — Collects clean trajectories from HDF5 demos
- `opportunity_scanner.py` — Offline: replays each frame x each skill to build opportunity map
- `quota_scheduler.py` — Selects (frame, skill, degree) tuples to fill quotas
- `context_replay.py` — InjectionEngine: replays to injection frame, calls inject() + validate()

### Vendored Dependencies

`shared/mimicgen_workspace/` contains pinned versions of:
- **robosuite** (commit c848ca84) — MuJoCo-based robot learning framework
- **mimicgen** — Demonstration augmentation system
- **robosuite-task-zoo** — Custom task environments

See [ROBOSUITE_VERSION_LOCK.txt](ROBOSUITE_VERSION_LOCK.txt) for version pinning details.

## Data Format

### Error Scenes (`*.npz` + `*.json`)

Each error scene contains:
- `sim_state`: pre-injection MuJoCo state
- `post_sim_state`: post-injection stable state
- Quaternions in (w,x,y,z) order
- JSON metadata: error skill, degree, injection frame, dataset path

### Recovery Demos (`*.npz`)

Each recovery demonstration contains:
- `actions`: robot action sequence (T, action_dim)
- `states`: full MuJoCo state sequence (T, state_dim)
- `camera_images`: RGB observations (T, H, W, 3)
- `eef_positions`: end-effector positions (T, 3)
- `eef_orientations`: end-effector orientations (T, 4)
- `gripper_states`: gripper state (T, gripper_dim)
- `target_poses`: MimicGen-compatible waypoints
- `task_name`, `error_name`, `degree`, `scene_id`, `demo_id`: metadata
- `success`, `replay_passed`, `validation_passed`: quality flags

## Testing

```bash
# Unit tests (no GPU required)
pytest error_benchmark/tests/unit/test_v5_core.py -v

# All unit tests (requires robosuite installed)
pytest error_benchmark/tests/unit/ -v
```

## Citation

<!-- TODO: Add citation when paper is published -->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Vendored dependencies in `shared/mimicgen_workspace/` retain their original licenses.
