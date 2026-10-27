# Error Recovery Benchmark

A robotics benchmark for collecting and augmenting **human recovery demonstrations** in MuJoCo-simulated manipulation tasks. A human operator teleoperates a robot arm (Sawyer for pick_place, Panda for the other five tasks) using a SpaceMouse, recovering from injected error states (e.g., dropped objects, misaligned grasps, collisions) and completing the original task.

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
# Repository installation (Python 3.10 recommended)
conda create -n error_recovery python=3.10 -y
conda activate error_recovery
pip install -e ".[dev]"

# Install MuJoCo
pip install mujoco==2.3.2

# Install robosuite and mimicgen from vendored submodules
git submodule update --init --recursive
pip install -e shared/mimicgen_workspace/robosuite
pip install -e shared/mimicgen_workspace/mimicgen
pip install -e shared/mimicgen_workspace/robosuite-task-zoo
# Optional: MimicGen data generation / augmentation (paper version)
pip install -e ".[augmentation]"
```

On Windows/macOS, configure the pinned simulator's local display support after installation:

```bash
python scripts/configure_local_sim.py
```

This adds a local `macros_private.py` and, on Windows, copies the installed
MuJoCo 2.3.2 DLL to the location expected by the pinned robosuite. Existing private
settings are preserved. Use `MUJOCO_GL=glfw` for local display runs.

`3c_generate_from_error_scene.py` is the Linux/EGL batch
generator; it requires prepared MimicGen sources and training scenes. The
clean-start evaluator additionally requires `openpi-client` and `tyro` in the
policy-evaluation environment. These full policy runs require trained checkpoints.
Dependency patches are provided under `patches/`; review and run
`git apply --check` in the corresponding pinned submodule before applying them.

### Set Environment Variables

```bash
export MUJOCO_GL=egl                  # For headless GPU rendering (use "glfw" for windowed display)
export PYTHONPATH="$(pwd):$(pwd)/shared/mimicgen_workspace/robosuite:$(pwd)/shared/mimicgen_workspace/mimicgen:${PYTHONPATH:-}"
```

### Download Data

The v1.1.2 release contains five standard data archives (approximately 9.7 GB), with image observations stored directly in demonstration NPZ files. The corpus contains 973 human, 668 generated and 8,957 augmented trajectories (10,598 total). A separate 739-trajectory HDF5 artifact supports the reported recovery-policy experiments. See [data formats](docs/DATA_FORMAT.md) for image alignment and provenance, and [the release guide](docs/RELEASE.md) for reproducibility.

```bash
python scripts/download_data.py data
# Optional: also fetch upstream seed demos for scene generation
# python scripts/download_data.py data --seeds
```

See [examples/](examples/) for small samples and a CPU-only format check:

```bash
python scripts/inspect_examples.py
```

`setup_env.sh` is only for exported portable collection packages, not this repository.
The download script does not need the Hugging Face CLI. Pass `--archives eval_data.tar`
to fetch only the evaluation set. Select seed HDF5 paths in
`error_benchmark/configs/task_registry.yaml` when regenerating scenes.

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
# BC-RNN baseline training with the paper's configs (error_benchmark/configs/bc_rnn/, 50 epochs)
bash error_benchmark/scripts/training/train_bc_rnn_baseline.sh stack 0 recovery /path/to/stack_recovery.hdf5

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

The human NPZ format contains the following fields. Augmented files contain actions, states, eef_positions and gripper_states. See [DATA_FORMAT.md](docs/DATA_FORMAT.md).
- `actions`: robot action sequence (T, action_dim)
- `states`: full MuJoCo state sequence (T+1, state_dim) for the human NPZ format
- `camera_images`: optional RGB observations; frame count and resolution vary by recording
- `eef_positions`: end-effector positions (T+1, 3)
- `eef_orientations`: optional end-effector orientations (T+1, 4)
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

Zijia Tang, Ganlong Zhao, Xingping Chen, Junye Chen, Zihao Mo, Zedi Wang, Guanbin Li. RecoverBench: A Systematic Benchmark for Error Recovery in Robotic Manipulation. NeurIPS 2026, Evaluations and Datasets Track (accepted). See [CITATION.cff](CITATION.cff).

## Attribution and maintenance

Developed by Zijia Tang, Ganlong Zhao, Xingping Chen, Junye Chen, Zihao Mo, Zedi Wang, Guanbin Li.
Author affiliations are recorded in [CITATION.cff](CITATION.cff). Correspondence: Guanbin Li (liguanbin@mail.sysu.edu.cn); first-author contact: Zijia Tang (zt81@duke.edu). Code and dataset provenance is documented in
[DATASHEET.md](DATASHEET.md) and [the dataset card](docs/HF_DATASET_CARD.md).
Report reproducibility problems through the [issue tracker](https://github.com/ErrorRecoveryBenchmark/ErrorRecoveryBenchmark/issues).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Vendored dependencies and adapted components retain their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
