# Error Recovery Benchmark

A robotics benchmark for collecting and augmenting **human recovery demonstrations** in MuJoCo-simulated manipulation tasks. A human operator teleoperates a Sawyer robot arm using a SpaceMouse, recovering from injected error states (e.g., dropped objects, misaligned grasps, collisions) and completing the original task.

## Overview

The benchmark pipeline has three stages:

1. **Error scene generation** (v5 pipeline) — inject errors into clean trajectories to produce error scenes
2. **Recovery demo collection** — human teleoperates from error scenes, producing recovery demonstrations
3. **MimicGen augmentation** — augment human demos via scene-configuration warping, cross-degree, and cross-subtype transfer

### Error Taxonomy

12 Error Skills × 2 Degrees (D0/D1) = 24 subtypes, grouped into 5 Recovery Behavior Groups (RBGs):

| RBG | Recovery Primitive | Error Skills |
|-----|-------------------|-------------|
| A (Re-grasp) | retract → re_orient → re_grasp → re_lift → re_transport → re_place | grasp_misalignment, grasp_wrong_pose |
| B (Retrieve) | navigate_to_object → re_grasp → re_lift → re_transport → re_place | drop_in_transit, drop_at_wrong_place |
| C (Retract) | retract → navigate_to_object | collision_holding/empty/eef_object/self |
| D (Redirect) | release → navigate_to_object → re_grasp → re_lift → re_transport → re_place | wrong_object |
| E (Realign) | correct_position → resume_task | trajectory_regression, stuck_no_progress, position_error |

### Tasks

6 manipulation tasks across 3 tiers:
- **Tier 1** (full coverage): pick_place, stack
- **Tier 2** (core): coffee, threading
- **Tier 3** (minimal, transfer from tier 1/2): stack_three, three_piece_assembly

## Quick Start

### Prerequisites

- Python 3.8+
- MuJoCo 2.3.2
- Conda

### Environment Setup

Retrieving notices: - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - done
═══════════════════════════════════════════════════════
 Recovery Collection Environment Setup
═══════════════════════════════════════════════════════
Creating conda env: recovery_collect (Python 3.10)...
Retrieving notices: - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - failed

### Set Environment Variables



### Download Data

The full dataset (∼30 GB) includes seed demos, prepared MimicGen datasets, error scenes, and recovery demonstrations.

<!-- TODO: Add Zenodo/HuggingFace download links -->



See [examples/](examples/) for small sample data that lets you test the pipeline without the full dataset.

## Usage

### V5 Error Scene Pipeline

v5 Step 0: Collecting clean trajectories...
python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py 	--config error_benchmark/configs/benchmark_v5.yaml --task pick_place --label_phases
v5 Smoke Test: 2 demos, 5 max injections...
python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py 	--config error_benchmark/configs/benchmark_v5.yaml --task pick_place --num_demos 2 	--output_dir error_benchmark/outputs/v5_smoke/clean_trajectories

### Recovery Demo Collection

═══════════════════════════════════════════════════════
 Recovery Demo Collection
 Task:     pick_place
 Subtype:  grasp_misalignment_D0
 Demos:    8
 Scenes:   2310 available
═══════════════════════════════════════════════════════

Controls:
  SpaceMouse push  -> move end-effector (XYZ)
  SpaceMouse twist -> rotate end-effector
  Left button hold -> close gripper
  Right button     -> discard demo, load next scene

Recovery Stage 1: Collecting demos for stack...
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python error_benchmark/scripts/collection/2_collect_recovery_demos.py 	--task stack --all_subtypes --config error_benchmark/configs/recovery_collection.yaml

### Recovery Augmentation & Training Data

Recovery Stage 3: Augmenting demos for stack...
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py 	--task stack --config error_benchmark/configs/recovery_collection.yaml --target_per_subtype 100
Recovery Stage 4A: Generating MCM data for stack...
python error_benchmark/scripts/conversion/4a_generate_mcm_training_data.py 	--task stack --config error_benchmark/configs/recovery_collection.yaml
Recovery Stage 4B: Generating diffusion data for stack...
python error_benchmark/scripts/conversion/4b_generate_diffusion_training_data.py 	--task stack --config error_benchmark/configs/recovery_collection.yaml

## Architecture

### Core Framework (`error_benchmark/framework/`)

- `core.py` — Central dataclasses: ErrorSpec, ErrorScene, CleanTrajectory, etc.
- `env_wrapper.py` — Abstracts robosuite internals; all modules interact through this wrapper
- `error_taxonomy_v5.py` — ErrorSkillName enum, ErrorDegree enum, skill definitions
- `error_skills/` — One module per error skill (e01–e12)
- `recovery_types.py` — RecoveryDemo, RecoverySubtask, subtype-to-RBG mapping
- `recovery_segmenter.py` — Segments recovery demos into MimicGen-compatible subtask sequences
- `recovery_mimicgen.py` — Bridges recovery demos to MimicGen augmentation

### Pipeline Modules

- `clean_trajectory_collector.py` — Collects clean trajectories from HDF5 demos
- `opportunity_scanner.py` — Offline: replays each frame × each skill to build opportunity map
- `quota_scheduler.py` — Selects (frame, skill, degree) tuples to fill quotas
- `context_replay.py` — InjectionEngine: replays to injection frame, calls inject() + validate()

### Vendored Dependencies

`shared/mimicgen_workspace/` contains patched forks of:
- **robosuite** (commit c848ca84) — MuJoCo-based robot learning framework
- **mimicgen** — Demonstration augmentation system
- **robosuite-task-zoo** — Custom task environments

See [ROBOSUITE_VERSION_LOCK.txt](ROBOSUITE_VERSION_LOCK.txt) for version pinning details.

## Data Format

### Error Scenes (`*.npz` + `*.json`)

- `sim_state`: pre-injection MuJoCo state
- `post_sim_state`: post-injection stable state
- Quaternions in (w,x,y,z) order
- JSON metadata: error skill, degree, injection frame, dataset path

### Recovery Demos (`*.npz`)

- `states`: MuJoCo state sequence
- `actions`: robot action sequence
- `recovery_subtypes`: recovery behavior labels per subtask

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
