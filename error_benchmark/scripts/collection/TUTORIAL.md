# Recovery Demo Collection Tutorial

> **Full Pipeline technical documentation** at [`error_benchmark/docs/recovery_pipeline.md`](../../docs/recovery_pipeline.md)

## Summary (English)

Operational tutorial for collecting human recovery demonstrations in RecoverBench (Stage 2 of the data pipeline). The shell commands and code blocks work as written; the surrounding prose is in English for the collection team. The rest of this document covers, in order: project overview, environment setup (including SpaceMouse on Linux), the SpaceMouse-driven teleoperation workflow, success-validation rules (10-frame `_check_success` hold + action-replay validation), batch collection scripts, and troubleshooting. For the broader RecoverBench framing, see [`../../../README.md`](../../../README.md), [`../../../EVALUATION.md`](../../../EVALUATION.md), and [`../../docs/recovery_pipeline.md`](../../docs/recovery_pipeline.md).

---

## 1. Overview

RecoverBench requires human recovery demonstrations to train recovery policies. Operators use a SpaceMouse to teleoperate a Sawyer robot arm, recovering from injected error scenarios and completing the task.

**Supported tasks are defined by `error_benchmark/configs/task_registry.yaml` in the portable package.**

View which tasks are included in the current package:

```bash
sed -n 's/^  \([A-Za-z0-9_][A-Za-z0-9_]*\):$/\1/p' error_benchmark/configs/task_registry.yaml
```

View how many training scenes each task currently has:

```bash
for task in $(sed -n 's/^  \([A-Za-z0-9_][A-Za-z0-9_]*\):$/\1/p' error_benchmark/configs/task_registry.yaml); do
    count=$(find "error_benchmark/outputs/v5_training/$task/scenes" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
    printf "%-24s %s\n" "$task" "$count"
done
```

Each error scene (injected by the v5 pipeline) represents a specific error state that occurred during task execution. The operator must recover from that state and complete the task.

## 2. Environment Setup

### 2.1 System Dependencies

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install -y libhidapi-dev libglfw3-dev
```

### 2.2 Conda Environment

```bash
bash setup_env.sh
```

This script creates the `recovery_collect` conda environment and installs all dependencies (MuJoCo 2.3.2, robosuite, hidapi, etc.).

### 2.3 SpaceMouse Linux Configuration

Create a udev rule to allow non-root users to access the SpaceMouse:

```bash
# Create the rule file
sudo tee /etc/udev/rules.d/99-spacemouse.rules << 'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"
EOF

# Reload rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2.4 Verify Installation

```bash
conda activate recovery_collect

# Verify core packages
python -c "import mujoco; import robosuite; print('OK')"

# Verify SpaceMouse connection
python -c "import hid; devs = [d for d in hid.enumerate() if d['vendor_id']==0x256f]; print(f'Found {len(devs)} SpaceMouse device(s)'); [print(f'  {d[\"product_string\"]}') for d in devs]"
```

## 3. SpaceMouse Operation Guide

### Control Mapping

| Action | Effect |
|--------|--------|
| Push laterally | End-effector XY plane movement |
| Push/pull longitudinally | Z-axis up/down |
| Twist around axis | End-effector rotation (roll/pitch/yaw) |
| **Hold left button** | **Close gripper** (release button = open gripper) |
| **Right button** | **Discard current demo, load next scene** |

### Sensitivity Adjustment

Default sensitivity is configured in `error_benchmark/configs/recovery_collection.yaml`:
- `pos_sensitivity: 1.0` — Position control sensitivity (robosuite default)
- `rot_sensitivity: 1.0` — Rotation control sensitivity (robosuite default)

If the controls are too sensitive or too sluggish, modify these values (higher values = more sensitive).

## 4. Collection Workflow Details

1. **Start script** — Automatically loads an error scene (robot arm is in an error state)
2. **Render window** — Shows the scene from the agentview perspective
3. **Perform recovery** — Use SpaceMouse to control the robot arm and recover from the error
4. **Task success** — After completing the task, the demo is automatically saved and the next scene is loaded
5. **Timeout** — If not completed within 500 steps, marked as failed; **retries the same scene** (until success or manual skip)
6. **Manual skip** — Press SpaceMouse right button to discard the current demo and switch to a new scene

## 5. Error Types and Recovery Strategy Guide

| RBG | Representative Error | Scene Description | What You Need to Do |
|-----|---------------------|-------------------|---------------------|
| **A** (Re-grasp) | `grasp_misalignment_D0` | Grasp is misaligned, object is incorrectly positioned in gripper | Open gripper -> Adjust position -> Re-grasp -> Transport to target -> Place |
| **B** (Retrieve) | `drop_in_transit_D0` | Object dropped during transport | Move to drop location -> Pick up -> Continue to target -> Place |
| **C** (Retract) | `collision_holding_D0` | Robot arm collided with something | Back away -> Avoid obstacle -> Re-approach target |
| **D** (Redirect) | `wrong_object_D0` | Grasped the wrong object | Release wrong object -> Find correct object -> Grasp -> Transport and place |
| **E** (Realign) | `position_error_D0` | End-effector position error | Correct position -> Continue completing the task |

### Operation Tips

- **Observe first**: After each scene loads, observe the robot arm and object states to understand what went wrong
- **Gentle operation**: SpaceMouse input is scaled, so large movements are not needed
- **Stay still while grasping**: Keep the SpaceMouse still when pressing the left button to close the gripper, to avoid offset during the grasp
- **Align when placing**: Move the object directly above the target before slowly descending to place

## 6. Running Commands

```bash
conda activate recovery_collect
cd /path/to/recovery_collection_portable  # or wherever you scp'd it

# View available tasks in current package
sed -n 's/^  \([A-Za-z0-9_][A-Za-z0-9_]*\):$/\1/p' error_benchmark/configs/task_registry.yaml

# Basic usage
bash run_collection.sh <task> <subtype> [num_demos]

# Examples (specific tasks depend on current package)
bash run_collection.sh pick_place grasp_misalignment_D0 8
bash run_collection.sh stack collision_holding_D0 6
bash run_collection.sh threading position_error_D0 6

# D1 difficulty examples (optional, as needed)
bash run_collection.sh pick_place grasp_misalignment_D1 4
bash run_collection.sh stack drop_in_transit_D1 4
```

### View Available Scenes

```bash
# List all available subtypes and their counts for a task
TASK=pick_place  # Change to a task found in the previous step
for json in error_benchmark/outputs/v5_training/$TASK/scenes/*.json; do
    python -c "import json,sys; d=json.load(open(sys.argv[1])); es=d.get('error_spec',{}); print(es.get('error_name','')+'_'+es.get('degree',''))" "$json"
done | sort | uniq -c | sort -rn
```

## 7. Output and Quality Check

### Save Path

```
outputs/recovery/demos/
├── pick_place/
│   ├── grasp_misalignment_D0/
│   │   ├── recovery_pick_place_grasp_misalignment_D0_0000.npz
│   │   └── ...
│   ├── drop_in_transit_D0/
│   │   └── ...
│   ├── manifest.json   # pick_place demo metadata
│   └── human_demo_test_report.json   # MimicGen compatibility test results
├── stack/
│   ├── grasp_misalignment_D0/
│   │   └── ...
│   ├── collision_holding_D0/
│   │   └── ...
│   ├── manifest.json   # stack demo metadata
│   └── human_demo_test_report.json   # MimicGen compatibility test results
├── coffee/
│   ├── ...
│   ├── manifest.json   # coffee demo metadata
│   └── human_demo_test_report.json   # MimicGen compatibility test results
└── ...
```

### NPZ File Contents

Each demo NPZ contains:
- `actions` — Action sequence (N, action_dim)
- `states` — MuJoCo sim states (N+1, state_dim)
- `eef_positions` — End-effector positions (N+1, 3)
- `gripper_states` — Gripper open/close degree (N+1,)
- `camera_images` — Camera images (one every 4 frames)
- `obj_*` — Object positions

### manifest.json

Records for each demo:
- `demo_id`, `task_name`, `error_name`, `degree`
- `success` — Whether recovery was successfully completed
- `num_steps` — Number of action steps
- `scene_id` — Source error scene
- `collection_validation` — Post-collection MimicGen compatibility check results

Only demos with `success=true` and passing compatibility checks are recommended for subsequent MimicGen augmentation.

### human_demo_test_report.json

Automatically summarizes for each successful human demo:
- `action_replay` — Whether replaying with `states[0] + actions` still succeeds
- `scene_augmentation` — Whether at least one successful augmentation on a same-subtype target scene is possible
- `summary.augmentable` — Whether this demo meets current MimicGen augmentation requirements

### View Collection Progress

```bash
python -c "
import json, os
for task in sorted(os.listdir('outputs/recovery/demos')):
    task_dir = f'outputs/recovery/demos/{task}'
    if not os.path.isdir(task_dir): continue
    manifest = os.path.join(task_dir, 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        success = sum(1 for d in m['demos'] if d.get('success'))
        print(f'{task}: {len(m[\"demos\"])} demos ({success} successful)')
"
```

## 8. After Collection

Transfer collection results back to the cluster via scp:

```bash
scp -r outputs/recovery/demos/ user@cluster:/path/to/release_code/error_benchmark/outputs/recovery/demos/
```

Note: The portable package no longer bundles `robosuite`; instead it is installed locally on the target machine via `setup_env.sh` at a locked commit. `mimicgen` is still included. Lock information is in `ROBOSUITE_VERSION_LOCK.txt`.

Then run MimicGen augmentation on the cluster:
```bash
cd /path/to/release_code
conda activate mimicgen_env
make recovery-augment RECOVERY_TASK=<task>
# Run separately for each collected task
```

## 9. FAQ

### SpaceMouse Not Responding
1. Check USB connection: `lsusb | grep 256f`
2. Check if udev rules are active
3. Verify HID visibility: `python -c "import hid; print([d for d in hid.enumerate() if d['vendor_id']==0x256f])"`
4. May need to unplug and replug the device

### Render Window Black Screen
- Confirm `MUJOCO_GL=glfw` (not `egl`)
- Check display driver: `glxinfo | head -5`
- If using remote desktop, ensure hardware acceleration is available

### EnvironmentMismatchError
MuJoCo version mismatch. Need to install MuJoCo 2.3.2:
```bash
pip install mujoco==2.3.2
```

### Controls Too Sensitive/Sluggish
Edit `pos_sensitivity` and `rot_sensitivity` in `error_benchmark/configs/recovery_collection.yaml`.

### Cannot Find Scenes
Confirm that the task and subtype you specified exist in the training scenes. Use the commands in Section 6 to view available subtypes.
