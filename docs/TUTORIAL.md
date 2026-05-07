# RecoverBench — User Tutorial

## Summary (English)

This is the end-user tutorial for the RecoverBench data-collection and pipeline workflow. The shell commands work as written. Reviewers who only need the **evaluation protocol** should read [`../EVALUATION.md`](../EVALUATION.md); reviewers who only need the **dataset card** should read [`../DATASHEET.md`](../DATASHEET.md); reviewers tracing the **headline findings** should read [`../INSIGHTS.md`](../INSIGHTS.md).

The sections below cover, in order: (1) environment setup including SpaceMouse on Linux; (2) recovery demonstration collection via SpaceMouse teleoperation; (3) v5 error scene generation; (4) MimicGen augmentation of recovery demonstrations; and (5) data preparation for downstream training. Cross-references to file paths and command lines apply identically to all readers.

---

## 1. Environment Setup

```bash
# 1. Install system dependencies (Linux)
sudo apt-get install -y libhidapi-dev libglfw3-dev

# 2. Create conda environment (one-time setup)
bash setup_env.sh

# 3. Activate environment
conda activate recovery_collect

# 4. Verify
python -c "import mujoco; import robosuite; print('OK')"
```

### SpaceMouse Configuration (Linux)

```bash
sudo tee /etc/udev/rules.d/99-spacemouse.rules << 'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 2. Collecting Recovery Demonstrations (Stage 2)

### Quick Start

```bash
conda activate recovery_collect

# Collect a single subtype
bash run_collection.sh stack grasp_misalignment_D0 8

# Batch collection
bash collect_all.sh --task stack        # Single task
bash collect_all.sh --d0_only           # D0 difficulty only
bash collect_all.sh --resume            # Resume from checkpoint
```

### SpaceMouse Controls

| Action | Effect |
|--------|--------|
| Push laterally | XY plane movement |
| Push/pull longitudinally | Z-axis up/down |
| Twist around axis | Rotation |
| **Hold left button** | **Close gripper** |
| **Right button** | **Discard current demo, switch to next scene** |

### Collection Workflow

1. Script automatically loads an error scene (optional: plays injection animation)
2. Use SpaceMouse to control robot arm to recover from error and complete the task
3. After 10 consecutive frames satisfying `check_success()`, automatic validation (release + lift 10cm)
4. Automatic MimicGen compatibility check (action replay + scene augmentation)
5. Demos passing validation count toward the quota, saved and next scene loaded

### Recovery Strategy Reference

| Error Type | What You Need to Do |
|-----------|---------------------|
| Misaligned/wrong-pose grasp | Release -> Re-align -> Grasp -> Deliver to target |
| Dropped during transport | Move to drop location -> Pick up -> Deliver to target |
| Collision | Back away -> Navigate around -> Re-approach |
| Grasped wrong object | Release -> Find correct object -> Grasp -> Deliver to target |
| Position error | Correct position -> Continue to complete |

---

## 3. MimicGen Augmentation (Stage 3)

```bash
# Augment a single task (target 100 per subtype)
python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
    --task stack --target_per_subtype 100

# Augment a specific subtype
python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
    --task stack --subtype grasp_misalignment_D0 --target_per_subtype 100

# With video recording (for debugging)
python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
    --task stack --target_per_subtype 10 --render
```

Three augmentation modes:
- **3A Scene Augmentation** — Replay recovery actions on different error scenes (object positions vary)
- **3B Cross-degree** — D0 demo -> D1 (add rotation/displacement perturbation)
- **3C Cross-subtype** — Transfer between different error types within the same RBG

---

## 4. Data Output

### Directory Structure

```
error_benchmark/outputs/recovery/
├── demos/{task}/{subtype}/*.npz       # Human demonstrations
├── demos/{task}/manifest.json          # Metadata
└── augmented/{task}/{subtype}/aug_*.npz # Augmented data
```

### NPZ Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `actions` | (N, 7) | Action sequence |
| `states` | (N+1, state_dim) | MuJoCo sim states |
| `eef_positions` | (N+1, 3) | End-effector positions |
| `target_poses` | (N, 4, 4) | OSC controller target poses |
| `gripper_states` | (N+1,) | Gripper open/close degree |
| `obj_{name}` | (N+1, 3) | Object positions |
| `obj_{name}_quat` | (N+1, 4) | Object quaternions (w,x,y,z) |

### View Collection Progress

```bash
python -c "
import json, os
for task in sorted(os.listdir('error_benchmark/outputs/recovery/demos')):
    manifest = f'error_benchmark/outputs/recovery/demos/{task}/manifest.json'
    if os.path.exists(manifest):
        m = json.load(open(manifest))
        success = sum(1 for d in m['demos'] if d.get('success'))
        counted = sum(1 for d in m['demos'] if d.get('counts_toward_target'))
        print(f'{task}: {len(m[\"demos\"])} total, {success} success, {counted} counted')
"
```

---

## 5. Transfer to Cluster After Collection

```bash
scp -r error_benchmark/outputs/recovery/demos/ user@cluster:/path/to/error_recovery/error_benchmark/outputs/recovery/demos/
```

---

## 6. FAQ

| Problem | Solution |
|---------|----------|
| SpaceMouse not responding | Check USB connection `lsusb \| grep 256f`, check udev rules, unplug and replug |
| Render black screen | Confirm `MUJOCO_GL=glfw`, check display driver |
| EnvironmentMismatchError | `pip install mujoco==2.3.2` |
| Controls too sensitive/sluggish | Modify `pos_sensitivity` / `rot_sensitivity` in `recovery_collection.yaml` |
| Demo succeeds but does not count toward quota | Check `quota_reason` in `manifest.json`; use `--no_validation` to skip validation |

---
