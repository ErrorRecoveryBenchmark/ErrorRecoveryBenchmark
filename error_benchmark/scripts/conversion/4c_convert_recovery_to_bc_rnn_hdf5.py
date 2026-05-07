#!/usr/bin/env python
"""
Convert recovery demos (NPZ) to robomimic HDF5 format for BC-RNN fine-tuning.

Recovery demos from human teleop or MimicGen augmentation are stored in NPZ format.
BC-RNN training requires robomimic HDF5 format with:
  - Observations: low_dim (robot0_eef_pos/quat/gripper_qpos, object) + rgb images
  - Actions: 7-DoF
  - States: full MuJoCo state

This script:
  1. Loads recovery NPZ files from outputs/recovery/demos/{task}/{subtype}/
  2. Replays states in environment to extract camera images
  3. Builds complete robomimic-format observations
  4. Writes HDF5 file compatible with BC-RNN training

Usage:
  conda activate mimicgen_env
  MUJOCO_GL=egl python 4c_convert_recovery_to_bc_rnn_hdf5.py --task stack --gpu 0

  # Dry run to check available demos
  python 4c_convert_recovery_to_bc_rnn_hdf5.py --task stack --dry_run

  # Only successful demos
  python 4c_convert_recovery_to_bc_rnn_hdf5.py --task stack --only_success
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import yaml

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
RECOVERY_DEMOS_DIR = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "demos"
MIMICGEN_DATASETS_DIR = Path(
    os.environ.get("MIMICGEN_DATASETS_DIR")
    or "${BENCHMARK_DATA}/mimicgen_prepared"
)
OUTPUT_DIR = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "bc_rnn_training"

# Ensure framework is importable
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALL_TASKS = [
    "pick_place",
    "coffee",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
]

# BC-RNN observation keys (from config.json)
BC_RNN_OBS_KEYS_LOW_DIM = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]

BC_RNN_OBS_KEYS_RGB = [
    "agentview_image",
    "robot0_eye_in_hand_image",
]


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_hdf5_attr(attrs, key: str, value: Any) -> None:
    if value is None:
        return
    value = _to_python_scalar(value)
    if isinstance(value, (dict, list, tuple)):
        attrs[key] = json.dumps(value)
    else:
        attrs[key] = value


def find_recovery_demos(task: str, only_success: bool = False) -> List[Dict]:
    """Find all recovery demo NPZ files for a task."""
    task_dir = RECOVERY_DEMOS_DIR / task
    if not task_dir.exists():
        logger.warning(f"No recovery demos directory: {task_dir}")
        return []

    # Load manifest
    manifest_path = task_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning(f"No manifest.json in {task_dir}")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    demos = []
    for entry in manifest.get("demos", []):
        if only_success and not entry.get("success", False):
            continue

        npz_path = task_dir / entry.get("subtype_id", "") / f"{entry['demo_id']}.npz"
        if not npz_path.exists():
            # Try alternative path structure
            npz_path = task_dir / entry.get("subtype_id", "") / f"{entry['demo_id']}.npz"

        if npz_path.exists():
            demos.append({
                "npz_path": npz_path,
                "manifest_entry": entry,
            })
        else:
            logger.warning(f"NPZ not found: {npz_path}")

    return demos


def load_recovery_npz(npz_path: Path) -> Dict:
    """Load a recovery demo NPZ file."""
    data = np.load(str(npz_path), allow_pickle=True)

    result = {
        "actions": data["actions"],
        "states": data.get("states", None),
        "success": bool(data.get("success", False)),
        "num_steps": len(data["actions"]),
    }

    # Try to get camera images if available
    if "camera_images" in data:
        result["camera_images"] = data["camera_images"]

    # Get other metadata
    for key in data.files:
        if key not in {"actions", "states", "camera_images", "success"}:
            result[key] = _to_python_scalar(data[key])

    return result


def get_env_args_from_mimicgen_dataset(task: str) -> str:
    """Get env_args JSON from existing MimicGen dataset."""
    # Map task name to dataset file
    dataset_map = {
        "pick_place": "pick_place_d0.hdf5",
        "coffee": "coffee_d0.hdf5",
        "stack": "stack_d0.hdf5",
        "stack_three": "stack_three_d0.hdf5",
        "threading": "threading_d0.hdf5",
        "three_piece_assembly": "three_piece_assembly_d0.hdf5",
    }

    dataset_file = dataset_map.get(task)
    if not dataset_file:
        raise ValueError(f"Unknown task: {task}")

    dataset_path = MIMICGEN_DATASETS_DIR / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"MimicGen dataset not found: {dataset_path}")

    with h5py.File(str(dataset_path), "r") as f:
        env_args = f["data"].attrs.get("env_args", "")
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")
        return env_args


def replay_demo_for_images(
    env,
    env_wrapper,
    states: np.ndarray,
    actions: np.ndarray,
    camera_resolution: int = 84,
) -> List[Dict]:
    """
    Replay demo states in environment to extract camera images.

    Returns list of observation dicts per timestep.
    """
    obs_sequence = []
    num_steps = len(actions)

    # Reset environment first
    env.reset()

    for t in range(num_steps + 1):  # +1 for final state
        try:
            # Set state
            if states is not None and t < len(states):
                env_wrapper.set_sim_state_flat(states[t])

            # Get observation (includes camera images)
            obs = env.get_observation()

            # Build robomimic-format obs dict
            obs_dict = {
                "robot0_eef_pos": obs.get("robot0_eef_pos", np.zeros(3)),
                "robot0_eef_quat": obs.get("robot0_eef_quat", np.zeros(4)),
                "robot0_gripper_qpos": obs.get("robot0_gripper_qpos", np.zeros(2)),
                "object": obs.get("object-state", obs.get("object", np.zeros(14))),
            }

            # Add camera images
            if "agentview_image" in obs:
                img = obs["agentview_image"]
                if img.shape[0] != camera_resolution or img.shape[1] != camera_resolution:
                    # Resize if needed (simple crop or use cv2)
                    if img.shape[0] >= camera_resolution and img.shape[1] >= camera_resolution:
                        # Center crop
                        h, w = img.shape[:2]
                        start_h = (h - camera_resolution) // 2
                        start_w = (w - camera_resolution) // 2
                        img = img[start_h:start_h+camera_resolution, start_w:start_w+camera_resolution]
                obs_dict["agentview_image"] = img.astype(np.uint8)

            if "robot0_eye_in_hand_image" in obs:
                img = obs["robot0_eye_in_hand_image"]
                if img.shape[0] != camera_resolution or img.shape[1] != camera_resolution:
                    if img.shape[0] >= camera_resolution and img.shape[1] >= camera_resolution:
                        h, w = img.shape[:2]
                        start_h = (h - camera_resolution) // 2
                        start_w = (w - camera_resolution) // 2
                        img = img[start_h:start_h+camera_resolution, start_w:start_w+camera_resolution]
                obs_dict["robot0_eye_in_hand_image"] = img.astype(np.uint8)

            obs_sequence.append(obs_dict)

        except Exception as e:
            logger.warning(f"  Step {t}: Failed to extract obs: {e}")
            # Use neutral observation as fallback
            obs_sequence.append({
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
                "object": np.zeros(14),
            })

    return obs_sequence


def write_recovery_hdf5(
    output_path: Path,
    demos: List[Dict],
    env_args: str,
    task: str,
    camera_resolution: int = 84,
) -> Dict:
    """Write recovery demos data to robomimic HDF5 format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_demos": len(demos),
        "successful_demos": 0,
        "failed_demos": 0,
        "total_samples": 0,
        "by_subtype": {},
    }

    with h5py.File(str(output_path), "w") as f:
        data_group = f.create_group("data")
        data_group.attrs["env_args"] = env_args

        scene_id_map = {}

        for idx, demo_data in enumerate(demos):
            demo_key = f"demo_{idx}"
            manifest_entry = demo_data["manifest_entry"]
            npz_data = demo_data["npz_data"]

            subtype_id = manifest_entry.get("subtype_id", "unknown")
            scene_id = manifest_entry.get("scene_id", "")

            # Track stats
            if subtype_id not in stats["by_subtype"]:
                stats["by_subtype"][subtype_id] = {"count": 0, "success": 0, "samples": 0}
            stats["by_subtype"][subtype_id]["count"] += 1

            if npz_data["success"]:
                stats["successful_demos"] += 1
                stats["by_subtype"][subtype_id]["success"] += 1
            else:
                stats["failed_demos"] += 1

            # Create demo group
            group = data_group.create_group(demo_key)
            group.attrs["num_samples"] = npz_data["num_steps"]
            group.attrs["scene_id"] = scene_id
            group.attrs["subtype_id"] = subtype_id
            group.attrs["success"] = npz_data["success"]
            group.attrs["error_name"] = manifest_entry.get("error_name", "")
            group.attrs["degree"] = manifest_entry.get("degree", "")
            group.attrs["rbg"] = manifest_entry.get("rbg", "")
            group.attrs["demo_origin"] = "recovery_teleop"

            # Write actions
            group.create_dataset("actions", data=npz_data["actions"].astype(np.float64))

            # Write states if available
            if npz_data["states"] is not None:
                # Use states[:-1] to match actions length
                states = npz_data["states"]
                if len(states) > npz_data["num_steps"]:
                    states = states[:npz_data["num_steps"]]
                group.create_dataset("states", data=states.astype(np.float64))

            # Write observations (from replay or NPZ)
            obs_group = group.create_group("obs")

            if demo_data.get("obs_sequence"):
                # Use replay-extracted observations
                obs_seq = demo_data["obs_sequence"]
                for key in BC_RNN_OBS_KEYS_LOW_DIM + BC_RNN_OBS_KEYS_RGB:
                    if key in obs_seq[0]:
                        arrays = [obs[key] for obs in obs_seq[:npz_data["num_steps"]]]
                        obs_group.create_dataset(key, data=np.stack(arrays))
            elif npz_data.get("camera_images"):
                # Use images from NPZ directly
                images = npz_data["camera_images"]
                # camera_images shape: (T//4, H, W, 3) at 5Hz, need to upsample to match actions
                num_images = len(images)
                num_actions = npz_data["num_steps"]
                # Upsample by repeating
                upsample_factor = num_actions // num_images + 1
                upsampled_images = np.repeat(images, upsample_factor, axis=0)[:num_actions]
                obs_group.create_dataset("agentview_image", data=upsampled_images.astype(np.uint8))

                # Low-dim obs from states (best effort)
                # This is incomplete - proper approach needs replay
                logger.warning(f"  {demo_key}: Using fallback obs (no replay)")

            stats["total_samples"] += npz_data["num_steps"]
            scene_id_map[demo_key] = scene_id

        data_group.attrs["total"] = stats["total_samples"]
        data_group.attrs["scene_id_map"] = json.dumps(scene_id_map)

    return stats


# ═══════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════

def process_task(task: str, args) -> None:
    """Convert recovery demos for a single task."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task}")
    logger.info(f"{'='*60}")

    # Find demos
    demo_entries = find_recovery_demos(task, only_success=args.only_success)
    if not demo_entries:
        logger.warning(f"No recovery demos found for {task}")
        return

    logger.info(f"Found {len(demo_entries)} recovery demo files")

    if args.dry_run:
        logger.info("\nDry run - showing demo info:")
        for entry in demo_entries[:10]:
            manifest = entry["manifest_entry"]
            logger.info(f"  {manifest.get('demo_id', 'unknown')}: "
                       f"{manifest.get('subtype_id', 'unknown')}, "
                       f"success={manifest.get('success', False)}, "
                       f"steps={manifest.get('num_steps', 0)}")
        if len(demo_entries) > 10:
            logger.info(f"  ... and {len(demo_entries) - 10} more")
        return

    # Get env_args from MimicGen dataset
    try:
        env_args = get_env_args_from_mimicgen_dataset(task)
        logger.info(f"Loaded env_args from MimicGen dataset")
    except Exception as e:
        logger.warning(f"Failed to load env_args: {e}")
        env_args = ""

    # Load NPZ files
    demos_to_write = []
    for entry in demo_entries:
        npz_path = entry["npz_path"]
        manifest = entry["manifest_entry"]

        try:
            npz_data = load_recovery_npz(npz_path)
            demos_to_write.append({
                "npz_path": npz_path,
                "manifest_entry": manifest,
                "npz_data": npz_data,
                "obs_sequence": None,  # Will be filled if replay enabled
            })
            logger.info(f"  Loaded {npz_path.name}: {npz_data['num_steps']} steps, success={npz_data['success']}")
        except Exception as e:
            logger.error(f"  Failed to load {npz_path}: {e}")

    if not demos_to_write:
        logger.warning(f"No valid demos loaded for {task}")
        return

    # Optionally replay to extract images
    if args.replay_images:
        logger.info("\nReplaying demos to extract camera images...")
        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
        from error_benchmark.framework.env_wrapper import EnvWrapper

        task_reg = load_task_registry(task)
        task_config_path = os.path.join(str(PROJECT_ROOT), task_reg["task_config"])
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)

        os.environ["MUJOCO_GL"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        env = create_env(
            task_config,
            task_reg["dataset_path"],
            enable_camera=True,
            camera_resolution=args.camera_resolution,
        )
        env_wrapper = EnvWrapper(env, task_config)

        for demo_data in demos_to_write:
            try:
                obs_seq = replay_demo_for_images(
                    env, env_wrapper,
                    demo_data["npz_data"]["states"],
                    demo_data["npz_data"]["actions"],
                    args.camera_resolution,
                )
                demo_data["obs_sequence"] = obs_seq
                logger.info(f"  Extracted obs for {demo_data['manifest_entry']['demo_id']}")
            except Exception as e:
                logger.error(f"  Failed to extract obs: {e}")

        env.close()

    # Write HDF5
    output_path = OUTPUT_DIR / task / f"recovery_{task}_d{'0' if 'd0' in args.task else '1' if 'd1' in args.task else 'all'}.hdf5"
    if args.only_success:
        output_path = OUTPUT_DIR / task / f"recovery_{task}_success_only.hdf5"

    logger.info(f"\nWriting HDF5: {output_path}")
    stats = write_recovery_hdf5(
        output_path,
        demos_to_write,
        env_args,
        task,
        args.camera_resolution,
    )

    logger.info(f"\nConversion stats:")
    logger.info(f"  Total demos: {stats['total_demos']}")
    logger.info(f"  Successful: {stats['successful_demos']}")
    logger.info(f"  Failed: {stats['failed_demos']}")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  By subtype:")
    for subtype, sub_stats in sorted(stats["by_subtype"].items()):
        logger.info(f"    {subtype}: {sub_stats['count']} demos, {sub_stats['success']} success, {sub_stats['samples']} samples")

    # Save conversion report
    report_path = OUTPUT_DIR / task / "conversion_report.json"
    report = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "output_file": str(output_path),
        "stats": stats,
        "args": {
            "only_success": args.only_success,
            "replay_images": args.replay_images,
            "camera_resolution": args.camera_resolution,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert recovery demos to BC-RNN HDF5 format"
    )
    parser.add_argument("--task", type=str, required=True, choices=ALL_TASKS)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for replay")
    parser.add_argument("--only_success", action="store_true",
                        help="Only convert successful demos")
    parser.add_argument("--replay_images", action="store_true",
                        help="Replay states to extract camera images")
    parser.add_argument("--camera_resolution", type=int, default=84,
                        help="Camera image resolution (default 84 for BC-RNN)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show demo info without conversion")
    args = parser.parse_args()

    process_task(args.task, args)


if __name__ == "__main__":
    main()