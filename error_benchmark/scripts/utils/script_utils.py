"""
Shared utilities for benchmark scripts.

Extracted from 1_generate_scenes.py, 1c_generate_from_policy.py, and
5_baseline_accuracy.py to eliminate ~210 lines of duplicated code.
"""

import json
import logging
import os

import h5py
import numpy as np
import robosuite
import yaml


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Import mimicgen to register MimicGen _D0 environments with robosuite.
# This ensures envs like Coffee_D0, Stack_D0 etc. are available via robosuite.make()
# and return the correct observation dimensions (matching training data).
try:
    import mimicgen
except ImportError:
    pass

# Project root (used by load_task_registry for config resolution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_env(task_config: dict, dataset_path: str, enable_camera: bool = False,
               camera_resolution: int = 256, has_renderer: bool = False):
    """Create a robosuite environment from HDF5 dataset metadata.

    Reads ``env_args`` from the dataset's HDF5 attributes and creates the
    environment with the correct controller configuration.  Tries the original
    MimicGen _D0 env name first, then falls back to the base env name.

    Args:
        task_config: Task configuration dict (currently unused but kept for
            interface consistency with EnvWrapper).
        dataset_path: Path to the HDF5 demo dataset.
        enable_camera: If True, enable offscreen rendering for camera
            observations (required for VLA / image-based policies).
        camera_resolution: Camera image height/width (default 256 for VLA,
            84 for BC-RNN).

    Returns:
        A robosuite environment instance.

    Raises:
        ValueError: If *dataset_path* does not exist.
        RuntimeError: If the environment cannot be created.
    """
    if not dataset_path or not os.path.isfile(dataset_path):
        raise ValueError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
        env_kwargs = env_args.get('env_kwargs', {})

    # Remove incompatible kwargs
    for bad_key in ["z_rotation", "replacement_probs", "task_id"]:
        env_kwargs.pop(bad_key, None)

    # Rendering configuration
    env_kwargs["has_renderer"] = has_renderer
    if enable_camera:
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
        env_kwargs["camera_heights"] = camera_resolution
        env_kwargs["camera_widths"] = camera_resolution
    else:
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["use_camera_obs"] = False

    # Extract env_name and robots
    env_name = env_args.get('env_name', '')
    robots = env_kwargs.pop("robots", None)

    # Cross-check: HDF5 env_name must be compatible with task_config env_name
    expected_env_name = task_config.get('env_args', {}).get('env_name', '')
    if expected_env_name and env_name:
        def _base_env_name(n):
            parts = n.split('_')
            if parts[-1] in ('D0', 'D1', 'd0', 'd1'):
                return '_'.join(parts[:-1])
            return n

        if _base_env_name(env_name) != _base_env_name(expected_env_name):
            raise ValueError(
                f"Environment name mismatch: HDF5 dataset contains env '{env_name}' "
                f"but task_config expects '{expected_env_name}'. "
                f"Check that dataset_path matches the task."
            )

    # Try using MimicGen _D0 env directly (preserves full observables for BC-RNN).
    # Only fall back to base env if _D0 variant is not registered.
    base_env_name = env_name
    if '_' in env_name and env_name.split('_')[-1] in ['D0', 'D1', 'd0', 'd1']:
        base_env_name = '_'.join(env_name.split('_')[:-1])

    env = None
    last_error = None
    for try_name in [env_name, base_env_name]:
        try:
            env = robosuite.make(try_name, robots=robots, **env_kwargs)
            break
        except Exception as e:
            last_error = e
            continue

    if env is None:
        raise RuntimeError(
            f"Could not create env with name '{env_name}' or '{base_env_name}'. "
            f"Last error: {last_error}"
        ) from last_error

    # Apply per-task camera overrides from task_config
    camera_overrides = task_config.get("camera_overrides", {})
    if camera_overrides and hasattr(env, "sim"):
        from robosuite.utils.mjcf_utils import string_to_array
        for cam_name, cam_cfg in camera_overrides.items():
            try:
                cam_id = env.sim.model.camera_name2id(cam_name)
                if "pos" in cam_cfg:
                    env.sim.model.cam_pos[cam_id] = string_to_array(cam_cfg["pos"])
                if "quat" in cam_cfg:
                    env.sim.model.cam_quat[cam_id] = string_to_array(cam_cfg["quat"])
            except Exception:
                pass

    return env


def _kill_3dconnexion_driver():
    """Kill 3DConnexion user-space driver processes on macOS.

    The 3DConnexion driver grabs exclusive HID access, preventing hidapi
    from reading the SpaceMouse directly. This kills the user-space
    processes; the DriverKit system extension remains and is harmless.
    """
    import subprocess
    import sys

    if sys.platform != 'darwin':
        return

    targets = [
        '3DxNLServer', '3DxRadialMenu', '3DxVirtualNumpad',
        '3DconnexionHelper', '3Dconnexion Home',
    ]
    for name in targets:
        subprocess.run(['killall', name], capture_output=True)


def detect_spacemouse_product_id() -> int:
    """Auto-detect the product_id of a connected SpaceMouse.

    Enumerates HID devices from 3Dconnexion (vendor 0x256f), filters out
    virtual devices created by the 3DConnexion driver, and returns the
    product_id of the real hardware device.

    If the device cannot be opened (driver is holding it), automatically
    kills the 3DConnexion user-space driver and retries.

    Returns:
        int: product_id of the detected SpaceMouse.

    Raises:
        RuntimeError: If no real SpaceMouse hardware is found.
    """
    import hid

    VENDOR_3DCONNEXION = 0x256f
    VIRTUAL_PRODUCT_IDS = {0xc671, 0xc672}

    def _find_candidates():
        candidates = {}
        for d in hid.enumerate():
            if d['vendor_id'] != VENDOR_3DCONNEXION:
                continue
            pid = d['product_id']
            if pid in VIRTUAL_PRODUCT_IDS:
                continue
            if pid not in candidates:
                candidates[pid] = d['product_string']
        return candidates

    candidates = _find_candidates()
    if not candidates:
        raise RuntimeError(
            "No SpaceMouse hardware detected. Check USB/BT connection and "
            "3DConnexion driver. Run hid.enumerate() to debug.")

    pid, name = next(iter(candidates.items()))

    # Verify we can actually open the device; if not, kill the driver and retry
    import time
    device = hid.device()
    try:
        device.open(VENDOR_3DCONNEXION, pid)
        device.close()
    except OSError:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "SpaceMouse occupied by 3DConnexion driver, killing driver processes...")
        _kill_3dconnexion_driver()

        # Wait for the driver to fully release the HID handle
        for attempt in range(6):
            time.sleep(0.5)
            candidates = _find_candidates()
            if not candidates:
                continue
            pid, name = next(iter(candidates.items()))
            try:
                device = hid.device()
                device.open(VENDOR_3DCONNEXION, pid)
                device.close()
                logger.info("SpaceMouse available after %.1fs", (attempt + 1) * 0.5)
                return pid
            except OSError:
                continue

        raise RuntimeError(
            "SpaceMouse still not accessible after killing 3DConnexion driver. "
            "Try unplugging and re-plugging the device.")

    return pid


def load_task_registry(task_name: str) -> dict:
    """Load task-specific paths from task_registry.yaml.

    Args:
        task_name: Task key (e.g. ``'coffee'``, ``'stack'``, ``'threading'``).

    Returns:
        Dict with ``task_config``, ``dataset_path``, ``output_subdir``, etc.

    Raises:
        ValueError: If the task name is not found in the registry.
    """
    registry_path = os.path.join(PROJECT_ROOT, 'error_benchmark', 'configs', 'task_registry.yaml')
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    tasks = registry.get('tasks', {})
    if task_name not in tasks:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(tasks.keys())}")
    info = tasks[task_name]
    if not info.get('dataset_path'):
        raise ValueError(
            f"Task '{task_name}' has no dataset_path in task_registry.yaml. "
            f"Each task must specify its own dataset_path."
        )
    # Resolve relative dataset_path to absolute
    dp = info.get('dataset_path', '')
    if dp and not os.path.isabs(dp):
        info['dataset_path'] = os.path.join(PROJECT_ROOT, dp)
    return info


def load_error_scenes(scenes_dir, subtype_id: str) -> list:
    """Load error scene metadata for a given subtype from v5 output.

    Scans ``scenes_dir`` for JSON scene files, matches by subtype_id, and
    verifies the companion NPZ file exists. Returns scenes sorted by filename.

    Args:
        scenes_dir: Path to the directory containing scene JSON + NPZ files.
        subtype_id: Subtype ID to filter by (e.g., 'grasp_misalignment_D0').

    Returns:
        List of scene dicts, each with an added ``_npz_path`` key.
    """
    from pathlib import Path
    scenes_dir = Path(scenes_dir)
    scenes = []
    if not scenes_dir.exists():
        return scenes

    for scene_file in sorted(scenes_dir.glob("*.json")):
        try:
            with open(scene_file) as f:
                scene = json.load(f)
            scene_subtype = scene.get('labels', {}).get('subtype_id', '')
            if not scene_subtype:
                error_name = scene.get('error_spec', {}).get('error_name', '')
                degree = scene.get('error_spec', {}).get('degree', '')
                scene_subtype = f"{error_name}_{degree}" if error_name and degree else ''
            if scene_subtype != subtype_id:
                continue
            npz_path = scenes_dir / f"{scene['scene_id']}.npz"
            if npz_path.exists():
                scene['_npz_path'] = str(npz_path)
                scenes.append(scene)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load scene {scene_file}: {e}")

    return scenes


def validate_trajectories_for_task(trajectories, task_name: str):
    """Validate that all trajectories belong to the expected task.

    Args:
        trajectories: List of CleanTrajectory objects.
        task_name: Expected task name (from --task argument).

    Raises:
        ValueError: If any trajectory has a mismatched task_name.
    """
    logger = logging.getLogger(__name__)
    for traj in trajectories:
        if not traj.task_name:
            logger.warning(
                f"Trajectory {traj.trajectory_id} has no task_name "
                f"(legacy data?) — skipping validation"
            )
            continue
        if traj.task_name != task_name:
            raise ValueError(
                f"Trajectory-task mismatch: {traj.trajectory_id} has "
                f"task_name='{traj.task_name}' but expected '{task_name}'"
            )
