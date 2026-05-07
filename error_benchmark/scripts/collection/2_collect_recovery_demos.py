#!/usr/bin/env python
"""
Stage 1: Recovery Demo Collection via Teleoperation

Loads error scenes (NPZ) from the v5 pipeline, presents them to a human
operator via SpaceMouse/keyboard, and records recovery demonstrations.

Usage:
    # Collect recovery demos for stack task, collision_holding_D0 subtype
    python error_benchmark/scripts/collection/2_collect_recovery_demos.py \
        --task stack \
        --subtype collision_holding_D0 \
        --num_demos 5

    # Collect all subtypes for a task using the configured collection mode
    python error_benchmark/scripts/collection/2_collect_recovery_demos.py \
        --task stack \
        --all_subtypes

    # Resume collection from a previous session
    python error_benchmark/scripts/collection/2_collect_recovery_demos.py \
        --task stack \
        --resume

Output:
    {recovery_demos_dir}/{task}/{subtype_id}/
        demo_*.npz   (arrays: actions, states, camera_images, eef_positions, gripper_states, obj_*)
        manifest.json (metadata for all demos)
"""

import argparse
import json
import logging
import sys
import time
import warnings
from collections import deque
import cv2
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress noisy robosuite warnings
warnings.filterwarnings("ignore", message=".*private macro file.*")
warnings.filterwarnings("ignore", message=".*robosuite task zoo.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, RecoveryCollectionStatus, SUBTYPE_TO_RBG,
)
from error_benchmark.framework.recovery_segmenter import RecoverySegmenter
from error_benchmark.framework.injection_replay import InjectionReplay
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.framework.recovery_collection_validation import (
    DEFAULT_REPORT_FILENAME,
    RecoveryDemoValidator,
    dedupe_demo_entries,
    determine_counts_toward_target,
    get_next_demo_attempt_index,
    load_validation_records,
    log_validation_result,
    normalize_quota_rule,
    recompute_collected_counts,
    recompute_collected_scene_ids,
    save_manifest,
    save_validation_report,
    tag_npz_validation,
    upsert_demo_entry,
    upsert_validation_record,
)
from error_benchmark.framework.mimicgen_source_validation import (
    get_stack_source_validation_config,
    settle_scene_for_source_collection,
)
from error_benchmark.scripts.utils.script_utils import (
    NumpyEncoder, create_env, load_task_registry, load_error_scenes,
)
from robosuite.utils.input_utils import input2action


# load_error_scenes imported from script_utils


def create_teleop_device(device_type: str, teleop_config: dict):
    """Create teleoperation device interface.

    Args:
        device_type: "spacemouse" or "keyboard".
        teleop_config: Dict with pos_sensitivity, rot_sensitivity, etc.

    Returns:
        A robosuite Device instance.
    """
    pos_sens = teleop_config.get('pos_sensitivity', 1.0)
    rot_sens = teleop_config.get('rot_sensitivity', 1.0)

    if device_type == "spacemouse":
        try:
            from robosuite.devices import SpaceMouse
            from error_benchmark.scripts.utils.script_utils import detect_spacemouse_product_id
            product_id = detect_spacemouse_product_id()
            logging.getLogger(__name__).info(
                f"Detected SpaceMouse (product_id=0x{product_id:04x})")
            device = SpaceMouse(
                product_id=product_id,
                pos_sensitivity=pos_sens,
                rot_sensitivity=rot_sens,
            )
            return device
        except ImportError:
            logging.getLogger(__name__).warning(
                "SpaceMouse not available, falling back to keyboard")
            device_type = "keyboard"

    if device_type == "keyboard":
        try:
            from robosuite.devices import Keyboard
            device = Keyboard(
                pos_sensitivity=pos_sens,
                rot_sensitivity=rot_sens,
            )
            return device
        except ImportError:
            pass

    raise RuntimeError(f"No teleoperation device available (tried: {device_type})")


def _shutdown_device(device):
    """Gracefully shut down the SpaceMouse background thread."""
    if device is None:
        return
    try:
        device._enabled = False
        if hasattr(device, 'device'):
            device.device.close()
        if hasattr(device, 'thread'):
            device.thread.join(timeout=1.0)
    except Exception:
        pass


def build_manifest_entry(
    demo: RecoveryDemo,
    *,
    quota_rule: str,
    counts_toward_target: bool | None,
    quota_reason: str,
    npz_path: Path | None = None,
    validation_record: dict | None = None,
) -> dict:
    """Build a manifest entry with collection-quota annotations."""

    entry = {
        **demo.to_dict(),
        "teleop_success": bool(demo.success),
        "counts_toward_target": counts_toward_target,
        "quota_rule": quota_rule,
        "quota_reason": quota_reason,
    }
    if npz_path is not None:
        entry["npz_path"] = str(npz_path)
    if validation_record is not None:
        entry["collection_validation"] = validation_record
    return entry


def discover_scene_subtypes(scenes_dir: Path) -> list[str]:
    """Return subtype_ids that actually exist in the training scene pool."""

    meta_path = scenes_dir.parent / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            subtypes = sorted((meta.get("subtypes") or {}).keys())
            if subtypes:
                return subtypes
        except Exception:
            pass

    subtypes = set()
    for scene_file in sorted(scenes_dir.glob("*.json")):
        try:
            with open(scene_file) as f:
                scene = json.load(f)
            subtype_id = scene.get("labels", {}).get("subtype_id", "")
            if not subtype_id:
                error_name = scene.get("error_spec", {}).get("error_name", "")
                degree = scene.get("error_spec", {}).get("degree", "")
                if error_name and degree:
                    subtype_id = f"{error_name}_{degree}"
            if subtype_id:
                subtypes.add(subtype_id)
        except Exception:
            continue
    return sorted(subtypes)


def refresh_collection_status(
    status: RecoveryCollectionStatus,
    demos: list,
    *,
    target_counts: dict,
    quota_rule: str,
    collection_mode: str,
    target_scene_ids: dict | None = None,
) -> RecoveryCollectionStatus:
    """Refresh status counters from manifest entries for the active collection mode."""

    status.collection_mode = collection_mode
    status.target_demos = dict(target_counts)
    if collection_mode == "per_scene":
        target_scene_ids = target_scene_ids or {}
        status.target_scene_ids = {
            subtype_id: sorted({str(scene_id) for scene_id in scene_ids})
            for subtype_id, scene_ids in target_scene_ids.items()
        }
        status.collected_scene_ids = recompute_collected_scene_ids(
            demos,
            status.target_scene_ids,
            quota_rule=quota_rule,
        )
        status.collected_demos = {
            subtype_id: len(scene_ids)
            for subtype_id, scene_ids in status.collected_scene_ids.items()
        }
    else:
        status.target_scene_ids = {}
        status.collected_scene_ids = {}
        status.collected_demos = recompute_collected_counts(
            demos,
            target_counts,
            quota_rule=quota_rule,
        )
    return status


def collect_single_demo(
    env_wrapper,
    scene_data: dict,
    task_config: dict,
    teleop_config: dict,
    segmenter: RecoverySegmenter,
    task_name: str,
    error_name: str,
    degree: str,
    demo_idx: int,
    camera_name: str = "agentview",
    camera_resolution: int = 256,
    enable_rendering: bool = True,
    dual_view: bool = False,
    device=None,
    injection_replay: InjectionReplay = None,
    clean_traj_dir: Path = None,
    replay_config: dict = None,
) -> RecoveryDemo:
    """
    Collect a single recovery demo from human teleoperation.

    1. Load error scene state from NPZ
    2. Start human teleoperation
    3. Record actions, states, camera images
    4. Check success and segment

    Returns:
        RecoveryDemo (success may be True or False)
    """
    logger = logging.getLogger(__name__)
    subtype_id = f"{error_name}_{degree}"
    max_steps = teleop_config.get('max_episode_steps', 500)
    control_freq = teleop_config.get('control_freq', 20)

    # Load error scene state
    npz_path = scene_data.get('_npz_path', '')
    scene_id = scene_data.get('scene_id', '')

    # --- Injection animation replay (optional) ---
    # If replay is enabled, first play context + injection animation,
    # after which the environment is in post-injection state (equivalent to direct load)
    replay_cfg = replay_config or {}
    replay_enabled = replay_cfg.get('enabled', False)
    env = env_wrapper._env

    # Pick the best available side camera for dual_view
    _side_cam = "sideview"
    if dual_view:
        _available_cams = [
            env.sim.model.camera_id2name(i)
            for i in range(env.sim.model.ncam)
        ]
        if "sideview" not in _available_cams:
            _side_cam = "frontview" if "frontview" in _available_cams else "birdview"
            logger.info(f"  dual_view: 'sideview' not available, using '{_side_cam}'")

    def _make_render_fn():
        """Build a render callback for the current rendering mode."""
        if not enable_rendering:
            return None
        if dual_view:
            _H = int(camera_resolution * 1.5)
            _half = _H // 2
            def _multi_render():
                try:
                    _agent = env_wrapper.get_camera_obs("agentview", (_H, _H))
                    _side = env_wrapper.get_camera_obs(_side_cam, (_half, _half))
                    _wrist = env_wrapper.get_camera_obs("robot0_eye_in_hand", (_half, _half))
                    _right = np.concatenate([_side, _wrist], axis=0)
                    _combined = np.concatenate([_agent, _right], axis=1)
                    cv2.imshow("Recovery Collection [agent | side+wrist]",
                               _combined[:, :, ::-1])
                    cv2.waitKey(1)
                except Exception:
                    if hasattr(env, 'renderer') and env.has_renderer:
                        env.render()
            return _multi_render
        if hasattr(env, 'renderer') and env.has_renderer:
            return lambda: env.render()
        return None

    if replay_enabled and injection_replay is not None and clean_traj_dir is not None:
        replay_render = _make_render_fn()

        ok = injection_replay.replay_animation(
            scene_data=scene_data,
            scene_npz_path=npz_path,
            clean_traj_dir=clean_traj_dir,
            render_fn=replay_render,
            speed=replay_cfg.get('speed', 1.0),
            context_frames=replay_cfg.get('context_frames', 50),
            pause_after=replay_cfg.get('pause_after', 1.5),
        )
        if not ok:
            logger.warning(f"  Replay failed for {scene_id}, falling back to direct load")
            npz_data = np.load(npz_path)
            if 'post_sim_state' in npz_data:
                env_wrapper.set_sim_state_flat(npz_data['post_sim_state'])
            else:
                env_wrapper.set_sim_state_flat(npz_data['sim_state'])
            env_wrapper.forward()
    else:
        # Case 1: Directly load stable post-injection state (for training/validation)
        npz_data = np.load(npz_path)
        if 'post_sim_state' in npz_data:
            env_wrapper.set_sim_state_flat(npz_data['post_sim_state'])
        else:
            # Compatible with old format NPZ (only sim_state = pre-injection)
            logger.warning(f"  NPZ missing post_sim_state, loading pre-injection state for {scene_id}")
            env_wrapper.set_sim_state_flat(npz_data['sim_state'])
        env_wrapper.forward()

    settle_metadata = {}
    if task_name == "stack":
        settle_render = _make_render_fn()
        settle_result = settle_scene_for_source_collection(
            env_wrapper=env_wrapper,
            task_config=task_config,
            logger=logger,
            render_fn=settle_render,
        )
        settle_metadata.update(settle_result.metadata)
        if not settle_result.accepted:
            logger.warning(
                "  Scene %s rejected before teleop: %s",
                scene_id,
                settle_result.reason,
            )
            return None

        stack_validation_config = get_stack_source_validation_config(task_config)
        if (
            bool(stack_validation_config["reject_if_success_at_start"])
            and env_wrapper.check_success()
        ):
            logger.warning(
                "  Scene %s already satisfies check_success() after settling — "
                "skipping invalid error scene",
                scene_id,
            )
            return None
    elif env_wrapper.check_success():
        logger.warning(
            f"  Scene {scene_id} already satisfies check_success() after loading — "
            f"skipping invalid error scene"
        )
        return None  # Return None to let caller switch to a new scene

    # Extract target object from scene labels
    target_object = scene_data.get('labels', {}).get('target_object', '')

    # Display error info
    error_desc = scene_data.get('error_spec', {}).get('error_name', error_name)
    task_desc = task_config.get('task_description', task_name)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Task: {task_desc}")
    logger.info(f"  Demo {demo_idx} | {subtype_id} | Scene: {scene_id}")
    logger.info(f"  Error: {error_desc} ({degree})")
    logger.info(f"  Instructions: Recover from the error and complete the task.")
    logger.info(f"  Max steps: {max_steps}")
    logger.info(f"{'='*60}")

    # Data buffers
    actions_list = []
    states_list = [env_wrapper.get_sim_state_flat()]
    eef_positions = [env_wrapper.get_eef_pos()]
    eef_orientations = [env_wrapper.get_eef_quat()]
    target_poses = []
    gripper_states = [env_wrapper.get_gripper_closed_norm()]
    camera_images = []

    # Record object positions and orientations
    obj_names = env_wrapper.get_all_object_names()
    obj_positions = {}
    obj_orientations = {}
    for name in obj_names:
        pos, quat = env_wrapper.get_object_pose(name)
        obj_positions[name] = [pos]
        obj_orientations[name] = [quat]

    # Capture initial camera image
    if enable_rendering:
        try:
            img = env_wrapper.get_camera_obs(camera_name, (camera_resolution, camera_resolution))
            camera_images.append(img)
        except Exception:
            enable_rendering = False

    # Reset device state before starting teleoperation
    if device is not None:
        device.start_control()

    if target_object:
        print(f"\n  >>> Target object to recover: {target_object} <<<\n")

    # Teleoperation loop
    success = False
    reset_requested = False
    min_recovery_steps = teleop_config.get('min_recovery_steps', 5)
    success_hold_frames = teleop_config.get('success_hold_frames', 10)
    consecutive_success = 0
    for step in range(max_steps):
        # Get action from teleoperation device
        if device is not None:
            action, grasp_signal = input2action(
                device=device, robot=env.robots[0],
            )
            if action is None:
                # Reset signal from device (e.g. right-click on SpaceMouse)
                reset_requested = True
                break
        else:
            # Non-interactive mode: generate placeholder neutral actions
            # (for testing the pipeline without a physical device)
            action = env_wrapper.get_neutral_action()

        target_poses.append(env_wrapper.action_to_target_pose(action))

        # Step environment
        obs, reward, done, info = env_wrapper.step(action)

        # Render for interactive teleoperation
        if enable_rendering:
            if dual_view:
                # 3-camera cv2 window: agentview (left) + sideview/wrist (right, stacked)
                _H = int(camera_resolution * 1.5)
                _half = _H // 2
                try:
                    _agent_img = env_wrapper.get_camera_obs("agentview", (_H, _H))
                    _side_img = env_wrapper.get_camera_obs(_side_cam, (_half, _half))
                    _wrist_img = env_wrapper.get_camera_obs("robot0_eye_in_hand", (_half, _half))
                    _right_col = np.concatenate([_side_img, _wrist_img], axis=0)
                    _combined = np.concatenate([_agent_img, _right_col], axis=1)
                    cv2.imshow("Recovery Collection [agent | side+wrist]",
                               _combined[:, :, ::-1])
                    cv2.waitKey(1)
                except Exception:
                    # Fallback to on-screen renderer
                    if hasattr(env, 'renderer') and env.has_renderer:
                        env.render()
            elif hasattr(env, 'renderer') and env.has_renderer:
                env.render()

        # Record data
        actions_list.append(action.copy())
        states_list.append(env_wrapper.get_sim_state_flat())
        eef_positions.append(env_wrapper.get_eef_pos())
        eef_orientations.append(env_wrapper.get_eef_quat())
        gripper_states.append(env_wrapper.get_gripper_closed_norm())

        for name in obj_names:
            pos, quat = env_wrapper.get_object_pose(name)
            obj_positions[name].append(pos)
            obj_orientations[name].append(quat)

        if enable_rendering and step % 4 == 0:  # Record every 4th frame (5Hz)
            try:
                img = env_wrapper.get_camera_obs(
                    camera_name, (camera_resolution, camera_resolution))
                camera_images.append(img)
            except Exception:
                pass

        # Check success (must satisfy for success_hold_frames consecutive frames)
        if step >= min_recovery_steps and env_wrapper.check_success():
            consecutive_success += 1
            if consecutive_success >= success_hold_frames:
                # Release + lift 10cm to verify (ensure not held by gripper)
                logger.info(f"  Held {success_hold_frames} frames, "
                             f"verifying with release + lift...")
                verify_render = _make_render_fn()
                env_wrapper.set_gripper_state(1.0, steps=10, render_fn=verify_render)
                env_wrapper.apply_eef_offset(
                    np.array([0.0, 0.0, 0.10]), max_steps=30,
                    render_fn=verify_render)
                if env_wrapper.check_success():
                    success = True
                    logger.info(f"  Recovery successful at step {step + 1} "
                                 f"(verified: release + lift 10cm)!")
                else:
                    success = False
                    logger.info(f"  Release+lift verification FAILED — "
                                 f"object not stable!")
                break
        else:
            consecutive_success = 0

        if done:
            break

    num_steps = len(actions_list)

    if reset_requested:
        logger.info(f"  Demo reset by user at step {num_steps}.")
        return None  # Caller should retry with a new scene

    # Trim leading zero-action frames (OSC controller transient before human input)
    trim_start = 0
    for i, act in enumerate(actions_list):
        if np.linalg.norm(act[:6]) > 1e-6:
            trim_start = i
            break
    if trim_start > 0:
        logger.info(f"  Trimming {trim_start} leading zero-action frames (OSC transient)")
        actions_list = actions_list[trim_start:]
        states_list = states_list[trim_start:]       # states[i] is the state BEFORE actions[i]
        eef_positions = eef_positions[trim_start:]
        eef_orientations = eef_orientations[trim_start:]
        gripper_states = gripper_states[trim_start:]
        target_poses = target_poses[trim_start:]
        for name in obj_names:
            obj_positions[name] = obj_positions[name][trim_start:]
            obj_orientations[name] = obj_orientations[name][trim_start:]
        # camera_images are sampled every 4 frames, adjust accordingly
        if camera_images:
            cam_trim = trim_start // 4
            camera_images = camera_images[cam_trim:]
        num_steps = len(actions_list)

    logger.info(f"  Demo complete: {num_steps} steps, success={success}")

    # Create RecoveryDemo
    demo = RecoveryDemo(
        demo_id=f"recovery_{task_name}_{subtype_id}_{demo_idx:04d}",
        task_name=task_name,
        error_name=error_name,
        degree=degree,
        scene_id=scene_id,
        scene_npz_path=npz_path,
        success=success,
        num_steps=num_steps,
        actions=np.array(actions_list) if actions_list else None,
        states=np.array(states_list) if states_list else None,
        camera_images=camera_images if camera_images else None,
        eef_positions=np.array(eef_positions) if eef_positions else None,
        eef_orientations=np.array(eef_orientations) if eef_orientations else None,
        target_poses=np.array(target_poses) if target_poses else None,
        gripper_states=np.array(gripper_states) if gripper_states else None,
        object_positions={
            name: np.array(pos) for name, pos in obj_positions.items()
        } if obj_positions else None,
        object_orientations={
            name: np.array(quat) for name, quat in obj_orientations.items()
        } if obj_orientations else None,
        metadata={
            'target_object': target_object,
            'scene_labels': scene_data.get('labels', {}),
            'trimmed_leading_frames': trim_start,
            **settle_metadata,
        },
    )

    # Segment recovery trajectory
    if success and demo.eef_positions is not None:
        demo.subtasks = segmenter.segment(demo)

    return demo


def save_demo(demo: RecoveryDemo, output_dir: Path):
    """Save a recovery demo to disk (NPZ + JSON metadata)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays to NPZ
    npz_path = output_dir / f"{demo.demo_id}.npz"
    save_dict = {}

    # Metadata
    save_dict['success'] = np.bool_(demo.success)
    save_dict['num_steps'] = np.int32(demo.num_steps)
    save_dict['demo_id'] = np.array(demo.demo_id)
    save_dict['task_name'] = np.array(demo.task_name)
    save_dict['error_name'] = np.array(demo.error_name)
    save_dict['degree'] = np.array(demo.degree)
    save_dict['scene_id'] = np.array(demo.scene_id)

    if demo.actions is not None:
        save_dict['actions'] = demo.actions
    if demo.states is not None:
        save_dict['states'] = demo.states
    if demo.eef_positions is not None:
        save_dict['eef_positions'] = demo.eef_positions
    if demo.eef_orientations is not None:
        save_dict['eef_orientations'] = demo.eef_orientations
    if demo.target_poses is not None:
        save_dict['target_poses'] = demo.target_poses
    if demo.gripper_states is not None:
        save_dict['gripper_states'] = demo.gripper_states

    # Save object positions and orientations
    if demo.object_positions is not None:
        for name, pos in demo.object_positions.items():
            save_dict[f'obj_{name}'] = pos
    if demo.object_orientations is not None:
        for name, quat in demo.object_orientations.items():
            save_dict[f'obj_{name}_quat'] = quat

    # Save camera images separately (can be large)
    if demo.camera_images:
        images_array = np.array(demo.camera_images)
        save_dict['camera_images'] = images_array

    np.savez_compressed(npz_path, **save_dict)

    return npz_path


def load_or_create_manifest(
    manifest_path: Path,
    task_name: str,
    target_counts: dict,
    quota_rule: str,
    collection_mode: str,
    target_scene_ids: dict | None = None,
) -> tuple:
    """Load existing manifest or create new one. Returns (demos_list, status, duplicates_removed)."""
    duplicates_removed = 0
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        demos = manifest.get('demos', [])
        demos, duplicates_removed = dedupe_demo_entries(demos)
        status = RecoveryCollectionStatus.from_dict(manifest.get('status', {}))
    else:
        demos = []
        status = RecoveryCollectionStatus(
            task_name=task_name,
            collection_mode=collection_mode,
            target_demos=target_counts,
        )

    status = refresh_collection_status(
        status,
        demos,
        target_counts=target_counts,
        quota_rule=quota_rule,
        collection_mode=collection_mode,
        target_scene_ids=target_scene_ids,
    )
    return demos, status, duplicates_removed



def print_progress(
    task_name: str,
    round_num: int,
    total_rounds: int,
    current_subtype: str,
    subtype_idx: int,
    total_subtypes: int,
    status,
    target_counts: dict,
):
    """Print a progress bar and round status for round-robin collection."""
    total_collected = sum(status.get_collected(s) for s in target_counts)
    total_target = sum(target_counts.values())
    pct = (total_collected / total_target * 100) if total_target > 0 else 0

    # Progress bar (40 chars wide)
    filled = int(40 * total_collected / total_target) if total_target > 0 else 0
    bar = '\u2588' * filled + '\u2591' * (40 - filled)

    subtype_collected = status.get_collected(current_subtype)
    subtype_target = target_counts.get(current_subtype, 0)

    print(f"\n{'='*62}")
    print(f"  Task: {task_name} | Round {round_num}/{total_rounds} "
          f"| Overall: {total_collected}/{total_target} ({pct:.1f}%)")
    print(f"  Current: {current_subtype} [{subtype_collected}/{subtype_target}] "
          f"(subtype {subtype_idx}/{total_subtypes})")
    print(f"  {bar}  {pct:.1f}%")
    print(f"{'='*62}")


def copy_to_collected_data(
    npz_path: Path,
    task_name: str,
    subtype_id: str,
    collected_data_dir: Path,
    logger,
):
    """Copy a validated demo NPZ to the collected_data directory."""
    dest_dir = collected_data_dir / task_name / subtype_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(npz_path).name
    import shutil
    shutil.copy2(str(npz_path), str(dest_path))
    logger.info(f"    Copied to collected_data: {dest_path}")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Collect recovery demos")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., stack, pick_place)")
    parser.add_argument("--subtype", type=str, default=None,
                        help="Specific subtype to collect (e.g., collision_holding_D0)")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Override number of demos to collect")
    parser.add_argument(
        "--collection_mode",
        type=str,
        default=None,
        choices=["quota", "per_scene"],
        help="Collection target mode: legacy subtype quota or one counted demo per scene",
    )
    parser.add_argument("--all_subtypes", action="store_true",
                        help="Collect all subtypes from the active scene pool / legacy quota config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous collection session")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/recovery_collection.yaml")
    parser.add_argument("--scenes_dir", type=str, default=None,
                        help="Override error scenes directory (default: v5_training/{task}/scenes)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, default="keyboard",
                        choices=["spacemouse", "keyboard", "none"],
                        help="Teleoperation device")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable camera image recording")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", action="store_true", default=True,
                        help="Replay injection animation before teleop (default: on)")
    parser.add_argument("--no_replay", dest="replay", action="store_false",
                        help="Skip injection animation replay")
    parser.add_argument("--replay_speed", type=float, default=1.0,
                        help="Replay speed multiplier (1.0 = real-time)")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable post-collection MimicGen compatibility validation")
    parser.add_argument("--validation_replay_only", action="store_true",
                        help="Run state-action replay validation only, skip augmentability testing")
    parser.add_argument("--validation_max_scenes", type=int, default=None,
                        help="Override number of target scenes tested for augmentability")
    parser.add_argument("--render_validation", action="store_true",
                        help="Record MP4 video during action replay validation")
    parser.add_argument("--render_mimicgen", action="store_true",
                        help="Record MP4 video for MimicGen augmentation attempts during validation")
    parser.add_argument("--dual_view", action="store_true",
                        help="Show 3-camera view (agentview + sideview + wrist) during teleop")
    parser.add_argument("--round_robin", action="store_true",
                        help="Round-robin collection: cycle through all subtypes per round "
                             "(1 demo per subtype per round, repeat num_demos rounds)")
    parser.add_argument("--collected_data_dir", type=str, default=None,
                        help="Copy validated demos to this root-level directory "
                             "(organized by task/subtype)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_episode_steps from teleop config")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configs
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        recovery_config = yaml.safe_load(f)

    task_info = load_task_registry(args.task)
    dataset_path = task_info['dataset_path']

    task_config_path = task_info.get('task_config', '')
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)

    teleop_config = recovery_config.get('teleoperation', {})
    if args.max_steps is not None:
        teleop_config['max_episode_steps'] = args.max_steps
    seg_config = recovery_config.get('segmentation', {})
    collection_config = recovery_config.get('collection', {})
    collection_mode = str(
        args.collection_mode or collection_config.get('mode', 'per_scene')
    ).strip().lower()
    if collection_mode not in {"quota", "per_scene"}:
        parser.error(
            f"Unsupported collection_mode '{collection_mode}'. Expected quota or per_scene."
        )
    validation_config = recovery_config.get('post_collection_validation', {}).copy()
    if args.no_validation:
        validation_config['enabled'] = False
    if args.validation_replay_only:
        validation_config['check_scene_augmentation'] = False
    if args.validation_max_scenes is not None:
        validation_config['max_target_scenes'] = args.validation_max_scenes
    quota_rule = normalize_quota_rule(validation_config.get('quota_rule', 'replay_only'))
    keep_rejected_successes = bool(
        validation_config.get('keep_rejected_successes', True)
    )
    validation_config['quota_rule'] = quota_rule
    validation_config['keep_rejected_successes'] = keep_rejected_successes
    if collection_mode == "per_scene" and args.num_demos is not None:
        parser.error("--num_demos is not compatible with collection_mode=per_scene")
    if not validation_config.get('enabled', True) and quota_rule != 'success_only':
        parser.error(
            "--no_validation is only compatible with quota_rule=success_only"
        )
    if quota_rule == 'replay_only' and not validation_config.get('check_action_replay', True):
        parser.error(
            "quota_rule=replay_only requires check_action_replay=true"
        )
    if (
        quota_rule == 'augmentable_only'
        and not validation_config.get('check_scene_augmentation', True)
    ):
        parser.error(
            "quota_rule=augmentable_only requires check_scene_augmentation=true"
        )
    if (
        quota_rule == 'replay_and_augmentable'
        and not validation_config.get('check_action_replay', True)
    ):
        parser.error(
            "quota_rule=replay_and_augmentable requires check_action_replay=true"
        )
    if (
        quota_rule == 'replay_and_augmentable'
        and not validation_config.get('check_scene_augmentation', True)
    ):
        parser.error(
            "quota_rule=replay_and_augmentable requires check_scene_augmentation=true"
        )

    # Paths
    output_base = args.output_dir or recovery_config['paths']['recovery_demos_dir']
    output_dir = PROJECT_ROOT / output_base / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes_base = args.scenes_dir or f"error_benchmark/outputs/v5_training/{args.task}/scenes"
    scenes_dir = PROJECT_ROOT / scenes_base

    # Legacy demo allocations for this task (still used in quota mode)
    allocations = recovery_config.get('demo_allocations', {}).get(args.task, {})
    if collection_mode == "quota" and not allocations:
        logger.error(f"No demo allocations found for task '{args.task}'")
        sys.exit(1)

    available_scene_subtypes = discover_scene_subtypes(scenes_dir)

    # Determine which subtypes to collect
    if args.subtype:
        subtypes_to_collect = [args.subtype]
    elif args.all_subtypes:
        subtypes_to_collect = (
            available_scene_subtypes
            if collection_mode == "per_scene"
            else list(allocations.keys())
        )
    else:
        logger.error("Specify --subtype or --all_subtypes")
        sys.exit(1)

    scene_pool_by_subtype = {
        subtype_id: load_error_scenes(scenes_dir, subtype_id)
        for subtype_id in subtypes_to_collect
    }
    target_scene_ids = {
        subtype_id: [scene.get("scene_id", "") for scene in scenes if scene.get("scene_id")]
        for subtype_id, scenes in scene_pool_by_subtype.items()
    }
    if collection_mode == "per_scene":
        target_counts = {
            subtype_id: len(target_scene_ids.get(subtype_id, []))
            for subtype_id in subtypes_to_collect
        }
    else:
        if args.num_demos is not None:
            target_counts = {
                subtype_id: args.num_demos
                for subtype_id in subtypes_to_collect
            }
        else:
            target_counts = {
                subtype_id: allocations.get(subtype_id, 0)
                for subtype_id in subtypes_to_collect
            }

    # Load manifest
    manifest_path = output_dir / "manifest.json"
    demos_list, status, duplicates_removed = load_or_create_manifest(
        manifest_path,
        args.task,
        target_counts,
        quota_rule=quota_rule,
        collection_mode=collection_mode,
        target_scene_ids=target_scene_ids if collection_mode == "per_scene" else None,
    )
    if duplicates_removed > 0:
        logger.warning(
            "Manifest dedupe removed %d duplicate demo_id entries from %s",
            duplicates_removed,
            manifest_path,
        )
        save_manifest(manifest_path, demos_list, status)

    # Create environment
    enable_camera = not args.no_render
    # dual_view uses cv2 offscreen rendering — no need for MuJoCo on-screen window
    has_renderer = (args.device != "none") and not args.dual_view
    env = create_env(task_config, dataset_path, enable_camera=enable_camera,
                     camera_resolution=args.resolution,
                     has_renderer=has_renderer)

    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    # Create segmenter
    segmenter = RecoverySegmenter(task_config, seg_config)

    # Post-collection validation setup
    validation_report_path = output_dir / validation_config.get(
        'report_filename', DEFAULT_REPORT_FILENAME)
    validation_records = []
    validator = None
    if validation_config.get('enabled', True):
        validation_records = load_validation_records(validation_report_path)
        validator = RecoveryDemoValidator(
            env_wrapper=env_wrapper,
            task_config=task_config,
            aug_config=recovery_config.get('augmentation', {}),
            scenes_dir=scenes_dir,
            validation_config=validation_config,
            rng=np.random.RandomState(args.seed + 1000),
            demos_dir=output_dir,
        )

    # Create injection replay (for showing injection animation before teleop)
    injection_replay = None
    clean_traj_dir = None
    replay_cfg = recovery_config.get('replay_animation', {})
    replay_config = {
        'enabled': args.replay,
        'speed': args.replay_speed,
        'context_frames': replay_cfg.get('context_frames', 50),
        'pause_after': replay_cfg.get('pause_after', 1.5),
    }
    if args.replay:
        # Load skills config from benchmark_v5 yaml
        benchmark_config_path = PROJECT_ROOT / "error_benchmark" / "configs" / "benchmark_v5.yaml"
        skills_config = {}
        if benchmark_config_path.exists():
            with open(benchmark_config_path) as f:
                bench_cfg = yaml.safe_load(f)
            skills_config = bench_cfg.get('error_skills', {})

        injection_replay = InjectionReplay(env_wrapper, task_config, skills_config)
        clean_traj_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "v5" / "clean_trajectories" / args.task
        if not clean_traj_dir.exists():
            logger.warning(f"Clean trajectory dir not found: {clean_traj_dir}")
            logger.warning("Injection replay will fall back to direct state load")

    # Create teleoperation device
    device = None
    if args.device != "none":
        try:
            device = create_teleop_device(args.device, teleop_config)
        except RuntimeError as e:
            logger.warning(f"Device creation failed: {e}. Running in non-interactive mode.")

    # Collected data directory (for validated demos)
    collected_data_dir = None
    if args.collected_data_dir:
        collected_data_dir = PROJECT_ROOT / args.collected_data_dir
        collected_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Recovery Demo Collection ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Collection mode: {collection_mode}")
    if args.round_robin:
        logger.info(f"Schedule: round-robin ({args.num_demos or 10} rounds)")
    logger.info(f"Subtypes: {len(subtypes_to_collect)} subtypes")
    logger.info(f"Target: {sum(target_counts.values())} total demos "
                f"({args.num_demos or 'config'} per subtype)")
    logger.info(f"Device: {args.device}")
    logger.info(f"Injection replay: {'ON' if args.replay else 'OFF'}"
                f"{f' (speed={args.replay_speed}x)' if args.replay else ''}")
    logger.info(f"Output: {output_dir}")
    if collected_data_dir:
        logger.info(f"Collected data: {collected_data_dir / args.task}")
    if validation_config.get('enabled', True):
        logger.info(
            "Post-collection validation: ON "
            f"(replay={'Y' if validation_config.get('check_action_replay', True) else 'N'}, "
            f"augment={'Y' if validation_config.get('check_scene_augmentation', True) else 'N'}, "
            f"max_scenes={validation_config.get('max_target_scenes', 10)}, "
            f"quota_rule={quota_rule})"
        )
        logger.info(f"Validation report: {validation_report_path}")
    else:
        logger.info(f"Post-collection validation: OFF (quota_rule={quota_rule})")

    rng = np.random.RandomState(args.seed)
    status_target_scene_ids = target_scene_ids if collection_mode == "per_scene" else None

    # ── Pre-compute subtype info ──
    subtype_info = {}
    for subtype_id in subtypes_to_collect:
        for d in ['D0', 'D1']:
            if subtype_id.endswith(f'_{d}'):
                subtype_info[subtype_id] = {
                    'error_name': subtype_id[:-(len(d) + 1)],
                    'degree': d,
                }
                break
        else:
            parts = subtype_id.rsplit('_', 1)
            subtype_info[subtype_id] = {
                'error_name': parts[0] if len(parts) == 2 else subtype_id,
                'degree': parts[1] if len(parts) == 2 else '?',
            }

    max_retries = teleop_config.get('max_retries_per_scene', 3)
    per_scene_mode = (collection_mode == "per_scene")

    # Sort subtypes: by error_skill name, D0 before D1
    def _subtype_sort_key(sid):
        info = subtype_info[sid]
        return (info['error_name'], 0 if info['degree'] == 'D0' else 1)
    subtypes_to_collect = sorted(subtypes_to_collect, key=_subtype_sort_key)

    # ── Helper: process a successful demo (validate, save, copy) ──
    def _process_success(demo, subtype_id, subtype_dir):
        """Validate, save, and optionally copy a successful demo.

        Returns (counts_toward_target: bool, npz_path_or_none).
        Updates demos_list, status, validation_records as side effects.
        """
        nonlocal demos_list, status, validation_records

        validation_record = None
        npz_path = save_demo(demo, subtype_dir)

        if validator is not None:
            logger.info("    Running MimicGen compatibility validation...")
            try:
                _val_video_dir = (
                    PROJECT_ROOT / "all_videos" / "action_replay"
                ) if args.render_validation else None
                _mimicgen_video_dir = (
                    PROJECT_ROOT / "all_videos" / "mimicgen"
                ) if args.render_mimicgen else None
                validation_record = validator.validate_demo(
                    demo, video_dir=_val_video_dir,
                    mimicgen_video_dir=_mimicgen_video_dir)
            except Exception as e:
                validation_record = {
                    'demo_id': demo.demo_id,
                    'task_name': demo.task_name,
                    'subtype_id': demo.subtype_id,
                    'scene_id': demo.scene_id,
                    'success_recorded': bool(demo.success),
                    'validated_at': datetime.now().isoformat(),
                    'action_replay': {'status': 'validator_exception'},
                    'scene_augmentation': {'status': 'validator_exception'},
                    'summary': {
                        'replay_success': None,
                        'augmentable': None,
                        'augment_success_count': 0,
                        'augment_tested_scene_count': 0,
                    },
                    'error': f"{type(e).__name__}: {e}",
                }
            upsert_validation_record(validation_records, validation_record)
            save_validation_report(
                validation_report_path, args.task,
                validation_records, validation_config,
            )
            log_validation_result(logger, validation_record)

        manifest_entry = build_manifest_entry(
            demo, quota_rule=quota_rule,
            counts_toward_target=None, quota_reason='pending_validation',
            validation_record=validation_record,
        )
        counts_toward_target, quota_reason = determine_counts_toward_target(
            manifest_entry, quota_rule=quota_rule,
        )
        manifest_entry['counts_toward_target'] = counts_toward_target
        manifest_entry['quota_reason'] = quota_reason

        should_save = counts_toward_target or keep_rejected_successes
        if should_save:
            manifest_entry['npz_path'] = str(npz_path)
            tag_npz_validation(npz_path, validation_record, counts_toward_target)
        else:
            try:
                npz_path.unlink()
            except OSError:
                pass

        if should_save:
            upsert_demo_entry(demos_list, manifest_entry)
            status = refresh_collection_status(
                status, demos_list,
                target_counts=target_counts, quota_rule=quota_rule,
                collection_mode=collection_mode,
                target_scene_ids=status_target_scene_ids,
            )
            save_manifest(manifest_path, demos_list, status)

        counted = status.get_collected(subtype_id)
        target_count = target_counts.get(subtype_id, 0)

        if counts_toward_target:
            logger.info(
                f"    [{subtype_id}] Accepted for quota {counted}/{target_count}")
            # Copy to collected_data
            if collected_data_dir and should_save:
                copy_to_collected_data(
                    npz_path, args.task, subtype_id,
                    collected_data_dir, logger)
        elif should_save:
            logger.info(
                f"    [{subtype_id}] Saved success but not counted "
                f"(reason={quota_reason}, quota={counted}/{target_count})")
        else:
            logger.info(
                f"    [{subtype_id}] Rejected success and not saved "
                f"(reason={quota_reason}, quota={counted}/{target_count})")

        return counts_toward_target, npz_path if should_save else None

    # ── Helper: record a failed demo ──
    def _record_failure(demo, subtype_id):
        nonlocal demos_list, status
        manifest_entry = build_manifest_entry(
            demo, quota_rule=quota_rule,
            counts_toward_target=False, quota_reason='teleop_failed',
        )
        upsert_demo_entry(demos_list, manifest_entry)
        status = refresh_collection_status(
            status, demos_list,
            target_counts=target_counts, quota_rule=quota_rule,
            collection_mode=collection_mode,
            target_scene_ids=status_target_scene_ids,
        )
        save_manifest(manifest_path, demos_list, status)

    # ── Helper: save progress and exit ──
    def _save_and_exit():
        save_manifest(manifest_path, demos_list, status)
        if validator is not None:
            save_validation_report(
                validation_report_path, args.task,
                validation_records, validation_config,
            )
        _shutdown_device(device)
        sys.exit(0)

    # ── Helper: collect one demo attempt ──
    def _attempt_one(subtype_id, scene):
        info = subtype_info[subtype_id]
        attempt_idx = get_next_demo_attempt_index(
            demos_list, args.task, subtype_id)
        return collect_single_demo(
            env_wrapper=env_wrapper,
            scene_data=scene,
            task_config=task_config,
            teleop_config=teleop_config,
            segmenter=segmenter,
            task_name=args.task,
            error_name=info['error_name'],
            degree=info['degree'],
            demo_idx=attempt_idx,
            camera_name=args.camera,
            camera_resolution=args.resolution,
            enable_rendering=enable_camera,
            dual_view=args.dual_view,
            device=device,
            injection_replay=injection_replay,
            clean_traj_dir=clean_traj_dir,
            replay_config=replay_config,
        )

    # ═══════════════════════════════════════════════════════════════
    # COLLECTION LOOP
    # ═══════════════════════════════════════════════════════════════

    if args.round_robin:
        # ── Round-robin mode ──
        # Cycle through all subtypes per round, collect 1 counted demo each.
        num_rounds = max(target_counts.values()) if target_counts else 0
        rr_scene = {sid: None for sid in subtypes_to_collect}
        rr_fail = {sid: 0 for sid in subtypes_to_collect}

        for round_num in range(1, num_rounds + 1):
            all_done = True
            for subtype_idx, subtype_id in enumerate(subtypes_to_collect, 1):
                target_count = target_counts.get(subtype_id, 0)
                if status.get_collected(subtype_id) >= target_count:
                    continue
                all_done = False

                scenes = scene_pool_by_subtype.get(subtype_id, [])
                if not scenes:
                    continue

                # Show progress
                print_progress(
                    args.task, round_num, num_rounds,
                    subtype_id, subtype_idx, len(subtypes_to_collect),
                    status, target_counts)

                subtype_dir = output_dir / subtype_id
                got_counted = False

                while not got_counted:
                    # Pick a scene
                    if rr_scene[subtype_id] is None:
                        rr_scene[subtype_id] = scenes[
                            rng.randint(0, len(scenes))]
                        rr_fail[subtype_id] = 0
                    scene = rr_scene[subtype_id]

                    try:
                        demo = _attempt_one(subtype_id, scene)

                        # User skip → try new scene, stay on same subtype
                        if demo is None:
                            rr_scene[subtype_id] = None
                            continue

                        # Failed → retry with limit
                        if not demo.success:
                            rr_fail[subtype_id] += 1
                            _record_failure(demo, subtype_id)
                            if rr_fail[subtype_id] >= max_retries:
                                logger.info(
                                    f"    [{subtype_id}] {rr_fail[subtype_id]} "
                                    f"failures, switching scene")
                                rr_scene[subtype_id] = None
                            continue

                        # Success → validate and save
                        counted, _ = _process_success(
                            demo, subtype_id, subtype_dir)
                        rr_scene[subtype_id] = None
                        if counted:
                            got_counted = True
                        # else: validation rejected, try again

                    except KeyboardInterrupt:
                        logger.info(
                            "\nCollection interrupted. Saving progress...")
                        _save_and_exit()
                    except Exception as e:
                        logger.error(f"    Error collecting demo: {e}")
                        rr_scene[subtype_id] = None
                        break

            if all_done:
                logger.info("\nAll subtypes have reached their targets!")
                break

    else:
        # ── Sequential mode (original behavior) ──
        for subtype_id in subtypes_to_collect:
            info = subtype_info[subtype_id]
            error_name, degree = info['error_name'], info['degree']
            target_count = status.get_target(subtype_id)
            collected = status.get_collected(subtype_id)

            scenes = scene_pool_by_subtype.get(subtype_id, [])
            if not scenes:
                logger.warning(
                    f"[{subtype_id}] No error scenes found in {scenes_dir}")
                continue

            if collected >= target_count:
                logger.info(
                    f"[{subtype_id}] Already collected "
                    f"{collected}/{target_count}, skipping")
                continue

            logger.info(
                f"\n--- Collecting {subtype_id}: "
                f"{collected}/{target_count} ---")
            logger.info(f"    Available scenes: {len(scenes)}")

            subtype_dir = output_dir / subtype_id

            if per_scene_mode:
                covered_scene_ids = set(
                    status.get_collected_scene_ids(subtype_id))
                pending_scenes = deque(
                    scene for scene in scenes
                    if scene.get("scene_id") not in covered_scene_ids
                )
                scene_fail_counts = {
                    scene.get("scene_id", ""): 0
                    for scene in pending_scenes
                    if scene.get("scene_id")
                }
                logger.info(
                    f"    Pending scene coverage: "
                    f"{len(pending_scenes)}/{len(scenes)}")
                if not pending_scenes:
                    continue
                scene = pending_scenes[0]
                scene_fail_count = scene_fail_counts.get(
                    scene.get("scene_id", ""), 0)
            else:
                pending_scenes = None
                scene_fail_counts = {}
                scene = None
                scene_fail_count = 0

            while (
                len(pending_scenes) > 0
                if per_scene_mode
                else status.get_collected(subtype_id) < target_count
            ):
                if per_scene_mode:
                    scene = pending_scenes[0]
                    scene_fail_count = scene_fail_counts.get(
                        scene.get("scene_id", ""), 0)
                elif scene is None:
                    scene = scenes[rng.randint(0, len(scenes))]
                    scene_fail_count = 0
                scene_id = scene.get("scene_id", "")

                try:
                    demo = _attempt_one(subtype_id, scene)

                    if demo is None:
                        if per_scene_mode:
                            logger.info(
                                f"    [{subtype_id}] User skipped scene "
                                f"{scene_id}")
                            pending_scenes.rotate(-1)
                        else:
                            scene = None
                        continue

                    if not demo.success:
                        scene_fail_count += 1
                        _record_failure(demo, subtype_id)

                        if per_scene_mode:
                            scene_fail_counts[scene_id] = scene_fail_count
                            if scene_fail_count >= max_retries:
                                logger.info(
                                    f"    [{subtype_id}] Scene {scene_id} "
                                    f"failed {scene_fail_count} times, "
                                    f"deferring")
                                scene_fail_counts[scene_id] = 0
                                pending_scenes.rotate(-1)
                            else:
                                logger.info(
                                    f"    [{subtype_id}] Demo failed "
                                    f"({scene_fail_count}/{max_retries}), "
                                    f"retrying...")
                        else:
                            if scene_fail_count >= max_retries:
                                logger.info(
                                    f"    [{subtype_id}] Scene failed "
                                    f"{scene_fail_count} times, skipping")
                                scene = None
                            else:
                                logger.info(
                                    f"    [{subtype_id}] Demo failed "
                                    f"({scene_fail_count}/{max_retries}), "
                                    f"retrying...")
                        continue

                    counted, _ = _process_success(
                        demo, subtype_id, subtype_dir)

                    if per_scene_mode:
                        counted_n = status.get_collected(subtype_id)
                        covered_after = set(
                            status.get_collected_scene_ids(subtype_id))
                        if scene_id in covered_after:
                            pending_scenes.popleft()
                            scene_fail_counts.pop(scene_id, None)
                            logger.info(
                                f"    [{subtype_id}] Covered scene "
                                f"{scene_id} ({counted_n}/{target_count})")
                        else:
                            pending_scenes.rotate(-1)
                    else:
                        scene = None

                except KeyboardInterrupt:
                    logger.info(
                        "\nCollection interrupted. Saving progress...")
                    _save_and_exit()
                except Exception as e:
                    logger.error(f"    Error collecting demo: {e}")
                    if per_scene_mode and pending_scenes:
                        pending_scenes.rotate(-1)
                    else:
                        scene = None
                    continue

    # Shutdown device and save
    _shutdown_device(device)
    save_manifest(manifest_path, demos_list, status)
    if validator is not None:
        save_validation_report(
            validation_report_path,
            args.task,
            validation_records,
            validation_config,
        )

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Collection Summary — {args.task}")
    logger.info(f"{'=' * 70}")

    summary = status.summary()
    total_success = sum(1 for d in demos_list if d.get('success'))
    total_counted = sum(1 for d in demos_list if d.get('counts_toward_target'))
    total_rejected_success = max(total_success - total_counted, 0)
    total_fail = sum(1 for d in demos_list if not d.get('success'))

    # Per-demo detail log
    logger.info(f"\n  --- Per-Demo Details ---")
    for i, d in enumerate(demos_list):
        if d.get('counts_toward_target'):
            s_tag = "COUNTED"
        elif d.get('success'):
            s_tag = "REJECTED"
        else:
            s_tag = "FAILED"
        n_steps = d.get('num_steps', '?')
        sid = d.get('subtype_id', '?')
        demo_id = d.get('demo_id', '?')
        quota_reason = d.get('quota_reason', '-')
        logger.info(
            f"  [{s_tag:^8}] {demo_id}  |  subtype: {sid}  |  "
            f"frames: {n_steps}  |  reason: {quota_reason}"
        )

    # Per-subtype breakdown
    logger.info(f"\n  --- Per-Subtype Breakdown ---")
    for sid in sorted(target_counts.keys()):
        c = status.get_collected(sid)
        t = status.get_target(sid)
        saved_success = sum(
            1 for d in demos_list
            if d.get('subtype_id') == sid and d.get('success')
        )
        marker = "OK" if c >= t else f"NEED {t - c}"
        if collection_mode == "per_scene":
            logger.info(
                f"  {sid}: covered_scenes {c}/{t}, saved_success {saved_success} [{marker}]"
            )
        else:
            logger.info(
                f"  {sid}: counted {c}/{t}, saved_success {saved_success} [{marker}]"
            )

    # Overall summary
    logger.info(f"\n  --- Overall ---")
    logger.info(f"  Task:             {args.task}")
    logger.info(f"  Collection mode:  {collection_mode}"
                f"{' (round-robin)' if args.round_robin else ''}")
    logger.info(
        f"  Total counted:    {summary['total_collected']}/{summary['total_target']}"
    )
    logger.info(f"  Saved successes:  {total_success}")
    logger.info(f"  Rejected success: {total_rejected_success}")
    logger.info(f"  Failed:           {total_fail}")
    logger.info(
        f"  Subtypes done:    {summary['subtypes_complete']}/{summary['subtypes_total']}"
    )

    # Collected data summary
    if collected_data_dir:
        task_data_dir = collected_data_dir / args.task
        if task_data_dir.exists():
            n_validated = sum(
                1 for f in task_data_dir.rglob("*.npz")
            )
            n_subtypes_with_data = sum(
                1 for d in task_data_dir.iterdir()
                if d.is_dir() and any(d.glob("*.npz"))
            )
            logger.info(f"\n  --- Validated Output ---")
            logger.info(f"  Directory:        {task_data_dir}")
            logger.info(f"  Validated demos:  {n_validated}")
            logger.info(f"  Subtypes covered: {n_subtypes_with_data}")

    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
