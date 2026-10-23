#!/usr/bin/env python
"""Generate trajectories from error scenes using MimicGen DataGenerator.

Applies the successful monkey-patch reset approach from generate_success_from_error.py:
- Patch env.reset() to inject error scene state after normal reset
- Call DataGenerator.generate() which handles waypoint warping correctly
- OSC controller is properly initialized via normal reset flow

This approach achieved 70% success rate (168/240 scenes) in previous runs.

Usage:
    python error_benchmark/scripts/augmentation/3c_generate_from_error_scene.py \
        --task stack --num-workers 48 --max-attempts 10

    python error_benchmark/scripts/augmentation/3c_generate_from_error_scene.py \
        --task coffee --dry-run 5 --max-attempts 20
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.configs import config_factory, MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface

from robomimic.utils.file_utils import get_env_metadata_from_dataset

from error_benchmark.framework.mimicgen_source_validation import DEFAULT_ENV_INTERFACE_TYPE, ensure_env_interface_attrs

logger = logging.getLogger(__name__)

# ── Task → MimicGen config mapping ────────────────────────────────────────
PREPARED_SOURCE_DIR = str(PROJECT_ROOT / "data" / "mimicgen_prepared")

TASK_REGISTRY = {
    "stack": {
        "mg_template": "stack.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "stack_d0_prepared.hdf5"),
        "env_interface_name": "MG_Stack",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
    "pick_place": {
        "mg_template": "pick_place.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "pick_place_d0_prepared.hdf5"),
        "env_interface_name": "MG_PickPlace",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
    "coffee": {
        "mg_template": "coffee.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "coffee_d0_prepared.hdf5"),
        "env_interface_name": "MG_Coffee",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
    "threading": {
        "mg_template": "threading.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "threading_d0_prepared.hdf5"),
        "env_interface_name": "MG_Threading",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
    "stack_three": {
        "mg_template": "stack_three.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "stack_three_d0_prepared.hdf5"),
        "env_interface_name": "MG_StackThree",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
    "three_piece_assembly": {
        "mg_template": "three_piece_assembly.json",
        "source_dataset": os.path.join(PREPARED_SOURCE_DIR, "three_piece_assembly_d0_prepared.hdf5"),
        "env_interface_name": "MG_ThreePieceAssembly",
        "env_interface_type": DEFAULT_ENV_INTERFACE_TYPE,
    },
}

MIMICGEN_ROOT = PROJECT_ROOT / "shared" / "mimicgen_workspace"

# ── Worker state ────────────────────────────────────────────────────────────

_worker_env = None
_worker_env_interface = None
_worker_data_generator = None
_worker_raw_env = None
_worker_task_name = None
_worker_max_attempts = 10


def _load_mg_config(task_name: str):
    """Load MimicGen config for a task from template."""
    template_dir = MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "exps" / "templates" / "robosuite"
    template_file = template_dir / TASK_REGISTRY[task_name]["mg_template"]

    with open(template_file) as f:
        ext_cfg = json.load(f)

    if "meta" in ext_cfg:
        del ext_cfg["meta"]

    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

    return mg_config


def _filter_valid_demos(source_dataset, all_demos, task_name):
    """Filter demos that have valid subtask termination signals.

    Some MimicGen-generated demos may have incomplete subtask signals.
    For stack task, we need demos with valid grasp transition.
    """
    import h5py

    valid_demo_keys = []
    bad_demo_count = 0

    if task_name == "stack":
        with h5py.File(source_dataset, 'r') as f:
            for ep in all_demos:
                try:
                    # Check for valid grasp termination signal
                    sts = f['data'][ep]['datagen_info']['subtask_term_signals']['grasp']
                    transitions = np.where(np.diff(sts) == 1)[0]
                    if len(transitions) > 0:
                        valid_demo_keys.append(ep)
                    else:
                        bad_demo_count += 1
                except Exception:
                    bad_demo_count += 1
        logger.info("Filtered %d bad demos (no grasp transition), using %d valid demos",
                   bad_demo_count, len(valid_demo_keys))
    else:
        valid_demo_keys = all_demos

    return valid_demo_keys


def _worker_init(task_name, source_dataset_override=None, max_attempts=10):
    """Initialize worker with MimicGen environment, interface, and data generator."""
    global _worker_env, _worker_env_interface, _worker_data_generator, _worker_raw_env, _worker_task_name, _worker_max_attempts
    _worker_task_name = task_name
    _worker_max_attempts = max_attempts

    # Override source dataset if provided
    if source_dataset_override:
        TASK_REGISTRY[task_name]["source_dataset"] = source_dataset_override

    source_dataset = TASK_REGISTRY[task_name]["source_dataset"]
    if not os.path.exists(source_dataset):
        logger.error("Source dataset not found: %s", source_dataset)
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")

    # Load MimicGen config
    mg_config = _load_mg_config(task_name)

    # Get env metadata from source dataset
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset)

    # Create robomimic environment (no rendering needed for generation)
    _worker_env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )

    # Get raw robosuite env
    _worker_raw_env = getattr(_worker_env, "base_env", _worker_env.env)

    # Get all source demo keys
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset,
        filter_key=None,
        start=None,
        n=None,
    )

    # Filter demos with valid subtask signals
    valid_demos = _filter_valid_demos(source_dataset, all_demos, task_name)

    if not valid_demos:
        raise ValueError(f"No valid demos found for task {task_name}")

    # Get interface info from source dataset (use valid demos only)
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset,
        demo_keys=valid_demos,
    )

    # Create environment interface
    _worker_env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        env=_worker_raw_env,
    )

    # Load task spec from config
    task_spec_json = mg_config.task.task_spec.dump()
    task_spec = MG_TaskSpec.from_json(json_string=task_spec_json)

    # Create data generator (use valid demos only)
    _worker_data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset,
        demo_keys=valid_demos,
    )

    logger.info("Worker %d initialized: %d valid source demos for %s", os.getpid(), len(valid_demos), task_name)


def _inject_error_state(raw_env, post_sim_state: np.ndarray):
    """Inject error scene sim state into robosuite environment."""
    raw_env.sim.set_state_from_flattened(post_sim_state)
    raw_env.sim.forward()


def _generate_from_error_scene(
    post_sim_state: np.ndarray,
    max_attempts: int = 10,
):
    """
    Generate a success trajectory from an error scene state.

    Key approach: monkey-patch env.reset() so that MimicGen's DataGenerator.generate()
    starts from our injected error state instead of a random reset state.

    This ensures OSC controller is properly initialized via normal reset flow,
    then we inject the error state.

    Returns:
        (trajectory_dict, num_attempts) on success
        (None, max_attempts) on failure
    """
    original_reset = _worker_env.reset

    for attempt in range(1, max_attempts + 1):
        # Patch env.reset to inject our state after the normal reset
        def patched_reset():
            original_reset()
            _inject_error_state(_worker_raw_env, post_sim_state)

        _worker_env.reset = patched_reset

        try:
            result = _worker_data_generator.generate(
                env=_worker_env,
                env_interface=_worker_env_interface,
                select_src_per_subtask=True,  # Select different source demo per subtask
                transform_first_robot_pose=False,
                interpolate_from_last_target_pose=True,
                render=False,
                video_writer=None,
                video_skip=5,
                camera_names=None,
                pause_subtask=False,
            )
        except Exception as gen_exc:
            # MimicGen can throw various exceptions (IK failure, controller limits, etc.)
            logger.debug("Attempt %d exception: %s: %s", attempt, type(gen_exc).__name__, gen_exc)
            continue
        finally:
            _worker_env.reset = original_reset

        success = result.get("success", False)
        if success:
            logger.info("SUCCESS on attempt %d, traj_len=%d", attempt, len(result.get("actions", [])))
            return result, attempt

    return None, max_attempts


def _generate_one_scene(item):
    """Generate trajectory for one error scene."""
    scene_npz_path, scene_json_path, scene_id, subtype, attempt_idx = item

    try:
        scene_data = np.load(scene_npz_path, allow_pickle=True)

        if "post_sim_state" in scene_data:
            error_state = scene_data["post_sim_state"]
        elif "sim_state" in scene_data:
            error_state = scene_data["sim_state"]
        else:
            return {"scene_id": scene_id, "subtype": subtype, "attempt": attempt_idx, "error": "NO_SIM_STATE"}

        # Validate state dimension
        expected_len = len(_worker_raw_env.sim.get_state().flatten())
        if len(error_state) != expected_len:
            return {
                "scene_id": scene_id,
                "subtype": subtype,
                "attempt": attempt_idx,
                "error": f"STATE_DIM_MISMATCH_{len(error_state)}_vs_{expected_len}"
            }

        result, num_attempts = _generate_from_error_scene(error_state, max_attempts=_worker_max_attempts)

        if result is None or not result.get("success"):
            return {
                "scene_id": scene_id,
                "subtype": subtype,
                "attempt": attempt_idx,
                "error": "GENERATION_FAILED",
                "attempts": num_attempts,
            }

        # Extract trajectory data
        actions = np.array(result["actions"])
        states = result["states"]

        # Compute EEF positions from states
        eef_positions = []
        for state in states:
            _worker_raw_env.sim.set_state_from_flattened(state)
            _worker_raw_env.sim.forward()
            eef_pos = _worker_raw_env.robots[0].controller.ee_pos.copy()
            eef_positions.append(eef_pos)

        return {
            "scene_id": scene_id,
            "subtype": subtype,
            "attempt": attempt_idx,
            "success": True,
            "action_len": len(actions),
            "states": np.array([s.flatten() for s in states]),
            "actions": actions,
            "eef_positions": np.array(eef_positions),
            "num_attempts": num_attempts,
        }

    except Exception as e:
        import traceback
        return {
            "scene_id": scene_id,
            "subtype": subtype,
            "attempt": attempt_idx,
            "error": f"EXCEPTION: {e}",
            "traceback": traceback.format_exc()[:500],
        }


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories from error scenes via MimicGen")
    parser.add_argument("--task", required=True, choices=["stack", "pick_place", "coffee", "threading", "stack_three", "three_piece_assembly"])
    parser.add_argument("--subtype", default=None, help="Specific subtype to process")
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--dry-run", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=10, help="Max attempts per scene")
    parser.add_argument("--source-dataset", default=None, help="Override source dataset path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Source dataset
    source_dataset = args.source_dataset or TASK_REGISTRY[args.task]["source_dataset"]
    if not os.path.exists(source_dataset):
        logger.error("Source dataset not found: %s", source_dataset)
        sys.exit(1)

    logger.info("Using source dataset: %s", source_dataset)

    # Error scenes
    scenes_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "v5_training" / args.task / "scenes"
    output_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "mimicgen_generated" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect scenes
    scene_items = []
    for scene_file in sorted(scenes_dir.glob("*.json")):
        scene_meta = json.load(open(scene_file))
        error_spec = scene_meta.get("error_spec", {})
        subtype = f"{error_spec.get('error_name', '')}_{error_spec.get('degree', '')}"
        if not subtype or subtype == "_":
            continue
        if args.subtype and subtype != args.subtype:
            continue

        scene_id = scene_meta.get("scene_id", scene_file.stem)
        npz_path = scenes_dir / f"{scene_id}.npz"
        if npz_path.exists():
            scene_items.append((str(npz_path), str(scene_file), scene_id, subtype, 0))

    if args.dry_run:
        # Limit per subtype for dry run
        limited_items = []
        subtype_seen = defaultdict(int)
        for item in scene_items:
            if subtype_seen[item[3]] < args.dry_run:
                limited_items.append(item)
                subtype_seen[item[3]] += 1
        scene_items = limited_items
        logger.info("Dry run: %d scenes", len(scene_items))

    logger.info("Processing %d scenes for task %s", len(scene_items), args.task)

    # Run generation
    t0 = time.time()
    results = []

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.task, source_dataset, args.max_attempts),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(_generate_one_scene, scene_items)):
            results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(scene_items):
                success_count = sum(1 for r in results if r.get("success"))
                logger.info("  %d/%d processed, %d success (%.1f%%)",
                           i + 1, len(scene_items), success_count,
                           100 * success_count / (i + 1))

    elapsed = time.time() - t0
    success_results = [r for r in results if r.get("success")]
    logger.info("Done: %d success / %d failed in %.1fs (%.1f%% SR)",
               len(success_results), len(results) - len(success_results), elapsed,
               100 * len(success_results) / len(results) if results else 0)

    # Save successful trajectories
    for subtype in set(r["subtype"] for r in success_results):
        subtype_dir = output_dir / subtype
        subtype_dir.mkdir(parents=True, exist_ok=True)

        subtype_results = [r for r in success_results if r["subtype"] == subtype]
        for idx, result in enumerate(subtype_results[:50]):  # limit 50 per subtype
            npz_path = subtype_dir / f"mg_{result['scene_id']}_{idx:03d}.npz"
            np.savez_compressed(
                npz_path,
                states=result["states"],
                actions=result["actions"],
                eef_positions=result["eef_positions"],
                scene_id=result["scene_id"],
                subtype=result["subtype"],
                source="mimicgen_datagenerator",
            )

        logger.info("Saved %d to %s", len(subtype_results[:50]), subtype_dir)

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {args.task}")
    print("=" * 80)
    subtype_stats = defaultdict(lambda: {"success": 0, "failed": 0})
    for r in results:
        key = r["subtype"]
        if r.get("success"):
            subtype_stats[key]["success"] += 1
        else:
            subtype_stats[key]["failed"] += 1

    for st, counts in sorted(subtype_stats.items()):
        total = counts["success"] + counts["failed"]
        sr = counts["success"] / total * 100 if total > 0 else 0
        print(f"  {st:<35} success={counts['success']:>3} failed={counts['failed']:>3} SR={sr:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()