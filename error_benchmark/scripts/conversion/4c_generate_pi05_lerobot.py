#!/usr/bin/env python
"""Convert recovery NPZ files to LeRoBot dataset for pi0.5 fine-tuning.

Replays sim_state through MuJoCo to render dual-camera (agentview + wrist)
84x84 images, computes 8-dim state, and writes to LeRoBot format.

Usage:
    python error_benchmark/scripts/conversion/4c_generate_pi05_lerobot.py --task coffee
    python error_benchmark/scripts/conversion/4c_generate_pi05_lerobot.py --task stack --num-workers 32
    python error_benchmark/scripts/conversion/4c_generate_pi05_lerobot.py --task coffee --include-original
"""

import argparse
import logging
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

import yaml

from error_benchmark.framework.env_wrapper import EnvWrapper
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry

logger = logging.getLogger(__name__)

TASK_PROMPTS = {
    "pick_place": "pick up the milk, cereal, bread, and can and place them in the correct bins",
    "stack": "stack the red block on top of the green block",
    "coffee": "make coffee",
    "threading": "insert the needle into the needle hole",
    "stack_three": "stack three blocks",
    "three_piece_assembly": "assemble the three pieces",
}

# ── helpers ──────────────────────────────────────────────────────────


def _quat_wxyz_to_axisangle(quat_wxyz):
    """Convert quaternion (w,x,y,z) to axis-angle (3D)."""
    # scipy expects (x,y,z,w)
    from scipy.spatial.transform import Rotation
    q_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(q_xyzw).as_rotvec()


# ── worker ───────────────────────────────────────────────────────────

_worker_env = None
_worker_tmp = None


def _worker_init(task_name, tmp_dir):
    global _worker_env, _worker_tmp
    _worker_tmp = tmp_dir

    task_info = load_task_registry(task_name)
    task_config_path = os.path.join(str(PROJECT_ROOT), task_info["task_config"])
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)
    dataset_path = task_info["dataset_path"]

    env = create_env(task_config, dataset_path, enable_camera=True, camera_resolution=84)
    _worker_env = EnvWrapper(env, task_config)
    logger.info("Worker %d ready (state_dim=%d)", os.getpid(), _worker_env._expected_state_len)


def _render_one_npz(item):
    """Render a single NPZ file. Saves frames to a temp .npz, returns metadata."""
    npz_path, source_type, file_index = item
    basename = os.path.basename(npz_path)

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return {"index": file_index, "file": basename, "source": source_type, "error": f"LOAD: {e}"}

    if "actions" not in data or "states" not in data:
        return {"index": file_index, "file": basename, "source": source_type, "error": "NO_ACTIONS/STATES"}

    actions = data["actions"]
    states = data["states"]
    n_frames = len(actions)

    if n_frames < 1:
        return {"index": file_index, "file": basename, "source": source_type, "error": "EMPTY"}

    agent_imgs = np.empty((n_frames, 84, 84, 3), dtype=np.uint8)
    wrist_imgs = np.empty((n_frames, 84, 84, 3), dtype=np.uint8)
    state8 = np.empty((n_frames, 8), dtype=np.float32)

    for i in range(n_frames):
        _worker_env.set_sim_state_flat(states[i])

        # Render both cameras
        agent_imgs[i] = _worker_env.get_camera_obs("agentview", (84, 84))
        wrist_imgs[i] = _worker_env.get_camera_obs("robot0_eye_in_hand", (84, 84))

        # Compute 8-dim state
        eef_pos = _worker_env.get_eef_pos()
        eef_quat_wxyz = _worker_env.get_eef_quat()
        eef_aa = _quat_wxyz_to_axisangle(eef_quat_wxyz)
        gripper = _worker_env.get_gripper_qpos_raw()
        state8[i] = np.concatenate([eef_pos, eef_aa, gripper]).astype(np.float32)

    # Save to temp file
    tmp_name = f"{file_index:06d}_{os.getpid()}.npz"
    tmp_path = os.path.join(_worker_tmp, tmp_name)
    np.savez_compressed(tmp_path,
                        agent_imgs=agent_imgs,
                        wrist_imgs=wrist_imgs,
                        state8=state8,
                        actions=actions.astype(np.float32))

    return {
        "index": file_index,
        "file": basename,
        "source": source_type,
        "n_frames": n_frames,
        "tmp_path": tmp_path,
    }


# ── main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert recovery NPZ → LeRoBot for pi0.5")
    parser.add_argument("--task", required=True)
    parser.add_argument("--include-recovery-human", action="store_true", default=True)
    parser.add_argument("--no-recovery-human", dest="include_recovery_human", action="store_false")
    parser.add_argument("--include-recovery-augmented", action="store_true", default=True)
    parser.add_argument("--no-recovery-augmented", dest="include_recovery_augmented", action="store_false")
    parser.add_argument("--include-original", action="store_true", default=False,
                        help="Include original HDF5 demos (images already present)")
    parser.add_argument("--min-action-len", type=int, default=30,
                        help="Skip demos shorter than this")
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Limit to first N NPZ files per source")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    task_prompt = TASK_PROMPTS.get(args.task, f"complete the {args.task} task")
    repo_id = f"benchmark/mimicgen_{args.task}_recovery_merged"
    output_path = Path(os.environ.get(
        "HF_LEROBOT_HOME",
        os.path.expanduser("~/.cache/huggingface/lerobot"),
    )) / repo_id

    # Check existing
    if output_path.exists():
        if args.overwrite:
            logger.info("Removing existing dataset at %s", output_path)
            shutil.rmtree(output_path)
        else:
            logger.error("Dataset exists at %s — use --overwrite", output_path)
            sys.exit(1)

    # Collect NPZ files
    npz_items = []
    recovery_root = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery"

    if args.include_recovery_human:
        human_dir = recovery_root / "demos" / args.task
        if human_dir.exists():
            for npz in sorted(human_dir.rglob("recovery_*.npz")):
                npz_items.append((str(npz), "human"))

    if args.include_recovery_augmented:
        aug_dir = recovery_root / "augmented" / args.task
        if aug_dir.exists():
            for npz in sorted(aug_dir.rglob("aug_*.npz")):
                npz_items.append((str(npz), "augmented"))

    # Filter by min-action-len (cheap: load header only)
    filtered = []
    for path, src in npz_items:
        try:
            d = np.load(path, allow_pickle=True)
            if "actions" in d and len(d["actions"]) >= args.min_action_len:
                filtered.append((path, src))
        except Exception:
            pass
    skipped = len(npz_items) - len(filtered)
    npz_items = filtered
    logger.info(
        "After min-action-len=%d filter: %d files (%d skipped)",
        args.min_action_len, len(npz_items), skipped,
    )

    if args.dry_run:
        npz_items = npz_items[: args.dry_run]
        logger.info("Dry run: limited to %d files", len(npz_items))

    if not npz_items:
        logger.error("No NPZ files to convert")
        sys.exit(1)

    # Assign indices
    npz_items = [(p, s, i) for i, (p, s) in enumerate(npz_items)]

    # Create temp dir for worker outputs
    tmp_dir = tempfile.mkdtemp(prefix=f"pi05_{args.task}_")
    logger.info("Temp dir: %s", tmp_dir)

    # Phase 1: Render (parallel)
    logger.info("=== Phase 1: Rendering %d NPZ files with %d workers ===", len(npz_items), args.num_workers)
    t0 = time.time()
    meta_results = []

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.task, tmp_dir),
    ) as pool:
        for i, meta in enumerate(pool.imap_unordered(_render_one_npz, npz_items)):
            meta_results.append(meta)
            if (i + 1) % 50 == 0 or (i + 1) == len(npz_items):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(npz_items) - i - 1) / rate if rate > 0 else 0
                ok = sum(1 for m in meta_results if "error" not in m)
                logger.info(
                    "  %d/%d rendered (%d ok, %d err)  %.1f files/s  ETA %.0fs",
                    i + 1, len(npz_items), ok, (i + 1) - ok, rate, eta,
                )

    render_elapsed = time.time() - t0
    ok_results = [m for m in meta_results if "error" not in m]
    err_results = [m for m in meta_results if "error" in m]
    total_frames = sum(m["n_frames"] for m in ok_results)
    logger.info(
        "Rendering done: %d ok / %d err, %d total frames in %.1fs",
        len(ok_results), len(err_results), total_frames, render_elapsed,
    )

    if err_results:
        err_file = os.path.join(tmp_dir, "_errors.txt")
        with open(err_file, "w") as f:
            for e in err_results:
                f.write(f"{e['file']}: {e.get('error', '?')}\n")
        logger.warning("Errors logged to %s", err_file)

    # Phase 2: Write LeRoBot dataset (single-threaded)
    logger.info("=== Phase 2: Writing LeRoBot dataset ===")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=20,
        features={
            "image": {"dtype": "image", "shape": (84, 84, 3),
                      "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (84, 84, 3),
                            "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=16,
        image_writer_processes=8,
    )

    # Sort by index for deterministic order
    ok_results.sort(key=lambda m: m["index"])

    t1 = time.time()
    for i, meta in enumerate(ok_results):
        tmp_data = np.load(meta["tmp_path"])
        agent_imgs = tmp_data["agent_imgs"]
        wrist_imgs = tmp_data["wrist_imgs"]
        state8 = tmp_data["state8"]
        actions = tmp_data["actions"]

        for j in range(meta["n_frames"]):
            dataset.add_frame({
                "image": agent_imgs[j],
                "wrist_image": wrist_imgs[j],
                "state": state8[j],
                "actions": actions[j],
                "task": task_prompt,
            })
        dataset.save_episode()

        if (i + 1) % 100 == 0 or (i + 1) == len(ok_results):
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            logger.info("  %d/%d episodes written  %.1f ep/s", i + 1, len(ok_results), rate)

    write_elapsed = time.time() - t1
    logger.info("LeRoBot write done: %d episodes in %.1fs", len(ok_results), write_elapsed)

    # Handle original HDF5 (if --include-original)
    if args.include_original:
        _convert_original_hdf5(args.task, dataset, task_prompt)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Temp dir cleaned up")

    total_elapsed = time.time() - t0
    logger.info(
        "=== COMPLETE: %d episodes, %d frames, %.1fs total ===",
        len(ok_results), total_frames, total_elapsed,
    )


def _convert_original_hdf5(task_name, dataset, task_prompt):
    """Append original HDF5 demos (images already present) to the LeRoBot dataset."""
    import h5py

    task_info = load_task_registry(task_name)
    dataset_path = task_info["dataset_path"]
    if not dataset_path or not os.path.isfile(dataset_path):
        logger.warning("Original HDF5 not found at %s, skipping", dataset_path)
        return

    logger.info("Appending original HDF5 demos from %s", dataset_path)
    with h5py.File(dataset_path, "r") as ff:
        container = ff["data"] if "data" in ff else ff
        demo_keys = sorted(container.keys())
        count = 0

        for demo_key in demo_keys:
            demo = container[demo_key]
            if "actions" not in demo or "obs" not in demo:
                continue

            obs = demo["obs"]
            if "agentview_image" not in obs or "robot0_eye_in_hand_image" not in obs:
                continue

            actions = demo["actions"][()]
            eef_pos = obs["robot0_eef_pos"][()]
            eef_quat = obs["robot0_eef_quat"][()]
            gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros((len(eef_pos), 2)))[()]

            # Convert (x,y,z,w) quat → axis-angle
            from scipy.spatial.transform import Rotation
            eef_aa = np.array([Rotation.from_quat(q).as_rotvec() for q in eef_quat])
            states = np.hstack((eef_pos, eef_aa, gripper_qpos))

            agent_imgs = obs["agentview_image"][()]
            wrist_imgs = obs["robot0_eye_in_hand_image"][()]

            data_len = min(len(actions), len(states), len(agent_imgs), len(wrist_imgs))

            for i in range(data_len):
                dataset.add_frame({
                    "image": agent_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state": states[i].astype(np.float32),
                    "actions": actions[i].astype(np.float32),
                    "task": task_prompt,
                })
            dataset.save_episode()
            count += 1

            if count % 100 == 0:
                logger.info("  %d original demos converted", count)

    logger.info("Original HDF5: %d demos appended", count)


if __name__ == "__main__":
    main()
