#!/usr/bin/env python
"""Convert success-from-error HDF5 trajectories to Pi0.5 LeRobot datasets.

Input:
    error_benchmark/outputs/mimicgen_success/<task>/success_demos.hdf5

Output:
    $HF_LEROBOT_HOME/benchmark/mimicgen_<task>_error_training

The converter replays MuJoCo states to render the two Pi0.5 camera streams,
recomputes the 8-dim Libero-style robot state, and writes LeRobot episodes.
It intentionally gates training use on validation, not just successful writes.
"""

import argparse
import json
import logging
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

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

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success"
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT / "error_benchmark" / "outputs" / "pi05_error_training_lerobot_reports"
)

_worker_env = None
_worker_tmp = None
_worker_hdf5_path = None
_worker_cfg = None
_worker_init_error = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _demo_sort_key(name: str):
    suffix = name.rsplit("_", 1)[-1]
    return (name.rsplit("_", 1)[0], int(suffix) if suffix.isdigit() else suffix)


def _quat_wxyz_to_axisangle(quat_wxyz):
    from scipy.spatial.transform import Rotation

    q_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(q_xyzw).as_rotvec()


def _quat_angle_error_xyzw(env_quat_wxyz, obs_quat_xyzw):
    from scipy.spatial.transform import Rotation

    env_xyzw = np.array(
        [env_quat_wxyz[1], env_quat_wxyz[2], env_quat_wxyz[3], env_quat_wxyz[0]],
        dtype=np.float64,
    )
    env_rot = Rotation.from_quat(env_xyzw)
    obs_rot = Rotation.from_quat(np.asarray(obs_quat_xyzw, dtype=np.float64))
    return float((env_rot.inv() * obs_rot).magnitude())


def _finite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def _load_task_config(task_name):
    task_info = load_task_registry(task_name)
    task_config_path = PROJECT_ROOT / task_info["task_config"]
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)
    return task_info, task_config


def _default_input_hdf5(task_name, source_root, input_name):
    return Path(source_root) / task_name / input_name


def repo_id_for_task(task_name, suffix):
    return f"benchmark/mimicgen_{task_name}_{suffix}"


def _select_action_replay_demos(demo_keys, sample_count, seed):
    if sample_count <= 0 or not demo_keys:
        return set()
    count = min(sample_count, len(demo_keys))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(demo_keys), size=count, replace=False)
    return {demo_keys[int(i)] for i in indices}


def _sample_action_steps(n_actions, max_steps, seed):
    if max_steps <= 0 or n_actions < 2:
        return []
    count = min(max_steps, n_actions - 1)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_actions - 1, size=count, replace=False)
    return sorted(int(i) for i in indices)


def _preflight_hdf5(hdf5_path, min_action_len):
    report = {
        "status": "PASS",
        "input_hdf5": str(hdf5_path),
        "total_demos": 0,
        "accepted_demos": 0,
        "total_frames": 0,
        "errors": [],
    }
    accepted = []
    required_obs = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

    with h5py.File(hdf5_path, "r") as h5_file:
        container = h5_file["data"] if "data" in h5_file else h5_file
        demo_keys = sorted(container.keys(), key=_demo_sort_key)
        report["total_demos"] = len(demo_keys)

        for demo_key in demo_keys:
            try:
                demo = container[demo_key]
                if "actions" not in demo or "states" not in demo or "obs" not in demo:
                    raise ValueError("missing actions/states/obs")
                actions = demo["actions"][()]
                states = demo["states"][()]
                obs = demo["obs"]

                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"actions shape must be (T,7), got {actions.shape}")
                if states.ndim != 2:
                    raise ValueError(f"states must be 2D, got {states.shape}")
                if len(actions) < min_action_len:
                    raise ValueError(
                        f"short demo: {len(actions)} actions < min_action_len={min_action_len}"
                    )
                if len(states) < len(actions):
                    raise ValueError(
                        f"states shorter than actions: {len(states)} < {len(actions)}"
                    )
                _finite("actions", actions)
                _finite("states", states)

                for key in required_obs:
                    if key not in obs:
                        raise ValueError(f"missing obs/{key}")
                    arr = obs[key][()]
                    if len(arr) < len(actions):
                        raise ValueError(
                            f"obs/{key} shorter than actions: {len(arr)} < {len(actions)}"
                        )
                    _finite(f"obs/{key}", arr[: len(actions)])

                accepted.append(demo_key)
                report["total_frames"] += int(len(actions))
            except Exception as exc:
                report["errors"].append(
                    {"demo": demo_key, "error": f"{type(exc).__name__}: {exc}"}
                )

    report["accepted_demos"] = len(accepted)
    if report["errors"] or not accepted:
        report["status"] = "FAIL"
    return accepted, report


def _worker_init(task_name, hdf5_path, tmp_dir, cfg):
    global _worker_env, _worker_tmp, _worker_hdf5_path, _worker_cfg, _worker_init_error
    _worker_tmp = tmp_dir
    _worker_hdf5_path = hdf5_path
    _worker_cfg = cfg
    _worker_init_error = None

    try:
        task_info, task_config = _load_task_config(task_name)
        env = create_env(
            task_config,
            task_info["dataset_path"],
            enable_camera=True,
            camera_resolution=cfg["camera_resolution"],
        )
        _worker_env = EnvWrapper(env, task_config)
    except Exception as exc:
        _worker_init_error = f"{type(exc).__name__}: {exc}"


def _load_demo_arrays(demo_key):
    with h5py.File(_worker_hdf5_path, "r") as h5_file:
        container = h5_file["data"] if "data" in h5_file else h5_file
        demo = container[demo_key]
        obs = demo["obs"]
        return {
            "actions": demo["actions"][()].astype(np.float32),
            "states": demo["states"][()].astype(np.float64),
            "eef_pos": obs["robot0_eef_pos"][()].astype(np.float64),
            "eef_quat": obs["robot0_eef_quat"][()].astype(np.float64),
            "gripper_qpos": obs["robot0_gripper_qpos"][()].astype(np.float64),
        }


def _state8_from_env():
    eef_pos = _worker_env.get_eef_pos()
    eef_aa = _quat_wxyz_to_axisangle(_worker_env.get_eef_quat())
    gripper = _worker_env.get_gripper_qpos_raw()
    if len(gripper) != 2:
        raise ValueError(f"expected 2 gripper qpos values, got {len(gripper)}")
    return np.concatenate([eef_pos, eef_aa, gripper]).astype(np.float32)


def _validate_action_steps(demo_key, arrays, demo_index):
    actions = arrays["actions"]
    states = arrays["states"]
    steps = _sample_action_steps(
        len(actions),
        _worker_cfg["action_replay_steps_per_demo"],
        _worker_cfg["seed"] + demo_index,
    )
    result = {
        "checked": 0,
        "failed": 0,
        "max_eef_err": 0.0,
        "max_qpos_rmse": 0.0,
        "examples": [],
    }

    for step_idx in steps:
        _worker_env.set_sim_state_flat(states[step_idx])
        _worker_env.step(actions[step_idx])
        pred_eef = _worker_env.get_eef_pos()
        pred_qpos = _worker_env.get_mj_data().qpos.copy()

        _worker_env.set_sim_state_flat(states[step_idx + 1])
        target_eef = _worker_env.get_eef_pos()
        target_qpos = _worker_env.get_mj_data().qpos.copy()

        eef_err = float(np.linalg.norm(pred_eef - target_eef))
        qpos_rmse = float(np.sqrt(np.mean(np.square(pred_qpos - target_qpos))))
        result["checked"] += 1
        result["max_eef_err"] = max(result["max_eef_err"], eef_err)
        result["max_qpos_rmse"] = max(result["max_qpos_rmse"], qpos_rmse)

        failed = (
            eef_err > _worker_cfg["action_eef_tolerance"]
            or qpos_rmse > _worker_cfg["action_qpos_rmse_tolerance"]
        )
        if failed:
            result["failed"] += 1
            if len(result["examples"]) < 5:
                result["examples"].append(
                    {
                        "demo": demo_key,
                        "step": step_idx,
                        "eef_err": eef_err,
                        "qpos_rmse": qpos_rmse,
                    }
                )
    return result


def _render_one_demo(item):
    demo_key, demo_index, validate_action = item
    if _worker_init_error is not None or _worker_env is None:
        return {"index": demo_index, "demo": demo_key, "error": f"WORKER_INIT: {_worker_init_error}"}
    try:
        arrays = _load_demo_arrays(demo_key)
        actions = arrays["actions"]
        states = arrays["states"]
        n_frames = len(actions)

        agent_imgs = np.empty((n_frames, 84, 84, 3), dtype=np.uint8)
        wrist_imgs = np.empty((n_frames, 84, 84, 3), dtype=np.uint8)
        state8 = np.empty((n_frames, 8), dtype=np.float32)

        max_eef_obs_err = 0.0
        max_quat_obs_err = 0.0
        max_gripper_obs_err = 0.0
        bad_image_frames = 0

        for frame_idx in range(n_frames):
            _worker_env.set_sim_state_flat(states[frame_idx])

            agent_img = _worker_env.get_camera_obs("agentview", (84, 84))
            wrist_img = _worker_env.get_camera_obs("robot0_eye_in_hand", (84, 84))
            agent_imgs[frame_idx] = agent_img
            wrist_imgs[frame_idx] = wrist_img
            state8[frame_idx] = _state8_from_env()

            if float(agent_img.std()) < _worker_cfg["image_min_std"]:
                bad_image_frames += 1
            if float(wrist_img.std()) < _worker_cfg["image_min_std"]:
                bad_image_frames += 1

            eef_obs_err = float(
                np.linalg.norm(_worker_env.get_eef_pos() - arrays["eef_pos"][frame_idx])
            )
            quat_obs_err = _quat_angle_error_xyzw(
                _worker_env.get_eef_quat(), arrays["eef_quat"][frame_idx]
            )
            gripper_obs_err = float(
                np.linalg.norm(
                    _worker_env.get_gripper_qpos_raw()
                    - arrays["gripper_qpos"][frame_idx]
                )
            )
            max_eef_obs_err = max(max_eef_obs_err, eef_obs_err)
            max_quat_obs_err = max(max_quat_obs_err, quat_obs_err)
            max_gripper_obs_err = max(max_gripper_obs_err, gripper_obs_err)

        action_result = {"checked": 0, "failed": 0}
        if validate_action:
            action_result = _validate_action_steps(demo_key, arrays, demo_index)

        failures = []
        if bad_image_frames:
            failures.append(f"{bad_image_frames} low-variance camera frames")
        if max_eef_obs_err > _worker_cfg["obs_eef_tolerance"]:
            failures.append(f"eef obs err {max_eef_obs_err:.6g}")
        if _worker_cfg.get("strict_obs_quat") and max_quat_obs_err > _worker_cfg["obs_quat_tolerance"]:
            failures.append(f"quat obs err {max_quat_obs_err:.6g}")
        if max_gripper_obs_err > _worker_cfg["obs_gripper_tolerance"]:
            failures.append(f"gripper obs err {max_gripper_obs_err:.6g}")
        if action_result.get("failed", 0):
            failures.append(f"{action_result['failed']} action replay samples failed")

        if failures:
            return {
                "index": demo_index,
                "demo": demo_key,
                "n_frames": n_frames,
                "error": "; ".join(failures),
                "action_replay": action_result,
                "state_obs": {
                    "max_eef_err": max_eef_obs_err,
                    "max_quat_angle": max_quat_obs_err,
                    "max_gripper_err": max_gripper_obs_err,
                },
            }

        tmp_name = f"{demo_index:06d}_{os.getpid()}.npz"
        tmp_path = os.path.join(_worker_tmp, tmp_name)
        np.savez_compressed(
            tmp_path,
            agent_imgs=agent_imgs,
            wrist_imgs=wrist_imgs,
            state8=state8,
            actions=actions.astype(np.float32),
        )

        return {
            "index": demo_index,
            "demo": demo_key,
            "n_frames": n_frames,
            "tmp_path": tmp_path,
            "action_replay": action_result,
            "state_obs": {
                "max_eef_err": max_eef_obs_err,
                "max_quat_angle": max_quat_obs_err,
                "max_gripper_err": max_gripper_obs_err,
            },
        }
    except Exception as exc:
        return {
            "index": demo_index,
            "demo": demo_key,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _postflight_lerobot(repo_id, expected_episodes, expected_frames):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    result = {"status": "PASS", "repo_id": repo_id, "errors": []}
    try:
        meta = LeRobotDatasetMetadata(repo_id)
        dataset = LeRobotDataset(repo_id)
        result["num_tasks"] = len(meta.tasks)
        result["num_frames"] = int(len(dataset))
        result["expected_frames"] = int(expected_frames)
        result["expected_episodes"] = int(expected_episodes)

        if len(dataset) != expected_frames:
            result["errors"].append(
                f"frame count mismatch: {len(dataset)} != {expected_frames}"
            )
        if len(meta.tasks) < 1:
            result["errors"].append("no LeRobot task metadata")

        sample_indices = sorted(set([0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)]))
        for idx in sample_indices:
            sample = dataset[idx]
            for key in ("image", "wrist_image", "state", "actions"):
                if key not in sample:
                    result["errors"].append(f"sample {idx} missing {key}")
            if "state" in sample and tuple(np.asarray(sample["state"]).shape)[-1:] != (8,):
                result["errors"].append(f"sample {idx} bad state shape")
            if "actions" in sample and tuple(np.asarray(sample["actions"]).shape)[-1:] != (7,):
                result["errors"].append(f"sample {idx} bad action shape")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    if result["errors"]:
        result["status"] = "FAIL"
    return result


def _write_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
        f.write("\n")
    os.replace(tmp, report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_PROMPTS))
    parser.add_argument("--input-hdf5", default=None)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--input-name", default="success_demos.hdf5")
    parser.add_argument("--repo-suffix", default="error_training")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--min-action-len", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-replay-demo-samples", type=int, default=32)
    parser.add_argument("--action-replay-steps-per-demo", type=int, default=3)
    parser.add_argument("--action-eef-tolerance", type=float, default=0.02)
    parser.add_argument("--action-qpos-rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--obs-eef-tolerance", type=float, default=0.005)
    parser.add_argument("--obs-quat-tolerance", type=float, default=0.02)
    parser.add_argument("--strict-obs-quat", action="store_true",
                        help="Fail if HDF5 obs/robot0_eef_quat disagrees with the replayed EnvWrapper quat")
    parser.add_argument("--obs-gripper-tolerance", type=float, default=0.005)
    parser.add_argument("--image-min-std", type=float, default=1.0)
    parser.add_argument("--image-writer-threads", type=int, default=16)
    parser.add_argument("--image-writer-processes", type=int, default=8)
    parser.add_argument("--lerobot-writer-python",
                        default=os.environ.get("LEROBOT_WRITER_PYTHON",
                                               "${CONDA_BASE}/envs/openpi05/bin/python"),
                        help="Python executable with lerobot installed; used only for writing cached frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    hdf5_path = (
        Path(args.input_hdf5)
        if args.input_hdf5
        else _default_input_hdf5(args.task, args.source_root, args.input_name)
    )
    repo_id = args.repo_id or repo_id_for_task(args.task, args.repo_suffix)
    output_path = Path(
        os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    ) / repo_id
    report_path = Path(args.report_root) / f"{args.task}_validation_report.json"

    report = {
        "status": "RUNNING",
        "task": args.task,
        "repo_id": repo_id,
        "output_path": str(output_path),
        "input_hdf5": str(hdf5_path),
        "validate_only": bool(args.validate_only),
        "allow_partial": bool(args.allow_partial),
        "thresholds": {
            "action_eef_tolerance": args.action_eef_tolerance,
            "action_qpos_rmse_tolerance": args.action_qpos_rmse_tolerance,
            "obs_eef_tolerance": args.obs_eef_tolerance,
            "obs_quat_tolerance": args.obs_quat_tolerance,
            "strict_obs_quat": args.strict_obs_quat,
            "obs_gripper_tolerance": args.obs_gripper_tolerance,
            "image_min_std": args.image_min_std,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    tmp_dir = None
    try:
        if not hdf5_path.exists():
            raise FileNotFoundError(f"input HDF5 not found: {hdf5_path}")

        logger.info("Preflight HDF5: %s", hdf5_path)
        demo_keys, preflight = _preflight_hdf5(hdf5_path, args.min_action_len)
        if args.dry_run:
            demo_keys = demo_keys[: args.dry_run]
            preflight["accepted_demos_after_dry_run"] = len(demo_keys)
            preflight["total_frames_after_dry_run"] = None
        report["preflight"] = preflight
        if preflight["status"] != "PASS" and not args.allow_partial:
            raise RuntimeError("HDF5 preflight failed")
        if not demo_keys:
            raise RuntimeError("no accepted demos to convert")

        action_demo_keys = _select_action_replay_demos(
            demo_keys, args.action_replay_demo_samples, args.seed
        )
        items = [
            (demo_key, i, demo_key in action_demo_keys)
            for i, demo_key in enumerate(demo_keys)
        ]

        worker_cfg = {
            "camera_resolution": 84,
            "seed": args.seed,
            "action_replay_steps_per_demo": args.action_replay_steps_per_demo,
            "action_eef_tolerance": args.action_eef_tolerance,
            "action_qpos_rmse_tolerance": args.action_qpos_rmse_tolerance,
            "obs_eef_tolerance": args.obs_eef_tolerance,
            "obs_quat_tolerance": args.obs_quat_tolerance,
            "strict_obs_quat": args.strict_obs_quat,
            "obs_gripper_tolerance": args.obs_gripper_tolerance,
            "image_min_std": args.image_min_std,
        }

        tmp_dir = tempfile.mkdtemp(prefix=f"pi05_error_training_{args.task}_")
        logger.info(
            "Rendering/validating %d demos with %d workers; temp=%s",
            len(items),
            args.num_workers,
            tmp_dir,
        )
        t0 = time.time()
        meta_results = []
        with multiprocessing.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(args.task, str(hdf5_path), tmp_dir, worker_cfg),
        ) as pool:
            for count, meta in enumerate(pool.imap_unordered(_render_one_demo, items), 1):
                meta_results.append(meta)
                if count % 25 == 0 or count == len(items):
                    ok = sum(1 for m in meta_results if "error" not in m)
                    elapsed = max(time.time() - t0, 1e-6)
                    logger.info(
                        "  %d/%d processed (%d ok, %d err) %.2f demos/s",
                        count,
                        len(items),
                        ok,
                        count - ok,
                        count / elapsed,
                    )

        ok_results = [m for m in meta_results if "error" not in m]
        err_results = [m for m in meta_results if "error" in m]
        ok_results.sort(key=lambda m: m["index"])
        total_frames = sum(int(m["n_frames"]) for m in ok_results)
        action_checked = sum(int(m.get("action_replay", {}).get("checked", 0)) for m in meta_results)
        action_failed = sum(int(m.get("action_replay", {}).get("failed", 0)) for m in meta_results)
        report["render_validation"] = {
            "status": "PASS" if not err_results else "FAIL",
            "processed_demos": len(meta_results),
            "ok_demos": len(ok_results),
            "error_demos": len(err_results),
            "total_frames": total_frames,
            "action_replay_checked": action_checked,
            "action_replay_failed": action_failed,
            "errors": err_results[:50],
        }

        if err_results and not args.allow_partial:
            raise RuntimeError(f"render/action validation failed for {len(err_results)} demos")
        if not ok_results:
            raise RuntimeError("no validated demos remain")

        if args.validate_only:
            report["postflight"] = {"status": "SKIPPED_VALIDATE_ONLY"}
            report["status"] = "DRY_RUN_PASS" if args.dry_run else "VALIDATE_ONLY_PASS"
            logger.info("Validate-only complete: %s", report["status"])
            return

        task_prompt = TASK_PROMPTS[args.task]
        manifest_path = Path(tmp_dir) / "_lerobot_manifest.json"
        writer_report_path = Path(tmp_dir) / "_lerobot_writer_report.json"
        manifest = {
            "repo_id": repo_id,
            "task": args.task,
            "task_prompt": task_prompt,
            "total_frames": total_frames,
            "image_writer_threads": args.image_writer_threads,
            "image_writer_processes": args.image_writer_processes,
            "episodes": [
                {
                    "index": int(meta["index"]),
                    "demo": meta["demo"],
                    "n_frames": int(meta["n_frames"]),
                    "tmp_path": meta["tmp_path"],
                }
                for meta in ok_results
            ],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        writer_script = PROJECT_ROOT / "error_benchmark" / "scripts" / "conversion" / "4e_write_pi05_lerobot_from_cache.py"
        cmd = [
            args.lerobot_writer_python,
            str(writer_script),
            "--manifest",
            str(manifest_path),
            "--report",
            str(writer_report_path),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        logger.info("Writing LeRobot with %s", args.lerobot_writer_python)
        subprocess.run(cmd, check=True)

        with open(writer_report_path) as f:
            report["postflight"] = json.load(f)
        if report["postflight"]["status"] != "PASS":
            raise RuntimeError("LeRobot postflight failed")
        report["status"] = "DRY_RUN_PASS" if args.dry_run else "PASS"
        logger.info("Complete: PASS %s", report_path)
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("FAILED: %s", report["error"])
        raise SystemExit(2) from exc
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_report(report, report_path)


if __name__ == "__main__":
    main()
