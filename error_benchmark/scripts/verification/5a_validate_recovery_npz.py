#!/usr/bin/env python
"""Validate recovery NPZ files via state replay and action replay.

For every human demo and augmented demo NPZ, verifies:
1. State replay: set_sim_state_flat(states[i]) produces correct EEF position
2. Action replay: states[0] + actions reproduces the trajectory

Usage:
    python error_benchmark/scripts/verification/5a_validate_recovery_npz.py --task coffee
    python error_benchmark/scripts/verification/5a_validate_recovery_npz.py --task stack --source augmented
    python error_benchmark/scripts/verification/5a_validate_recovery_npz.py --task coffee --dry-run 20
"""

import argparse
import csv
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

import yaml

from error_benchmark.framework.env_wrapper import EnvWrapper
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry

logger = logging.getLogger(__name__)

# Per-worker state (initialized once in each worker process)
_worker_env = None


def _worker_init(task_name):
    global _worker_env
    task_info = load_task_registry(task_name)
    task_config_path = os.path.join(str(PROJECT_ROOT), task_info["task_config"])
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)
    dataset_path = task_info["dataset_path"]

    env = create_env(task_config, dataset_path, enable_camera=False)
    _worker_env = EnvWrapper(env, task_config)


def _validate_one(item):
    npz_path, source_type = item
    result = {
        "file": os.path.basename(npz_path),
        "subtype": _extract_subtype(npz_path),
        "source": source_type,
    }

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        result["error"] = f"LOAD_ERROR: {e}"
        return result

    if "actions" not in data or "states" not in data:
        result["error"] = "MISSING_ACTIONS_OR_STATES"
        return result

    actions = data["actions"]
    states = data["states"]
    eef_positions = data.get("eef_positions", None)
    result["action_len"] = len(actions)

    if len(actions) < 1:
        result["error"] = "EMPTY_ACTIONS"
        return result

    # --- State replay: sample every Nth frame (max ~50 samples) ---
    n_states = len(states)
    skip = max(1, n_states // 50)
    state_errors = []
    state_load_ok = True
    first_error_msg = None

    for i in range(0, n_states, skip):
        try:
            _worker_env.set_sim_state_flat(states[i])
            if eef_positions is not None and i < len(eef_positions):
                err = np.linalg.norm(_worker_env.get_eef_pos() - eef_positions[i])
                state_errors.append(err)
        except Exception as e:
            state_load_ok = False
            if first_error_msg is None:
                first_error_msg = f"frame {i}: {e}"

    result["state_replay_ok"] = state_load_ok
    result["max_state_err"] = float(max(state_errors)) if state_errors else float("nan")
    result["mean_state_err"] = float(np.mean(state_errors)) if state_errors else float("nan")
    if first_error_msg:
        result["state_error_msg"] = first_error_msg

    # --- Action replay: states[0] + all actions → compare final EEF ---
    try:
        _worker_env.set_sim_state_flat(states[0])
        for i in range(len(actions)):
            _worker_env.step(actions[i])
        actual_final = _worker_env.get_eef_pos()
        if eef_positions is not None and len(eef_positions) > 0:
            final_err = float(np.linalg.norm(actual_final - eef_positions[-1]))
        else:
            final_err = float("nan")
        result["action_replay_ok"] = True
        result["final_eef_err"] = final_err
    except Exception as e:
        result["action_replay_ok"] = False
        result["final_eef_err"] = float("nan")
        result["action_replay_error"] = str(e)

    return result


def _extract_subtype(path):
    parts = path.split("/")
    for i, p in enumerate(parts):
        if p in ("demos", "augmented") and i + 2 < len(parts):
            return parts[i + 2]
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Validate recovery NPZ state/action replay")
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", choices=["human", "augmented", "all"], default="all")
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--max-eef-error", type=float, default=0.01,
                        help="State replay pass threshold (meters)")
    parser.add_argument("--max-final-error", type=float, default=0.05,
                        help="Action replay final EEF threshold (meters)")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Limit to first N files per source")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Collect NPZ files
    npz_files = []
    recovery_root = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery"

    if args.source in ("human", "all"):
        human_dir = recovery_root / "demos" / args.task
        if human_dir.exists():
            for npz in sorted(human_dir.rglob("recovery_*.npz")):
                npz_files.append((str(npz), "human"))

    if args.source in ("augmented", "all"):
        aug_dir = recovery_root / "augmented" / args.task
        if aug_dir.exists():
            for npz in sorted(aug_dir.rglob("aug_*.npz")):
                npz_files.append((str(npz), "augmented"))

    n_human = sum(1 for _, s in npz_files if s == "human")
    n_aug = sum(1 for _, s in npz_files if s == "augmented")
    logger.info(f"Found {n_human} human + {n_aug} augmented = {len(npz_files)} total")

    if not npz_files:
        logger.error(f"No NPZ files for task={args.task} source={args.source}")
        sys.exit(1)

    if args.dry_run:
        human = [f for f in npz_files if f[1] == "human"][: args.dry_run]
        aug = [f for f in npz_files if f[1] == "augmented"][: args.dry_run]
        npz_files = human + aug
        logger.info(f"Dry run: {len(npz_files)} files")

    # Run
    t0 = time.time()
    results = []
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.task,),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_validate_one, npz_files)):
            results.append(r)
            if (i + 1) % 200 == 0 or (i + 1) == len(npz_files):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(npz_files) - i - 1) / rate if rate > 0 else 0
                logger.info(f"  {i+1}/{len(npz_files)}  {rate:.1f} files/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(results)} files in {elapsed:.1f}s ({len(results)/elapsed:.1f} files/s)")

    # CSV
    out_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.task}_replay_report.csv"

    fields = [
        "file", "subtype", "source", "action_len",
        "state_replay_ok", "max_state_err", "mean_state_err",
        "action_replay_ok", "final_eef_err",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    logger.info(f"Report: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {args.task}")
    print("=" * 80)

    for src in ("human", "augmented"):
        src_r = [r for r in results if r.get("source") == src]
        if not src_r:
            continue
        load_ok = sum(1 for r in src_r if r.get("state_replay_ok"))
        state_pass = sum(
            1 for r in src_r
            if r.get("state_replay_ok")
            and r.get("max_state_err", 999) <= args.max_eef_error
        )
        act_ok = sum(1 for r in src_r if r.get("action_replay_ok"))
        act_pass = sum(
            1 for r in src_r
            if r.get("action_replay_ok")
            and r.get("final_eef_err", 999) <= args.max_final_error
        )
        print(f"\n  {src.upper()} ({len(src_r)} files):")
        print(f"    State replay loadable:      {load_ok}/{len(src_r)}")
        print(f"    State replay pass (<={args.max_eef_error}m):  {state_pass}/{len(src_r)}")
        print(f"    Action replay ok:           {act_ok}/{len(src_r)}")
        print(f"    Action replay pass (<={args.max_final_error}m): {act_pass}/{len(src_r)}")

    # Worst subtypes
    print(f"\n  WORST SUBTYPES (by mean state error):")
    sub_errs = defaultdict(list)
    for r in results:
        if r.get("state_replay_ok") and not np.isnan(r.get("mean_state_err", float("nan"))):
            sub_errs[r.get("subtype", "?")].append(r["mean_state_err"])
    for st, errs in sorted(sub_errs.items(), key=lambda x: -np.mean(x[1]))[:10]:
        print(f"    {st:<35} mean={np.mean(errs):.6f}m  max={max(errs):.6f}m  n={len(errs)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
