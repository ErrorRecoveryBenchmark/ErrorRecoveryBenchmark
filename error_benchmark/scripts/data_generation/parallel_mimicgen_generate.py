#!/usr/bin/env python3
"""
Parallel MimicGen dataset generation with N workers.

Spawns N independent MimicGen generation processes, each with a different seed
and target demo count, then merges all outputs into a single HDF5 file.

Usage:
    # Generate pick_place_d0 with 96 workers
    MUJOCO_GL=egl python parallel_mimicgen_generate.py \
        --task pick_place_d0 --num_demos 1000 --num_workers 96 \
        --output /HOME/.../mimicgen_datasets/core/pick_place_d0.hdf5

    # Generate threading_d1 with 96 workers
    MUJOCO_GL=egl python parallel_mimicgen_generate.py \
        --task threading_d1 --num_demos 1000 --num_workers 96 \
        --output /HOME/.../mimicgen_datasets/core/threading_d1.hdf5
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
MIMICGEN_ROOT = PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"
GENERATE_SCRIPT = MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"
SOURCE_DIR = MIMICGEN_ROOT / "datasets" / "source"
DATASET_DIR = Path(
    os.environ.get("MIMICGEN_DATASET_CORE_DIR")
    or PROJECT_DIR.parent / "mimicgen_datasets" / "core"
)
TEMPLATE_DIR = MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robosuite"

# Python executable for mimicgen_env (override via MIMICGEN_PYTHON env var)
MIMICGEN_PYTHON = Path(
    os.environ.get("MIMICGEN_PYTHON")
    or Path(sys.executable)
)

# ═══════════════════════════════════════════════════════════════
# Task definitions
# ═══════════════════════════════════════════════════════════════

TASK_CONFIGS = {
    "pick_place_d0": {
        "base_config": DATASET_DIR / "pick_place_d0" / "mg_config.json",
        "task_name": "PickPlace_D0",
        "exp_name": "pick_place_d0",
    },
    "threading_d0": {
        "template": TEMPLATE_DIR / "threading.json",
        "task_name": "Threading_D0",
        "source": SOURCE_DIR / "threading.hdf5",
        "exp_name": "threading_d0",
        "cameras": ["agentview", "robot0_eye_in_hand"],
    },
    "threading_d1": {
        "template": TEMPLATE_DIR / "threading.json",
        "task_name": "Threading_D1",
        "source": SOURCE_DIR / "threading.hdf5",
        "exp_name": "threading_d1",
        "cameras": ["agentview", "robot0_eye_in_hand"],
    },
}


# ═══════════════════════════════════════════════════════════════
# Config generation
# ═══════════════════════════════════════════════════════════════

def create_worker_config(task_key: str, worker_id: int, num_demos: int,
                         work_dir: str, seed_offset: int = 0) -> tuple:
    """Create a MimicGen config JSON for one worker.

    Returns (config_path, expected_output_dir).
    """
    task = TASK_CONFIGS[task_key]
    worker_name = f"w{worker_id:04d}"

    # Load base config
    if "base_config" in task:
        with open(task["base_config"]) as f:
            config = json.load(f)
    else:
        with open(task["template"]) as f:
            config = json.load(f)
        config["experiment"]["source"]["dataset_path"] = str(task["source"])
        config["obs"]["camera_names"] = task.get("cameras", [])
        config["experiment"]["generation"]["keep_failed"] = False

    # Per-worker overrides
    worker_gen_dir = os.path.join(work_dir, worker_name)
    exp_name = f"gen_{worker_name}"

    config["experiment"]["name"] = exp_name
    config["experiment"]["generation"]["path"] = worker_gen_dir
    config["experiment"]["generation"]["num_trials"] = num_demos
    config["experiment"]["generation"]["guarantee"] = True
    config["experiment"]["seed"] = 42 + seed_offset + worker_id * 997  # spread seeds
    config["experiment"]["render_video"] = False
    config["experiment"]["max_num_failures"] = max(num_demos * 200, 5000)
    config["experiment"]["log_every_n_attempts"] = max(num_demos, 10)

    # Task name override
    config["experiment"]["task"]["name"] = task["task_name"]

    # Write config
    os.makedirs(worker_gen_dir, exist_ok=True)
    config_path = os.path.join(worker_gen_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    expected_output = os.path.join(worker_gen_dir, exp_name, "demo.hdf5")
    return config_path, expected_output


# ═══════════════════════════════════════════════════════════════
# Worker execution
# ═══════════════════════════════════════════════════════════════

def _run_single_worker(args):
    """Run one MimicGen generation process (called via ProcessPoolExecutor)."""
    config_path, worker_id, python_exec, worker_timeout = args

    env = os.environ.copy()
    env["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    # Ensure robosuite/mimicgen are importable
    extra_paths = [
        str(MIMICGEN_ROOT.parent / "robosuite"),
        str(MIMICGEN_ROOT),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

    cmd = [
        str(python_exec),
        str(GENERATE_SCRIPT),
        "--config", config_path,
        "--auto-remove-exp",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=worker_timeout,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            # Extract last few lines of stderr for diagnostics
            err_tail = result.stderr[-800:] if result.stderr else "(no stderr)"
            out_tail = result.stdout[-800:] if result.stdout else "(no stdout)"
            print(f"[Worker {worker_id:4d}] FAILED after {elapsed:.0f}s\n"
                  f"  stderr: {err_tail}\n  stdout: {out_tail}")
            return worker_id, False
        print(f"[Worker {worker_id:4d}] Done in {elapsed:.0f}s")
        return worker_id, True
    except subprocess.TimeoutExpired:
        print(f"[Worker {worker_id:4d}] TIMEOUT ({worker_timeout}s)")
        return worker_id, False
    except Exception as e:
        print(f"[Worker {worker_id:4d}] EXCEPTION: {e}")
        return worker_id, False


# ═══════════════════════════════════════════════════════════════
# Merge
# ═══════════════════════════════════════════════════════════════

def merge_worker_hdf5s(demo_paths: list, output_path: str):
    """Merge multiple demo.hdf5 files (each with N episodes) into one."""
    print(f"\nMerging {len(demo_paths)} worker outputs → {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as out_f:
        data_grp = out_f.create_group("data")

        total_samples = 0
        demo_idx = 0
        env_args = None

        for demo_path in sorted(demo_paths):
            try:
                with h5py.File(demo_path, "r") as src_f:
                    if "data" not in src_f:
                        continue

                    src_data = src_f["data"]

                    # Capture env_args from first valid file
                    if env_args is None and "env_args" in src_data.attrs:
                        env_args = src_data.attrs["env_args"]

                    # Copy each episode with renumbered keys
                    ep_keys = sorted(
                        [k for k in src_data.keys() if k.startswith("demo_")],
                        key=lambda x: int(x.split("_")[1]),
                    )
                    for ep_key in ep_keys:
                        new_key = f"demo_{demo_idx}"
                        src_data.copy(ep_key, data_grp, name=new_key)
                        ns = data_grp[new_key].attrs.get("num_samples", 0)
                        total_samples += ns
                        demo_idx += 1
            except Exception as e:
                print(f"  WARNING: cannot open {demo_path}: {e}")
                continue

        data_grp.attrs["total"] = total_samples
        if env_args is not None:
            data_grp.attrs["env_args"] = env_args

    print(f"  Merged {demo_idx} episodes, {total_samples} total samples → {output_path}")
    return demo_idx


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parallel MimicGen dataset generation"
    )
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()),
                        help="Task variant to generate")
    parser.add_argument("--num_demos", type=int, default=1000,
                        help="Total number of successful demos to generate")
    parser.add_argument("--num_workers", type=int, default=96,
                        help="Number of parallel worker processes")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for the final merged HDF5 file")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Temporary working directory (default: <output_dir>/.parallel_gen_<task>)")
    parser.add_argument("--keep_work_dir", action="store_true",
                        help="Do not delete the working directory after merge")
    parser.add_argument("--python", type=str, default=str(MIMICGEN_PYTHON),
                        help="Python executable (in mimicgen_env)")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Offset added to the per-worker seed. Use a large value "
                             "(e.g. 100000) when generating additional demos that must "
                             "not overlap with a previous run that used seed_offset=0.")
    parser.add_argument(
        "--worker_timeout",
        type=int,
        default=int(os.environ.get("PARALLEL_MIMICGEN_WORKER_TIMEOUT", "7200")),
        help="Per-worker timeout in seconds (default: env PARALLEL_MIMICGEN_WORKER_TIMEOUT or 7200).",
    )
    args = parser.parse_args()

    task_key = args.task
    num_demos = args.num_demos
    num_workers = min(args.num_workers, num_demos)  # no more workers than demos
    output_path = os.path.abspath(args.output)

    # Working directory
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.path.join(
            os.path.dirname(output_path),
            f".parallel_gen_{task_key}",
        )

    print("=" * 60)
    print(f"Parallel MimicGen Generation")
    print(f"  Task:       {task_key}")
    print(f"  Demos:      {num_demos}")
    print(f"  Workers:    {num_workers}")
    print(f"  Output:     {output_path}")
    print(f"  Work dir:   {work_dir}")
    print(f"  Python:     {args.python}")
    print(f"  Timeout:    {args.worker_timeout}s")
    print("=" * 60)

    # Validate source data exists
    task = TASK_CONFIGS[task_key]
    if "base_config" in task:
        if not task["base_config"].exists():
            print(f"ERROR: Base config not found: {task['base_config']}")
            sys.exit(1)
    else:
        if not task["source"].exists():
            print(f"ERROR: Source dataset not found: {task['source']}")
            print("Download it first:")
            print(f"  conda run -n mimicgen_env python {MIMICGEN_ROOT}/mimicgen/scripts/download_datasets.py "
                  f"--dataset_type source --tasks {task_key.rsplit('_', 1)[0]}")
            sys.exit(1)

    # Distribute demos across workers
    base_per_worker = num_demos // num_workers
    remainder = num_demos % num_workers
    worker_targets = []
    for i in range(num_workers):
        target = base_per_worker + (1 if i < remainder else 0)
        worker_targets.append(target)

    # Create configs and collect expected outputs
    os.makedirs(work_dir, exist_ok=True)
    configs_and_outputs = []
    for i in range(num_workers):
        config_path, expected_output = create_worker_config(
            task_key, i, worker_targets[i], work_dir,
            seed_offset=args.seed_offset,
        )
        configs_and_outputs.append((config_path, expected_output, i))

    # Launch workers
    print(f"\nLaunching {num_workers} workers...")
    t_start = time.time()

    worker_args = [
        (cfg_path, wid, args.python, args.worker_timeout)
        for cfg_path, _, wid in configs_and_outputs
    ]

    succeeded = []
    failed = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_run_single_worker, wa): wa[1]
            for wa in worker_args
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                _, ok = future.result()
                if ok:
                    succeeded.append(wid)
                else:
                    failed.append(wid)
            except Exception as e:
                print(f"[Worker {wid}] Unhandled error: {e}")
                failed.append(wid)

    elapsed = time.time() - t_start
    print(f"\nAll workers finished in {elapsed:.0f}s")
    print(f"  Succeeded: {len(succeeded)}/{num_workers}")
    if failed:
        print(f"  Failed: {sorted(failed)}")

    # Collect successful worker outputs
    demo_paths = []
    for cfg_path, expected_out, wid in configs_and_outputs:
        if wid in failed:
            continue
        if os.path.exists(expected_out):
            demo_paths.append(expected_out)
        else:
            print(f"  WARNING: Worker {wid} succeeded but output not found: {expected_out}")

    if not demo_paths:
        print("ERROR: No successful outputs to merge!")
        sys.exit(1)

    # Merge
    total_episodes = merge_worker_hdf5s(demo_paths, output_path)

    # Verify
    print(f"\nVerification:")
    print(f"  Target demos:  {num_demos}")
    print(f"  Actual demos:  {total_episodes}")
    print(f"  File size:     {os.path.getsize(output_path) / (1024**2):.1f} MB")

    if total_episodes < num_demos:
        print(f"  WARNING: Got {total_episodes}/{num_demos} demos. "
              f"Some workers may have failed.")

    # Cleanup
    if not args.keep_work_dir:
        print(f"\nCleaning up work directory: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\nWork directory kept at: {work_dir}")

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
