#!/usr/bin/env python3
"""
Complete the single-degree BC-RNN baseline datasets for pick_place/threading.

This script is meant for the two tasks that do not have a D1 dataset:
  - pick_place
  - threading

Workflow per task:
  1. Reuse an existing core/{task}_d0_extra.hdf5 if it already has enough demos.
  2. Otherwise, salvage worker outputs from core/.parallel_gen_{task}_d0*.
  3. If the salvaged extra set is still short, generate only the missing demos.
  4. Build mixed_2000/{task}_mixed_2000.hdf5 via build_mixed_2000_hdf5.py.

Usage:
  conda activate mimicgen_env

  # Complete both tasks and build final mixed datasets
  python error_benchmark/scripts/training/complete_single_degree_baseline_data.py

  # Only complete pick_place with 16 workers
  python error_benchmark/scripts/training/complete_single_degree_baseline_data.py \
      --task pick_place --num_workers 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
CORE_DIR = Path(
    os.environ.get("MIMICGEN_DATASET_CORE_DIR")
    or "${BENCHMARK_DATA}/mimicgen_prepared"
)
PARALLEL_GEN_SCRIPT = (
    PROJECT_DIR
    / "error_benchmark"
    / "scripts"
    / "data_generation"
    / "parallel_mimicgen_generate.py"
)
BUILD_MIXED_SCRIPT = (
    PROJECT_DIR
    / "error_benchmark"
    / "scripts"
    / "training"
    / "build_mixed_2000_hdf5.py"
)

TASKS = ["pick_place", "threading"]
TASK_SEED_OFFSETS = {
    "pick_place": 500000,
    "threading": 600000,
}


def count_demos(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        return len([k for k in f["data"].keys() if k.startswith("demo_")])


def merge_hdf5s(input_paths: list[Path], output_path: Path) -> int:
    """Merge HDF5 files with a /data/demo_* layout into one file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    demo_idx = 0
    env_args = None

    with h5py.File(output_path, "w") as out_f:
        data_grp = out_f.create_group("data")

        for src_path in input_paths:
            with h5py.File(src_path, "r") as src_f:
                if "data" not in src_f:
                    continue

                src_data = src_f["data"]
                if env_args is None and "env_args" in src_data.attrs:
                    env_args = src_data.attrs["env_args"]

                ep_keys = sorted(
                    [k for k in src_data.keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                for ep_key in ep_keys:
                    new_key = f"demo_{demo_idx}"
                    src_data.copy(ep_key, data_grp, name=new_key)
                    total_samples += data_grp[new_key].attrs.get("num_samples", 0)
                    demo_idx += 1

        data_grp.attrs["total"] = total_samples
        if env_args is not None:
            data_grp.attrs["env_args"] = env_args

    return demo_idx


def find_worker_outputs(task: str) -> list[Path]:
    files = []
    for work_dir in sorted(CORE_DIR.glob(f".parallel_gen_{task}_d0*")):
        files.extend(sorted(work_dir.glob("w*/gen_w*/demo.hdf5")))
    return [p for p in files if p.is_file()]


def run_cmd(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["ERROR_RECOVERY_BENCHMARK_ROOT"] = str(PROJECT_DIR)
    env["MIMICGEN_DATASET_CORE_DIR"] = str(CORE_DIR)
    env.setdefault("MUJOCO_GL", "egl")
    if extra_env:
        env.update(extra_env)

    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def salvage_existing_extra(task: str, extra_path: Path) -> int:
    if extra_path.exists():
        demos = count_demos(extra_path)
        print(f"[{task}] reuse existing extra: {extra_path} ({demos} demos)")
        return demos

    worker_outputs = find_worker_outputs(task)
    if not worker_outputs:
        print(f"[{task}] no salvageable worker outputs found under {CORE_DIR}")
        return 0

    print(
        f"[{task}] salvage {len(worker_outputs)} worker outputs "
        f"-> {extra_path}"
    )
    demos = merge_hdf5s(worker_outputs, extra_path)
    print(f"[{task}] salvaged extra demos: {demos}")
    return demos


def top_up_extra(task: str, extra_path: Path, current_demos: int,
                 target_demos: int, num_workers: int, worker_timeout: int) -> int:
    remaining = target_demos - current_demos
    if remaining <= 0:
        return current_demos

    stamp = time.strftime("%Y%m%d_%H%M%S")
    topup_output = CORE_DIR / f"{task}_d0_extra_topup_{stamp}.hdf5"
    work_dir = CORE_DIR / f".parallel_gen_{task}_d0_topup_{stamp}"
    workers = min(num_workers, remaining)
    seed_offset = TASK_SEED_OFFSETS[task]

    run_cmd([
        sys.executable,
        str(PARALLEL_GEN_SCRIPT),
        "--task", f"{task}_d0",
        "--num_demos", str(remaining),
        "--num_workers", str(workers),
        "--output", str(topup_output),
        "--work_dir", str(work_dir),
        "--keep_work_dir",
        "--seed_offset", str(seed_offset),
        "--worker_timeout", str(worker_timeout),
    ])

    sources = [topup_output]
    if extra_path.exists():
        merged_tmp = extra_path.with_suffix(".tmp.hdf5")
        sources.insert(0, extra_path)
        demos = merge_hdf5s(sources, merged_tmp)
        shutil.move(str(merged_tmp), str(extra_path))
    else:
        shutil.move(str(topup_output), str(extra_path))
        demos = count_demos(extra_path)

    if demos < target_demos:
        raise RuntimeError(
            f"[{task}] top-up finished but extra is still short: "
            f"{demos}/{target_demos}"
        )

    print(f"[{task}] completed extra dataset: {extra_path} ({demos} demos)")
    return demos


def build_mixed(task: str) -> None:
    run_cmd([sys.executable, str(BUILD_MIXED_SCRIPT), "--task", task])


def complete_task(task: str, target_demos: int, num_workers: int,
                  worker_timeout: int, skip_generation: bool,
                  skip_build_mixed: bool) -> None:
    extra_path = CORE_DIR / f"{task}_d0_extra.hdf5"

    current_demos = salvage_existing_extra(task, extra_path)
    if current_demos < target_demos:
        if skip_generation:
            raise RuntimeError(
                f"[{task}] extra dataset still incomplete: "
                f"{current_demos}/{target_demos}"
            )
        current_demos = top_up_extra(
            task=task,
            extra_path=extra_path,
            current_demos=current_demos,
            target_demos=target_demos,
            num_workers=num_workers,
            worker_timeout=worker_timeout,
        )

    if not skip_build_mixed:
        build_mixed(task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=TASKS + ["all"],
        default="all",
        help="Task to complete (default: all)",
    )
    parser.add_argument(
        "--target_demos",
        type=int,
        default=1000,
        help="Target demo count for the *_d0_extra.hdf5 file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Worker count for any required top-up generation",
    )
    parser.add_argument(
        "--worker_timeout",
        type=int,
        default=int(os.environ.get("PARALLEL_MIMICGEN_WORKER_TIMEOUT", "7200")),
        help="Per-worker top-up timeout in seconds",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Do not launch new generation; only salvage what already exists",
    )
    parser.add_argument(
        "--skip_build_mixed",
        action="store_true",
        help="Do not build the final mixed_2000 file",
    )
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    for task in tasks:
        print("=" * 72)
        print(f"Complete baseline data: {task}")
        print("=" * 72)
        complete_task(
            task=task,
            target_demos=args.target_demos,
            num_workers=args.num_workers,
            worker_timeout=args.worker_timeout,
            skip_generation=args.skip_generation,
            skip_build_mixed=args.skip_build_mixed,
        )


if __name__ == "__main__":
    main()
