#!/usr/bin/env python3
"""
Build a 2000-demo mixed HDF5 per task for BC-RNN baseline training.

Sampling policy:
  - Tasks with D0 + D1 (coffee, stack, stack_three, three_piece_assembly):
    sample 1000 demos from D0 + 1000 from D1 -> 2000 total, renumbered demo_0000..demo_1999.
  - Tasks with only D0 (pick_place, threading): expects an *_extra.hdf5 with >= 1000
    extra demos; takes 1000 from the original D0 + 1000 from D0-extra -> 2000 total.
    (Extra datasets are produced by parallel_mimicgen_generate.py with --seed_offset.)

Output: /HOME/.../mimicgen_datasets/mixed_2000/{task}_mixed_2000.hdf5
Seed is fixed (42) for reproducible sampling.

Usage:
  conda activate mimicgen_env
  python build_mixed_2000_hdf5.py --task stack
  python build_mixed_2000_hdf5.py --task all      # run for all 6 tasks
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

CORE_DIR = Path(
    os.environ.get("MIMICGEN_DATASET_CORE_DIR")
    or "${BENCHMARK_DATA}/mimicgen_prepared"
)
OUT_DIR = Path(
    os.environ.get("MIMICGEN_MIXED_DIR")
    or "${BENCHMARK_DATA}/mimicgen_mixed_2000"
)

# (d0_file, d1_file_or_None, d0_extra_file_or_None)
# For dual-degree tasks, mix D0+D1. For single-degree tasks, mix D0+D0_extra.
TASK_SOURCES = {
    "coffee":               ("coffee_d0.hdf5",               "coffee_d1.hdf5",               None),
    "stack":                ("stack_d0.hdf5",                "stack_d1.hdf5",                None),
    "stack_three":          ("stack_three_d0.hdf5",          "stack_three_d1.hdf5",          None),
    "three_piece_assembly": ("three_piece_assembly_d0.hdf5", "three_piece_assembly_d1.hdf5", None),
    "pick_place":           ("pick_place_d0.hdf5",           None, "pick_place_d0_extra.hdf5"),
    "threading":            ("threading_d0.hdf5",            None, "threading_d0_extra.hdf5"),
}


def sample_demos(src_path: Path, n: int, rng: np.random.Generator) -> list:
    with h5py.File(src_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
    if len(demo_keys) < n:
        raise ValueError(f"{src_path.name} has only {len(demo_keys)} demos, need {n}")
    idx = rng.choice(len(demo_keys), size=n, replace=False)
    idx.sort()
    return [demo_keys[i] for i in idx]


def normalize_bc_training_keys(demo_grp) -> None:
    """Ensure robomimic dataset_keys have consistent 1D per-step shapes."""
    horizon = int(demo_grp["actions"].shape[0])

    if "rewards" in demo_grp:
        rewards = np.asarray(demo_grp["rewards"][()]).reshape(-1)
        del demo_grp["rewards"]
    else:
        rewards = np.zeros(horizon, dtype=np.float32)
    if rewards.shape[0] != horizon:
        raise ValueError(
            f"rewards length {rewards.shape[0]} does not match actions length {horizon}"
        )
    demo_grp.create_dataset("rewards", data=rewards)

    if "dones" in demo_grp:
        dones = np.asarray(demo_grp["dones"][()]).reshape(-1)
        del demo_grp["dones"]
    else:
        dones = np.zeros(horizon, dtype=np.int64)
        if horizon:
            dones[-1] = 1
    if dones.shape[0] != horizon:
        raise ValueError(
            f"dones length {dones.shape[0]} does not match actions length {horizon}"
        )
    demo_grp.create_dataset("dones", data=dones.astype(np.int64, copy=False))


def build_mixed(task: str, seed: int = 42) -> Path:
    d0_name, second_name, extra_name = TASK_SOURCES[task]
    d0_path = CORE_DIR / d0_name

    if second_name is not None:
        second_path = CORE_DIR / second_name
        label = f"D0(1000) + D1(1000)"
    elif extra_name is not None:
        second_path = CORE_DIR / extra_name
        label = f"D0(1000) + D0-extra(1000)"
    else:
        raise ValueError(f"Task {task} has neither D1 nor D0-extra configured")

    for p in (d0_path, second_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing source: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{task}_mixed_2000.hdf5"

    rng_first = np.random.default_rng(seed)
    rng_second = np.random.default_rng(seed + 1)
    first_keys = sample_demos(d0_path, 1000, rng_first)
    second_keys = sample_demos(second_path, 1000, rng_second)

    print(f"[{task}] {label}")
    print(f"  Source A: {d0_path.name} -> 1000 sampled")
    print(f"  Source B: {second_path.name} -> 1000 sampled")
    print(f"  Output:   {out_path}")

    total_samples = 0
    env_args = None

    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        idx = 0

        for src_path, demo_keys in [(d0_path, first_keys), (second_path, second_keys)]:
            with h5py.File(src_path, "r") as src_f:
                src_data = src_f["data"]
                if env_args is None and "env_args" in src_data.attrs:
                    env_args = src_data.attrs["env_args"]
                for ep_key in demo_keys:
                    new_key = f"demo_{idx}"
                    src_data.copy(ep_key, data_grp, name=new_key)
                    normalize_bc_training_keys(data_grp[new_key])
                    ns = data_grp[new_key].attrs.get("num_samples", 0)
                    total_samples += ns
                    idx += 1

        data_grp.attrs["total"] = total_samples
        if env_args is not None:
            data_grp.attrs["env_args"] = env_args

    print(f"  Wrote {idx} demos, {total_samples} total samples")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True,
                        choices=list(TASK_SOURCES.keys()) + ["all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = list(TASK_SOURCES.keys()) if args.task == "all" else [args.task]
    for t in tasks:
        try:
            build_mixed(t, seed=args.seed)
        except FileNotFoundError as e:
            print(f"[{t}] SKIP: {e}")
        except Exception as e:
            print(f"[{t}] ERROR: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
