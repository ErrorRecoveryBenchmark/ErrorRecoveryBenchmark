#!/usr/bin/env python
"""
Prepare MimicGen source datasets for all 6 tasks.

For each task, creates a small prepared HDF5 (~10 demos) with datagen_info
metadata that MimicGen's DataGenerator needs.

Requires GPU (MUJOCO_GL=egl) for robosuite simulation during preparation.

Usage:
    MUJOCO_GL=egl python error_benchmark/scripts/mimicgen/prepare_all_source_datasets.py
"""

import os
import sys
import shutil
import subprocess
import h5py
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIMICGEN_ROOT = PROJECT_ROOT / "shared" / "mimicgen_workspace"
MIMICGEN_SCRIPTS = MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "scripts"
SOURCE_DIR = MIMICGEN_ROOT / "mimicgen" / "datasets" / "source"
CORE_DIR = Path(
    os.environ.get("MIMICGEN_DATASET_CORE_DIR")
    or PROJECT_ROOT.parent / "mimicgen_datasets" / "core"
)
OUTPUT_DIR = PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success" / "prepared_source"

EXTRACT_SCRIPT = PROJECT_ROOT / "error_benchmark" / "scripts" / "mimicgen" / "extract_n_demos.py"
PREPARE_SCRIPT = MIMICGEN_SCRIPTS / "prepare_src_dataset.py"

PYTHON = sys.executable

# Number of source demos to extract (10 is plenty for MimicGen)
NUM_SOURCE_DEMOS = 10

# Task → preparation config
TASK_CONFIG = {
    "pick_place": {
        "source": str(SOURCE_DIR / "pick_place.hdf5"),  # Already prepared
        "env_interface": "MG_PickPlace",
        "needs_extract": False,
        "needs_prepare": False,  # Already has datagen_info
    },
    "coffee": {
        "source": str(SOURCE_DIR / "coffee.hdf5"),  # 10 demos, no datagen_info
        "env_interface": "MG_Coffee",
        "needs_extract": False,
        "needs_prepare": True,
    },
    "stack": {
        "source": str(CORE_DIR / "stack_d0.hdf5"),  # 1000 demos, no datagen_info
        "env_interface": "MG_Stack",
        "needs_extract": True,
        "needs_prepare": True,
    },
    "stack_three": {
        "source": str(CORE_DIR / "stack_three_d0.hdf5"),
        "env_interface": "MG_StackThree",
        "needs_extract": True,
        "needs_prepare": True,
    },
    "threading": {
        "source": str(CORE_DIR / "threading_d0.hdf5"),
        "env_interface": "MG_Threading",
        "needs_extract": True,
        "needs_prepare": True,
    },
    "three_piece_assembly": {
        "source": str(CORE_DIR / "three_piece_assembly_d0.hdf5"),
        "env_interface": "MG_ThreePieceAssembly",
        "needs_extract": True,
        "needs_prepare": True,
    },
}


def is_prepared(hdf5_path: str) -> bool:
    """Check if an HDF5 file has datagen_info with correct attrs."""
    if not os.path.exists(hdf5_path):
        return False
    try:
        with h5py.File(hdf5_path, "r") as f:
            demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            if not demos:
                return False
            dg_key = f"data/{demos[0]}/datagen_info"
            if dg_key not in f:
                return False
            dg = f[dg_key]
            if "eef_pose" not in dg:
                return False
            if "env_interface_name" not in dg.attrs:
                return False
            return True
    except Exception:
        return False


def _get_env():
    """Build environment dict with PYTHONPATH for MimicGen imports."""
    env = os.environ.copy()
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(MIMICGEN_ROOT / "robosuite"),
        str(MIMICGEN_ROOT / "mimicgen"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(pythonpath_parts + ([existing] if existing else []))
    return env


def run_cmd(cmd: list, desc: str):
    """Run a subprocess command with logging."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=_get_env())
    if result.returncode != 0:
        print(f"  ERROR in {desc}:")
        print(f"    stdout: {result.stdout[-500:]}")
        print(f"    stderr: {result.stderr[-500:]}")
        return False
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing MimicGen source datasets")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    results = {}
    for task_name, cfg in TASK_CONFIG.items():
        output_path = str(OUTPUT_DIR / f"{task_name}_src.hdf5")
        print(f"\n--- {task_name} ---")

        # Check if already prepared
        if is_prepared(output_path):
            print(f"  Already prepared: {output_path}")
            results[task_name] = "skipped (already prepared)"
            continue

        # Check source exists
        if not os.path.exists(cfg["source"]):
            print(f"  ERROR: Source not found: {cfg['source']}")
            results[task_name] = "ERROR: source not found"
            continue

        # Step A: Extract or copy
        if cfg["needs_extract"]:
            print(f"  Extracting {NUM_SOURCE_DEMOS} demos from {cfg['source']}...")
            ok = run_cmd(
                [PYTHON, str(EXTRACT_SCRIPT),
                 "--input", cfg["source"],
                 "--output", output_path,
                 "--n", str(NUM_SOURCE_DEMOS)],
                f"extract {task_name}",
            )
            if not ok:
                results[task_name] = "ERROR: extraction failed"
                continue
        else:
            print(f"  Copying from {cfg['source']}...")
            shutil.copy2(cfg["source"], output_path)

        # Step B: Prepare (add datagen_info) if needed
        if cfg["needs_prepare"]:
            print(f"  Preparing with env_interface={cfg['env_interface']}...")
            ok = run_cmd(
                [PYTHON, str(PREPARE_SCRIPT),
                 "--dataset", output_path,
                 "--env_interface", cfg["env_interface"],
                 "--env_interface_type", "robosuite"],
                f"prepare {task_name}",
            )
            if not ok:
                results[task_name] = "ERROR: preparation failed"
                continue

        # Verify
        if is_prepared(output_path):
            with h5py.File(output_path, "r") as f:
                n_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
                iface = f["data/demo_0/datagen_info"].attrs["env_interface_name"]
            print(f"  Verified: {n_demos} demos, interface={iface}")
            results[task_name] = f"OK ({n_demos} demos)"
        else:
            print(f"  WARNING: Verification failed for {output_path}")
            results[task_name] = "ERROR: verification failed"

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task_name, status in results.items():
        print(f"  {task_name:25s}: {status}")

    all_ok = all("OK" in v or "skipped" in v for v in results.values())
    if all_ok:
        print("\nAll datasets prepared successfully!")
    else:
        print("\nSome datasets failed. Check errors above.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
