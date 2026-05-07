#!/usr/bin/env python
"""
Create minimal HDF5 stub files containing only env_args metadata.

These stubs are sufficient for create_env() which only reads
f['data'].attrs['env_args'] from the HDF5 file.

Usage:
    python error_benchmark/scripts/utils/create_hdf5_stubs.py --output /path/to/output/
    python error_benchmark/scripts/utils/create_hdf5_stubs.py --output /path/to/output/ --tasks pick_place stack
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def create_stub(source_hdf5: str, output_path: str) -> int:
    """Extract env_args from source HDF5 and write a minimal stub.

    Returns the stub file size in bytes.
    """
    with h5py.File(source_hdf5, 'r') as src:
        env_args = src['data'].attrs['env_args']

    with h5py.File(output_path, 'w') as dst:
        grp = dst.create_group('data')
        grp.attrs['env_args'] = env_args

    return os.path.getsize(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create HDF5 stub files for portable collection")
    parser.add_argument("--output", required=True, help="Output directory for stub files")
    parser.add_argument("--tasks", nargs='*', default=None,
                        help="Specific tasks to create stubs for (default: all)")
    parser.add_argument("--registry", default=None,
                        help="Path to task_registry.yaml (default: auto-detect)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    registry_path = args.registry or str(PROJECT_ROOT / "error_benchmark" / "configs" / "task_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    tasks = args.tasks or list(registry['tasks'].keys())

    for task_name in tasks:
        task_info = registry['tasks'].get(task_name)
        if not task_info:
            print(f"  WARNING: Task '{task_name}' not in registry, skipping")
            continue

        dataset_path = task_info['dataset_path']
        if not os.path.isabs(dataset_path):
            dataset_path = str(PROJECT_ROOT / dataset_path)

        if not os.path.isfile(dataset_path):
            print(f"  WARNING: Dataset not found for {task_name}: {dataset_path}")
            continue

        output_path = os.path.join(args.output, f"{task_name}.hdf5")
        size = create_stub(dataset_path, output_path)
        print(f"  {task_name}: {output_path} ({size} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
