#!/usr/bin/env python
"""
Convert human-collected success demos (NPZ) to robomimic HDF5 format.

For most tasks this keeps the original direct merge behavior. For ``stack`` it
uses a stricter staged flow:
  1. write raw human demos to ``human_success_demos_raw.hdf5``
  2. run MimicGen ``prepare_src_dataset.py`` into ``human_success_demos_prepared.hdf5``
  3. validate prepared demos for source compatibility
  4. merge only accepted demos into the augmentable source dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import yaml

from error_benchmark.framework.mimicgen_source_validation import (
    STACK_ENV_INTERFACE_NAME,
    STACK_ENV_INTERFACE_TYPE,
    build_demo_replay_success_checker,
    validate_stack_prepared_demo,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIMICGEN_ROOT = PROJECT_ROOT / "shared" / "mimicgen_workspace"
PREPARE_SRC_SCRIPT = (
    MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "scripts" / "prepare_src_dataset.py"
)

ALL_TASKS = [
    "pick_place",
    "coffee",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
]

OBS_KEYS = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_vel_lin",
    "robot0_eef_vel_ang",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
]

STACK_RAW_HDF5 = "human_success_demos_raw.hdf5"
STACK_PREPARED_HDF5 = "human_success_demos_prepared.hdf5"
STACK_ACCEPTED_HDF5 = "human_success_demos.hdf5"
STACK_COMPATIBILITY_REPORT = "human_success_compatibility_report.json"


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(
        [k for k in data_group.keys() if k.startswith("demo_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_hdf5_attr(attrs, key: str, value: Any) -> None:
    if value is None:
        return
    value = _to_python_scalar(value)
    if isinstance(value, (dict, list, tuple)):
        attrs[key] = json.dumps(value)
    else:
        attrs[key] = value


def _load_registry_entry(task: str) -> dict:
    registry_path = PROJECT_ROOT / "error_benchmark" / "configs" / "task_registry.yaml"
    with open(registry_path) as handle:
        registry = yaml.safe_load(handle)
    tasks = registry.get("tasks", {})
    if task not in tasks:
        raise ValueError(f"Unknown task '{task}'")
    return tasks[task]


def load_task_config(task: str) -> dict:
    """Load task YAML config without importing robosuite-heavy helpers."""

    task_info = _load_registry_entry(task)
    config_path = PROJECT_ROOT / task_info["task_config"]
    with open(config_path) as handle:
        return yaml.safe_load(handle)


def get_fallback_dataset_path(task: str) -> str:
    task_info = _load_registry_entry(task)
    return str(task_info["dataset_path"])


def load_teleop_npz(npz_path: str) -> dict:
    """Load a human-collected NPZ and extract data for HDF5 conversion."""

    data = np.load(npz_path, allow_pickle=True)

    actions = data["actions"]
    num_steps = len(actions)

    states = data["states"]
    if len(states) == num_steps + 1:
        states = states[:-1]
    elif len(states) > num_steps:
        states = states[:num_steps]

    obs = {}
    for key in OBS_KEYS:
        npz_key = f"obs_{key}"
        if npz_key not in data:
            continue
        arr = data[npz_key]
        if len(arr) == num_steps + 1:
            arr = arr[1:]
        elif len(arr) > num_steps:
            arr = arr[:num_steps]
        obs[key] = arr

    metadata = {}
    for key in data.files:
        if key in {"actions", "states"} or key.startswith("obs_"):
            continue
        metadata[key] = _to_python_scalar(data[key])

    scene_id = str(metadata.get("scene_id", ""))
    metadata.setdefault("source_npz", Path(npz_path).name)

    return {
        "actions": actions,
        "states": states,
        "obs": obs,
        "scene_id": scene_id,
        "num_steps": num_steps,
        "metadata": metadata,
    }


def get_env_args_and_model(hdf5_path: str) -> tuple[str, str, int, int]:
    """Extract env metadata from an existing robomimic-style HDF5 file."""

    with h5py.File(hdf5_path, "r") as handle:
        env_args = handle["data"].attrs.get("env_args", "")
        total = int(handle["data"].attrs.get("total", 0))
        demo_keys = _sorted_demo_keys(handle["data"])
        model_file = ""
        if demo_keys:
            model_file = handle[f"data/{demo_keys[0]}"].attrs.get("model_file", "")
        return env_args, model_file, len(demo_keys), total


def write_demo_to_hdf5(
    handle: h5py.File,
    demo_key: str,
    demo_data: dict,
    model_file: str = "",
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a single demo to an HDF5 file."""

    group = handle.create_group(f"data/{demo_key}")
    group.attrs["num_samples"] = int(demo_data["num_steps"])
    if model_file:
        group.attrs["model_file"] = model_file

    for attr_name, attr_value in (extra_attrs or {}).items():
        _write_hdf5_attr(group.attrs, attr_name, attr_value)

    group.create_dataset("actions", data=demo_data["actions"].astype(np.float64))
    group.create_dataset("states", data=demo_data["states"].astype(np.float64))

    obs_group = group.create_group("obs")
    for key, array in demo_data["obs"].items():
        obs_group.create_dataset(key, data=array.astype(np.float64))


def merge_into_existing(
    existing_path: str,
    new_demos: list[dict],
    model_file: str,
    scene_id_map: dict,
) -> None:
    """Append generic demos to an existing success_demos.hdf5."""

    backup_path = existing_path + ".backup"
    shutil.copy2(existing_path, backup_path)

    try:
        with h5py.File(existing_path, "a") as handle:
            demo_keys = _sorted_demo_keys(handle["data"])
            max_idx = max((int(k.split("_")[1]) for k in demo_keys), default=-1)

            total_new_samples = 0
            for index, demo_data in enumerate(new_demos):
                demo_idx = max_idx + 1 + index
                demo_key = f"demo_{demo_idx}"
                write_demo_to_hdf5(handle, demo_key, demo_data, model_file)
                total_new_samples += int(demo_data["num_steps"])
                scene_id_map[demo_key] = demo_data.get("scene_id", "")

            old_total = int(handle["data"].attrs.get("total", 0))
            handle["data"].attrs["total"] = old_total + total_new_samples

            existing_map = {}
            if "scene_id_map" in handle["data"].attrs:
                existing_map = json.loads(handle["data"].attrs["scene_id_map"])
            existing_map.update(scene_id_map)
            handle["data"].attrs["scene_id_map"] = json.dumps(existing_map)

        os.remove(backup_path)
    except Exception:
        shutil.move(backup_path, existing_path)
        raise


def create_standalone(
    output_path: str,
    new_demos: list[dict],
    env_args: str,
    model_file: str,
) -> None:
    """Create a new HDF5 file with raw converted demos."""

    with h5py.File(output_path, "w") as handle:
        data_group = handle.create_group("data")
        if env_args:
            data_group.attrs["env_args"] = env_args

        total_samples = 0
        scene_id_map = {}
        for index, demo_data in enumerate(new_demos):
            demo_key = f"demo_{index}"
            write_demo_to_hdf5(handle, demo_key, demo_data, model_file)
            total_samples += int(demo_data["num_steps"])
            scene_id_map[demo_key] = demo_data.get("scene_id", "")

        data_group.attrs["total"] = total_samples
        data_group.attrs["scene_id_map"] = json.dumps(scene_id_map)


def write_stack_raw_hdf5(
    output_path: Path,
    demos: list[dict],
    env_args: str,
    model_file: str,
) -> None:
    """Write raw stack human demos with metadata attrs preserved."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        data_group = handle.create_group("data")
        if env_args:
            data_group.attrs["env_args"] = env_args

        total_samples = 0
        scene_id_map = {}
        for index, demo_data in enumerate(demos):
            demo_key = f"demo_{index}"
            extra_attrs = dict(demo_data.get("metadata", {}))
            extra_attrs["scene_id"] = demo_data.get("scene_id", "")
            extra_attrs.setdefault("demo_origin", "human_teleop")
            write_demo_to_hdf5(
                handle=handle,
                demo_key=demo_key,
                demo_data=demo_data,
                model_file=model_file,
                extra_attrs=extra_attrs,
            )
            total_samples += int(demo_data["num_steps"])
            scene_id_map[demo_key] = demo_data.get("scene_id", "")

        data_group.attrs["total"] = total_samples
        data_group.attrs["scene_id_map"] = json.dumps(scene_id_map)


def run_prepare_src_dataset(
    raw_path: Path,
    prepared_path: Path,
    env_interface_name: str,
) -> None:
    """Run MimicGen prepare_src_dataset.py into a staged prepared HDF5."""

    if prepared_path.exists():
        prepared_path.unlink()

    env = os.environ.copy()
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(MIMICGEN_ROOT / "robosuite"),
        str(MIMICGEN_ROOT / "mimicgen"),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        pythonpath_parts + ([existing_pythonpath] if existing_pythonpath else [])
    )

    cmd = [
        sys.executable,
        str(PREPARE_SRC_SCRIPT),
        "--dataset",
        str(raw_path),
        "--env_interface",
        env_interface_name,
        "--env_interface_type",
        STACK_ENV_INTERFACE_TYPE,
        "--output",
        str(prepared_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "prepare_src_dataset.py failed:\n"
            f"stdout:\n{result.stdout[-1000:]}\n"
            f"stderr:\n{result.stderr[-1000:]}"
        )


def validate_stack_prepared_hdf5(
    prepared_path: Path,
    task_config: dict,
    replay_success_checker=None,
) -> dict:
    """Validate prepared stack demos and annotate results in-place."""

    if replay_success_checker is None:
        replay_success_checker = build_demo_replay_success_checker(
            str(prepared_path),
            task_config,
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "prepared_dataset": str(prepared_path),
        "accepted": [],
        "rejected": [],
        "rejection_counts": {},
    }

    with h5py.File(prepared_path, "a") as handle:
        data_group = handle["data"]
        scene_id_map = {}
        if "scene_id_map" in data_group.attrs:
            scene_id_map = json.loads(data_group.attrs["scene_id_map"])

        for demo_key in _sorted_demo_keys(data_group):
            group = data_group[demo_key]
            if "scene_id" not in group.attrs and demo_key in scene_id_map:
                group.attrs["scene_id"] = scene_id_map[demo_key]

            result = validate_stack_prepared_demo(
                demo_key=demo_key,
                demo_group=group,
                task_config=task_config,
                replay_success_checker=replay_success_checker,
            )

            group.attrs["validation_status"] = "accepted" if result.accepted else "rejected"
            group.attrs["validation_reason"] = result.reason
            if "first_grasp_frame" in result.details and result.details["first_grasp_frame"] is not None:
                group.attrs["first_grasp_frame"] = int(result.details["first_grasp_frame"])

            serializable = result.to_dict()
            if result.accepted:
                report["accepted"].append(serializable)
            else:
                report["rejected"].append(serializable)
                report["rejection_counts"][result.reason] = (
                    report["rejection_counts"].get(result.reason, 0) + 1
                )

    report["accepted_count"] = len(report["accepted"])
    report["rejected_count"] = len(report["rejected"])
    return report


def write_stack_compatibility_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)


def _resolve_scene_id(group: h5py.Group, fallback_map: Dict[str, str], demo_key: str) -> str:
    if "scene_id" in group.attrs:
        return str(_to_python_scalar(group.attrs["scene_id"]))
    return str(fallback_map.get(demo_key, ""))


def merge_prepared_demos(
    existing_path: Path,
    prepared_path: Path,
    accepted_entries: list[dict],
) -> int:
    """Merge accepted prepared stack demos into an existing source dataset."""

    if not accepted_entries:
        return 0

    backup_path = existing_path.with_suffix(existing_path.suffix + ".backup")
    shutil.copy2(existing_path, backup_path)

    try:
        with h5py.File(prepared_path, "r") as src, h5py.File(existing_path, "a") as dst:
            src_data = src["data"]
            dst_data = dst["data"]
            scene_id_map = {}
            if "scene_id_map" in dst_data.attrs:
                scene_id_map = json.loads(dst_data.attrs["scene_id_map"])
            src_scene_id_map = {}
            if "scene_id_map" in src_data.attrs:
                src_scene_id_map = json.loads(src_data.attrs["scene_id_map"])

            demo_keys = _sorted_demo_keys(dst_data)
            max_idx = max((int(k.split("_")[1]) for k in demo_keys), default=-1)
            total_new_samples = 0

            for index, entry in enumerate(accepted_entries):
                src_key = entry["details"]["demo_key"]
                dst_key = f"demo_{max_idx + 1 + index}"
                src.copy(f"data/{src_key}", dst_data, name=dst_key)
                group = dst_data[dst_key]
                scene_id = _resolve_scene_id(group, src_scene_id_map, src_key)
                group.attrs["demo_origin"] = "human_teleop"
                group.attrs["scene_id"] = scene_id
                group.attrs["validation_status"] = "accepted"
                group.attrs["validation_reason"] = "accepted"
                scene_id_map[dst_key] = scene_id
                total_new_samples += int(
                    group.attrs.get("num_samples", group["actions"].shape[0])
                )

            dst_data.attrs["total"] = int(dst_data.attrs.get("total", 0)) + total_new_samples
            dst_data.attrs["scene_id_map"] = json.dumps(scene_id_map)

        backup_path.unlink()
    except Exception:
        shutil.move(str(backup_path), str(existing_path))
        raise

    return len(accepted_entries)


def create_prepared_standalone(
    output_path: Path,
    prepared_path: Path,
    accepted_entries: list[dict],
) -> int:
    """Write accepted prepared stack demos into a standalone augmentable HDF5."""

    with h5py.File(prepared_path, "r") as src, h5py.File(output_path, "w") as dst:
        src_data = src["data"]
        dst_data = dst.create_group("data")
        for attr_name in src_data.attrs:
            if attr_name == "scene_id_map":
                continue
            dst_data.attrs[attr_name] = src_data.attrs[attr_name]

        src_scene_id_map = {}
        if "scene_id_map" in src_data.attrs:
            src_scene_id_map = json.loads(src_data.attrs["scene_id_map"])

        total_samples = 0
        scene_id_map = {}
        for index, entry in enumerate(accepted_entries):
            src_key = entry["details"]["demo_key"]
            dst_key = f"demo_{index}"
            src.copy(f"data/{src_key}", dst_data, name=dst_key)
            group = dst_data[dst_key]
            scene_id = _resolve_scene_id(group, src_scene_id_map, src_key)
            group.attrs["demo_origin"] = "human_teleop"
            group.attrs["scene_id"] = scene_id
            group.attrs["validation_status"] = "accepted"
            group.attrs["validation_reason"] = "accepted"
            total_samples += int(group.attrs.get("num_samples", group["actions"].shape[0]))
            scene_id_map[dst_key] = scene_id

        dst_data.attrs["total"] = total_samples
        dst_data.attrs["scene_id_map"] = json.dumps(scene_id_map)

    return len(accepted_entries)


def build_conversion_report(
    task: str,
    demos: list[dict],
    skipped: int,
    mode: str,
    accepted_count: int,
    rejected_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    report = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "num_loaded": len(demos),
        "num_skipped": skipped,
        "mode": mode,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "demos": [
            {
                "scene_id": demo["scene_id"],
                "num_steps": demo["num_steps"],
                "obs_keys": list(demo["obs"].keys()),
                "metadata": {
                    key: _to_python_scalar(value)
                    for key, value in demo.get("metadata", {}).items()
                },
            }
            for demo in demos
        ],
    }
    if extra:
        report.update(extra)
    return report


def process_stack_task(task: str, demos: list[dict], args, existing_hdf5: Path) -> None:
    """Process stack demos with staged prepare + validation."""

    mimicgen_output = PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success"
    task_dir = mimicgen_output / task
    task_dir.mkdir(parents=True, exist_ok=True)

    if existing_hdf5.exists():
        env_args, model_file, existing_count, existing_total = get_env_args_and_model(
            str(existing_hdf5)
        )
    else:
        fallback_dataset = get_fallback_dataset_path(task)
        env_args, model_file, _, _ = get_env_args_and_model(fallback_dataset)
        existing_count = 0
        existing_total = 0

    print(f"  [{task}] Existing HDF5: {existing_count} demos, {existing_total} total samples")

    raw_path = task_dir / STACK_RAW_HDF5
    prepared_path = task_dir / STACK_PREPARED_HDF5
    compatibility_report_path = task_dir / STACK_COMPATIBILITY_REPORT

    write_stack_raw_hdf5(raw_path, demos, env_args, model_file)
    run_prepare_src_dataset(raw_path, prepared_path, STACK_ENV_INTERFACE_NAME)

    task_config = load_task_config(task)
    compatibility_report = validate_stack_prepared_hdf5(prepared_path, task_config)
    write_stack_compatibility_report(compatibility_report_path, compatibility_report)

    accepted_count = compatibility_report["accepted_count"]
    rejected_count = compatibility_report["rejected_count"]
    print(
        f"  [{task}] Compatibility: {accepted_count} accepted, {rejected_count} rejected"
    )

    if args.standalone or not existing_hdf5.exists():
        output_path = task_dir / STACK_ACCEPTED_HDF5
        written = create_prepared_standalone(
            output_path=output_path,
            prepared_path=prepared_path,
            accepted_entries=compatibility_report["accepted"],
        )
        print(f"  [{task}] Wrote standalone accepted dataset: {output_path} ({written} demos)")
        mode = "stack_standalone"
    else:
        merged = merge_prepared_demos(
            existing_path=existing_hdf5,
            prepared_path=prepared_path,
            accepted_entries=compatibility_report["accepted"],
        )
        print(f"  [{task}] Merged {merged} accepted demos into {existing_hdf5}")
        mode = "stack_merge"

    conversion_report = build_conversion_report(
        task=task,
        demos=demos,
        skipped=0,
        mode=mode,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        extra={
            "raw_dataset": str(raw_path),
            "prepared_dataset": str(prepared_path),
            "compatibility_report": str(compatibility_report_path),
            "rejection_counts": compatibility_report["rejection_counts"],
        },
    )
    report_path = mimicgen_output / "human_collected" / task / "conversion_report.json"
    with open(report_path, "w") as handle:
        json.dump(conversion_report, handle, indent=2)
    print(f"  [{task}] Report: {report_path}")


def process_task(task: str, args) -> None:
    """Process all human-collected NPZ files for one task."""

    mimicgen_output = PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success"
    collected_dir = mimicgen_output / "human_collected" / task
    existing_hdf5 = mimicgen_output / task / "success_demos.hdf5"

    if not collected_dir.exists():
        print(f"  [{task}] No collected demos directory: {collected_dir}")
        return

    npz_files = sorted(collected_dir.glob("*.npz"))
    if not npz_files:
        print(f"  [{task}] No NPZ files found in {collected_dir}")
        return

    print(f"\n  [{task}] Found {len(npz_files)} collected demos")

    demos = []
    skipped = 0
    for npz_path in npz_files:
        try:
            demo_data = load_teleop_npz(str(npz_path))
            if demo_data["num_steps"] == 0:
                print(f"    SKIP {npz_path.name}: empty demo")
                skipped += 1
                continue
            missing_obs = [key for key in OBS_KEYS if key not in demo_data["obs"]]
            if missing_obs:
                print(f"    WARN {npz_path.name}: missing obs keys: {missing_obs}")
            demos.append(demo_data)
        except Exception as exc:
            print(f"    ERROR loading {npz_path.name}: {exc}")
            skipped += 1

    if not demos:
        print(f"  [{task}] No valid demos to convert")
        return

    print(f"  [{task}] {len(demos)} valid demos, {skipped} skipped")

    if args.dry_run:
        for demo in demos[:5]:
            print(
                f"    {demo['scene_id']}: {demo['num_steps']} steps, "
                f"obs keys: {list(demo['obs'].keys())}, "
                f"validation={demo['metadata'].get('validation_reason', 'unknown')}"
            )
        if len(demos) > 5:
            print(f"    ... and {len(demos) - 5} more")
        return

    if task == "stack":
        process_stack_task(task, demos, args, existing_hdf5)
        return

    env_args = ""
    model_file = ""
    existing_count = 0

    if existing_hdf5.exists():
        env_args, model_file, existing_count, existing_total = get_env_args_and_model(
            str(existing_hdf5)
        )
        print(f"  [{task}] Existing HDF5: {existing_count} demos, {existing_total} total samples")
    else:
        fallback_dataset = get_fallback_dataset_path(task)
        env_args, model_file, _, _ = get_env_args_and_model(fallback_dataset)

    if args.standalone or not existing_hdf5.exists():
        output_path = mimicgen_output / task / "human_success_demos.hdf5"
        create_standalone(str(output_path), demos, env_args, model_file)
        mode = "standalone"
        print(f"  [{task}] Created standalone: {output_path} ({len(demos)} demos)")
    else:
        scene_id_map = {}
        merge_into_existing(str(existing_hdf5), demos, model_file, scene_id_map)
        mode = "merge"
        print(
            f"  [{task}] Merged: {existing_count} + {len(demos)} = "
            f"{existing_count + len(demos)} demos in {existing_hdf5}"
        )

    report = build_conversion_report(
        task=task,
        demos=demos,
        skipped=skipped,
        mode=mode,
        accepted_count=len(demos),
        rejected_count=0,
    )
    report_path = collected_dir / "conversion_report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"  [{task}] Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert human-collected NPZ demos to robomimic HDF5 format"
    )
    parser.add_argument("--task", type=str, default=None, help="Task to convert (default: all)")
    parser.add_argument("--all_tasks", action="store_true", help="Convert all tasks")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Create standalone HDF5 instead of merging",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be converted without writing",
    )
    args = parser.parse_args()

    if not args.task and not args.all_tasks:
        parser.error("Specify --task or --all_tasks")

    tasks = ALL_TASKS if args.all_tasks else [args.task]

    print(f"\n{'=' * 60}")
    print("  NPZ -> HDF5 Conversion")
    print(f"  Mode: {'standalone' if args.standalone else 'merge'}")
    print(f"  {'DRY RUN' if args.dry_run else ''}")
    print(f"{'=' * 60}")

    for task in tasks:
        process_task(task, args)

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
