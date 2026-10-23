#!/usr/bin/env python
"""
Build MimicGen source datasets from automatic success demos plus human recovery demos.

The human recovery demos under ``outputs/recovery/demos`` are NPZ rollouts from
error states to task success. They are not directly usable as MimicGen source
data because they do not carry ``datagen_info``. This script stages them through
MimicGen's ``prepare_src_dataset.py`` and appends the accepted prepared demos to
a copied ``success_demos.hdf5``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np

from error_benchmark.framework.mimicgen_source_validation import (
    DEFAULT_ENV_INTERFACE_TYPE,
    get_env_interface_name_for_task,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIMICGEN_ROOT = PROJECT_ROOT / "shared" / "mimicgen_workspace"
PREPARE_SRC_SCRIPT = (
    MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "scripts" / "prepare_src_dataset.py"
)
SUCCESS_ROOT = PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success"
RECOVERY_ROOT = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "demos"

ALL_TASKS = [
    "pick_place",
    "coffee",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
]

RAW_HUMAN_NAME = "human_recovery_success_raw.hdf5"
PREPARED_HUMAN_NAME = "human_recovery_success_prepared.hdf5"
FILTERED_HUMAN_NAME = "human_recovery_success_filtered.hdf5"
COMBINED_NAME = "success_demos_combined.hdf5"
REPORT_NAME = "combined_success_source_report.json"

OBS_PREFIX = "obs_"
CORE_ARRAY_KEYS = {"actions", "states"}


@dataclass
class RecoveryEntry:
    """One candidate human recovery demo discovered on disk."""

    npz_path: Path
    task_name: str
    demo_id: str
    subtype_id: str = ""
    scene_id: str = ""
    error_name: str = ""
    degree: str = ""
    rbg: str = ""
    manifest_entry: Optional[dict] = None


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(
        [key for key in data_group.keys() if key.startswith("demo_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_python_scalar(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_hdf5_attr(attrs: h5py.AttributeManager, key: str, value: Any) -> None:
    if value is None:
        return
    value = _to_python_scalar(value)
    if isinstance(value, (dict, list, tuple)):
        attrs[key] = json.dumps(value)
    else:
        attrs[key] = value


def _safe_bool(data: np.lib.npyio.NpzFile, key: str, default: Optional[bool] = None) -> Optional[bool]:
    if key not in data.files:
        return default
    return bool(_to_python_scalar(data[key]))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(_to_python_scalar(value))


def _load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _build_python_env() -> dict:
    env = os.environ.copy()
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(MIMICGEN_ROOT / "robosuite"),
        str(MIMICGEN_ROOT / "mimicgen"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        pythonpath_parts + ([existing] if existing else [])
    )
    return env


def get_env_args_and_model(source_hdf5: Path) -> tuple[str, str]:
    """Return environment metadata and a representative robosuite model file."""

    with h5py.File(source_hdf5, "r") as handle:
        data_group = handle["data"]
        env_args = data_group.attrs.get("env_args", "")
        model_file = ""
        demo_keys = _sorted_demo_keys(data_group)
        if demo_keys:
            model_file = data_group[demo_keys[0]].attrs.get("model_file", "")
        return _safe_str(env_args), _safe_str(model_file)


def discover_recovery_entries(
    task: str,
    recovery_root: Path,
) -> tuple[list[RecoveryEntry], dict]:
    """Find candidate human recovery NPZ files for one task."""

    task_dir = recovery_root / task
    report = {
        "task": task,
        "task_dir": str(task_dir),
        "manifest_used": False,
        "discovered": 0,
        "missing_npz": [],
        "duplicate_npz": [],
    }
    if not task_dir.exists():
        return [], report

    entries: list[RecoveryEntry] = []
    seen: set[Path] = set()
    manifest_path = task_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        report["manifest_used"] = True
        for item in manifest.get("demos", []):
            demo_id = str(item.get("demo_id", ""))
            subtype_id = str(item.get("subtype_id", ""))
            npz_path = task_dir / subtype_id / f"{demo_id}.npz"
            source_path = item.get("metadata", {}).get("source_path")
            if not npz_path.exists() and source_path:
                npz_path = Path(source_path)
            if not npz_path.exists():
                report["missing_npz"].append(
                    {
                        "demo_id": demo_id,
                        "expected_path": str(task_dir / subtype_id / f"{demo_id}.npz"),
                    }
                )
                continue
            resolved = npz_path.resolve()
            if resolved in seen:
                report["duplicate_npz"].append(
                    {"demo_id": demo_id, "path": str(npz_path)}
                )
                continue
            seen.add(resolved)
            entries.append(
                RecoveryEntry(
                    npz_path=npz_path,
                    task_name=str(item.get("task_name", task)),
                    demo_id=demo_id or npz_path.stem,
                    subtype_id=subtype_id,
                    scene_id=str(item.get("scene_id", "")),
                    error_name=str(item.get("error_name", "")),
                    degree=str(item.get("degree", "")),
                    rbg=str(item.get("rbg", "")),
                    manifest_entry=item,
                )
            )

    for npz_path in sorted(task_dir.rglob("*.npz")):
        resolved = npz_path.resolve()
        if resolved in seen:
            continue
        subtype_id = npz_path.parent.name if npz_path.parent != task_dir else ""
        entries.append(
            RecoveryEntry(
                npz_path=npz_path,
                task_name=task,
                demo_id=npz_path.stem,
                subtype_id=subtype_id,
            )
        )

    report["discovered"] = len(entries)
    return entries, report


def load_success_recovery_demo(entry: RecoveryEntry) -> tuple[Optional[dict], Optional[dict]]:
    """Load one recovery NPZ and return either demo data or a rejection record."""

    try:
        data = np.load(entry.npz_path, allow_pickle=True)
    except Exception as exc:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "bad_npz",
            "error": str(exc),
        }

    if "actions" not in data.files:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "missing_actions",
        }
    if "states" not in data.files:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "missing_states",
        }

    actions = np.asarray(data["actions"])
    states = np.asarray(data["states"])
    num_steps = int(actions.shape[0])
    if num_steps == 0:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "empty_actions",
        }
    if states.shape[0] < num_steps:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "states_shorter_than_actions",
            "states": int(states.shape[0]),
            "actions": num_steps,
        }

    success = _safe_bool(data, "success", None)
    manifest_success = (
        bool(entry.manifest_entry.get("success"))
        if entry.manifest_entry is not None and "success" in entry.manifest_entry
        else None
    )
    if success is not True and manifest_success is not True:
        return None, {
            "path": str(entry.npz_path),
            "demo_id": entry.demo_id,
            "reason": "not_success",
            "success": success,
            "manifest_success": manifest_success,
        }

    for key in ("validation_passed", "replay_passed"):
        flag = _safe_bool(data, key, None)
        if flag is False:
            return None, {
                "path": str(entry.npz_path),
                "demo_id": entry.demo_id,
                "reason": f"{key}_false",
            }

    obs = {}
    for key in data.files:
        if not key.startswith(OBS_PREFIX):
            continue
        obs_key = key[len(OBS_PREFIX):]
        arr = np.asarray(data[key])
        if arr.shape[0] == num_steps + 1:
            arr = arr[1:]
        elif arr.shape[0] > num_steps:
            arr = arr[:num_steps]
        if arr.shape[0] == num_steps:
            obs[obs_key] = arr

    metadata: Dict[str, Any] = {
        "task_name": entry.task_name,
        "demo_id": entry.demo_id,
        "subtype_id": entry.subtype_id,
        "scene_id": entry.scene_id,
        "error_name": entry.error_name,
        "degree": entry.degree,
        "rbg": entry.rbg,
        "source_npz": str(entry.npz_path),
        "demo_origin": "human_recovery",
        "success": True,
    }
    if entry.manifest_entry:
        metadata["manifest_entry"] = entry.manifest_entry

    for key in data.files:
        if key in CORE_ARRAY_KEYS or key.startswith(OBS_PREFIX):
            continue
        if key in {"scene_id", "task_name", "demo_id", "subtype_id", "error_name", "degree", "rbg"} and metadata.get(key):
            continue
        if key == "camera_images":
            metadata["has_camera_images"] = True
            continue
        if key.startswith("obj_"):
            continue
        value = _to_python_scalar(data[key])
        if isinstance(value, (dict, list)):
            metadata[key] = value
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    scene_id = str(metadata.get("scene_id") or "")
    if not scene_id and "scene_id" in data.files:
        scene_id = _safe_str(data["scene_id"])
    metadata["scene_id"] = scene_id

    return {
        "actions": actions[:num_steps],
        "states": states[:num_steps],
        "obs": obs,
        "num_steps": num_steps,
        "scene_id": metadata["scene_id"],
        "metadata": metadata,
    }, None


def collect_success_recovery_demos(
    entries: Iterable[RecoveryEntry],
    *,
    limit: Optional[int] = None,
) -> tuple[list[dict], list[dict], dict]:
    """Load and filter successful recovery demos."""

    demos: list[dict] = []
    rejected: list[dict] = []
    for entry in entries:
        demo, rejection = load_success_recovery_demo(entry)
        if rejection is not None:
            rejected.append(rejection)
            continue
        demos.append(demo)
        if limit is not None and len(demos) >= limit:
            break

    reason_counts = Counter(item["reason"] for item in rejected)
    return demos, rejected, {
        "accepted": len(demos),
        "rejected": len(rejected),
        "rejection_counts": dict(sorted(reason_counts.items())),
    }


def write_raw_human_hdf5(
    output_path: Path,
    demos: list[dict],
    env_args: str,
    model_file: str,
) -> None:
    """Write successful human recovery demos as a raw robomimic-style HDF5."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        data_group = handle.create_group("data")
        if env_args:
            data_group.attrs["env_args"] = env_args
        data_group.attrs["created_by"] = Path(__file__).name
        data_group.attrs["created_at"] = datetime.now().isoformat()

        total_samples = 0
        scene_id_map: dict[str, str] = {}
        for index, demo in enumerate(demos):
            demo_key = f"demo_{index}"
            group = data_group.create_group(demo_key)
            group.attrs["num_samples"] = int(demo["num_steps"])
            if model_file:
                group.attrs["model_file"] = model_file
            for attr_key, attr_value in demo.get("metadata", {}).items():
                _write_hdf5_attr(group.attrs, attr_key, attr_value)

            group.create_dataset("actions", data=demo["actions"].astype(np.float64))
            group.create_dataset("states", data=demo["states"].astype(np.float64))
            obs_group = group.create_group("obs")
            for obs_key, obs_value in demo.get("obs", {}).items():
                obs_group.create_dataset(obs_key, data=np.asarray(obs_value))

            total_samples += int(demo["num_steps"])
            scene_id_map[demo_key] = str(demo.get("scene_id", ""))

        data_group.attrs["total"] = total_samples
        data_group.attrs["scene_id_map"] = json.dumps(scene_id_map)


def run_prepare_src_dataset(
    raw_path: Path,
    prepared_path: Path,
    task: str,
) -> None:
    """Run MimicGen prepare_src_dataset.py for a human recovery raw HDF5."""

    if prepared_path.exists():
        prepared_path.unlink()

    cmd = [
        sys.executable,
        str(PREPARE_SRC_SCRIPT),
        "--dataset",
        str(raw_path),
        "--env_interface",
        get_env_interface_name_for_task(task),
        "--env_interface_type",
        DEFAULT_ENV_INTERFACE_TYPE,
        "--output",
        str(prepared_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_build_python_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "prepare_src_dataset.py failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )


def filter_prepared_source(prepared_path: Path, filtered_path: Path, task: str) -> dict:
    """Filter prepared human demos with the same source checks used by augmentation."""

    from error_benchmark.scripts.mimicgen.augment_success_demos import filter_valid_source

    if filtered_path.exists():
        filtered_path.unlink()
    return filter_valid_source(str(prepared_path), task, str(filtered_path))


def _read_scene_id_map(data_group: h5py.Group) -> dict:
    raw = data_group.attrs.get("scene_id_map", "{}")
    try:
        return json.loads(_safe_str(raw))
    except json.JSONDecodeError:
        return {}


def _copy_attrs(src_attrs: h5py.AttributeManager, dst_attrs: h5py.AttributeManager) -> None:
    for key in src_attrs:
        dst_attrs[key] = src_attrs[key]


def merge_sources(
    existing_path: Path,
    human_path: Optional[Path],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Create a combined source HDF5 without modifying the original source."""

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing_path, output_path)

    report = {
        "existing_source": str(existing_path),
        "human_source": str(human_path) if human_path else None,
        "combined_source": str(output_path),
        "existing_count": 0,
        "human_merged": 0,
        "total_count": 0,
        "total_samples": 0,
    }

    with h5py.File(output_path, "a") as dst:
        dst_data = dst["data"]
        dst_keys = _sorted_demo_keys(dst_data)
        report["existing_count"] = len(dst_keys)
        next_index = max((int(key.split("_")[1]) for key in dst_keys), default=-1) + 1
        total_samples = 0
        scene_id_map = _read_scene_id_map(dst_data)

        for demo_key in dst_keys:
            group = dst_data[demo_key]
            if "demo_origin" not in group.attrs:
                group.attrs["demo_origin"] = "mimicgen_success"
            total_samples += int(group.attrs.get("num_samples", group["actions"].shape[0]))

        if human_path is not None and human_path.exists():
            with h5py.File(human_path, "r") as src:
                src_data = src["data"]
                src_scene_id_map = _read_scene_id_map(src_data)
                for offset, src_key in enumerate(_sorted_demo_keys(src_data)):
                    dst_key = f"demo_{next_index + offset}"
                    src.copy(f"data/{src_key}", dst_data, name=dst_key)
                    dst_group = dst_data[dst_key]
                    dst_group.attrs["demo_origin"] = "human_recovery"
                    dst_group.attrs["source_demo_key"] = src_key
                    scene_id = (
                        _safe_str(dst_group.attrs.get("scene_id", ""))
                        or str(src_scene_id_map.get(src_key, ""))
                    )
                    if scene_id:
                        dst_group.attrs["scene_id"] = scene_id
                    scene_id_map[dst_key] = scene_id
                    total_samples += int(
                        dst_group.attrs.get("num_samples", dst_group["actions"].shape[0])
                    )
                    report["human_merged"] += 1

        dst_data.attrs["total"] = total_samples
        dst_data.attrs["scene_id_map"] = json.dumps(scene_id_map)
        dst_data.attrs["combined_source_created_at"] = datetime.now().isoformat()
        dst_data.attrs["combined_source_existing"] = str(existing_path)
        if human_path:
            dst_data.attrs["combined_source_human"] = str(human_path)
        report["total_count"] = len(_sorted_demo_keys(dst_data))
        report["total_samples"] = total_samples

    return report


def validate_combined_source(path: Path) -> dict:
    """Return lightweight schema counts for a combined HDF5."""

    counts = {
        "demo_count": 0,
        "missing_actions": 0,
        "missing_states": 0,
        "missing_datagen_info": 0,
        "missing_obs_group": 0,
        "origin_counts": {},
    }
    origin_counts: Counter[str] = Counter()
    with h5py.File(path, "r") as handle:
        data_group = handle["data"]
        for demo_key in _sorted_demo_keys(data_group):
            counts["demo_count"] += 1
            group = data_group[demo_key]
            if "actions" not in group:
                counts["missing_actions"] += 1
            if "states" not in group:
                counts["missing_states"] += 1
            if "datagen_info" not in group:
                counts["missing_datagen_info"] += 1
            if "obs" not in group:
                counts["missing_obs_group"] += 1
            origin_counts[_safe_str(group.attrs.get("demo_origin", "unknown"))] += 1
    counts["origin_counts"] = dict(sorted(origin_counts.items()))
    return counts


def build_for_task(task: str, args) -> dict:
    """Build combined source data for one task."""

    task_dir = Path(args.success_root) / task
    existing_path = task_dir / args.existing_name
    if not existing_path.exists():
        raise FileNotFoundError(f"Missing existing source dataset: {existing_path}")

    entries, discovery_report = discover_recovery_entries(task, Path(args.recovery_root))
    demos, rejected, filter_report = collect_success_recovery_demos(
        entries,
        limit=args.limit,
    )
    report = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "dry_run": bool(args.dry_run),
        "discovery": discovery_report,
        "human_npz_filter": filter_report,
        "human_npz_rejected": rejected,
        "paths": {
            "existing_source": str(existing_path),
            "raw_human": str(task_dir / args.raw_human_name),
            "prepared_human": str(task_dir / args.prepared_human_name),
            "filtered_human": str(task_dir / args.filtered_human_name),
            "combined_source": str(task_dir / args.output_name),
        },
    }

    if args.dry_run:
        return report

    env_args, model_file = get_env_args_and_model(existing_path)
    raw_path = task_dir / args.raw_human_name
    prepared_path = task_dir / args.prepared_human_name
    filtered_path = task_dir / args.filtered_human_name

    human_source_for_merge: Optional[Path] = None
    if demos:
        write_raw_human_hdf5(raw_path, demos, env_args, model_file)
        if args.reuse_prepared and prepared_path.exists():
            report["prepare"] = {"mode": "reused", "path": str(prepared_path)}
        else:
            run_prepare_src_dataset(raw_path, prepared_path, task)
            report["prepare"] = {"mode": "created", "path": str(prepared_path)}
        source_filter_report = filter_prepared_source(prepared_path, filtered_path, task)
        report["source_filter"] = source_filter_report
        if source_filter_report.get("valid_count", 0) > 0:
            human_source_for_merge = filtered_path
    else:
        report["prepare"] = {"mode": "skipped_no_human_success_demos"}
        report["source_filter"] = {
            "valid_count": 0,
            "rejected_count": 0,
            "rejection_counts": {},
        }

    output_path = task_dir / args.output_name
    merge_report = merge_sources(
        existing_path=existing_path,
        human_path=human_source_for_merge,
        output_path=output_path,
        overwrite=args.overwrite,
    )
    report["merge"] = merge_report
    report["combined_validation"] = validate_combined_source(output_path)

    report_path = task_dir / REPORT_NAME
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare successful human recovery demos as MimicGen source data and "
            "merge them with existing success_demos.hdf5."
        )
    )
    parser.add_argument("--task", choices=ALL_TASKS, help="Single task to process")
    parser.add_argument("--all-tasks", action="store_true", help="Process all tasks")
    parser.add_argument("--success-root", default=str(SUCCESS_ROOT))
    parser.add_argument("--recovery-root", default=str(RECOVERY_ROOT))
    parser.add_argument("--existing-name", default="success_demos.hdf5")
    parser.add_argument("--output-name", default=COMBINED_NAME)
    parser.add_argument("--raw-human-name", default=RAW_HUMAN_NAME)
    parser.add_argument("--prepared-human-name", default=PREPARED_HUMAN_NAME)
    parser.add_argument("--filtered-human-name", default=FILTERED_HUMAN_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Limit accepted human demos per task")
    parser.add_argument("--reuse-prepared", action="store_true", help="Reuse existing prepared human HDF5")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing combined HDF5")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect inputs; do not write HDF5 files")
    args = parser.parse_args()

    if not args.task and not args.all_tasks:
        parser.error("Specify --task or --all-tasks")
    return args


def main() -> int:
    args = parse_args()
    tasks = ALL_TASKS if args.all_tasks else [args.task]

    print("=" * 72)
    print("Build combined MimicGen success source")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Mode: {'dry-run' if args.dry_run else 'write'}")
    print("=" * 72)

    results = []
    failures = []
    for task in tasks:
        print(f"\n[{task}]")
        try:
            report = build_for_task(task, args)
            results.append(report)
            human = report["human_npz_filter"]["accepted"]
            rejected = report["human_npz_filter"]["rejected"]
            print(f"  human NPZ accepted={human}, rejected={rejected}")
            if not args.dry_run:
                merge = report["merge"]
                print(
                    "  combined: "
                    f"{merge['existing_count']} existing + {merge['human_merged']} human "
                    f"= {merge['total_count']} demos"
                )
                print(f"  output: {report['paths']['combined_source']}")
                print(f"  report: {report['report_path']}")
        except Exception as exc:
            failures.append({"task": task, "error": str(exc)})
            print(f"  ERROR: {exc}")

    summary_path = Path(args.success_root) / "combined_success_source_summary.json"
    if not args.dry_run:
        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                    "failures": failures,
                },
                handle,
                indent=2,
            )
        print(f"\nSummary: {summary_path}")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  {item['task']}: {item['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
