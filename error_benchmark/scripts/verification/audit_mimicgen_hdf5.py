#!/usr/bin/env python3
"""Audit and safely filter the active MimicGen policy-training HDF5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))
FAMILIES = ("core", "mixed_2000", "bc_rnn_pi05", "bc_rnn_humandemo")
TASKS = (
    "three_piece_assembly",
    "stack_three",
    "pick_place",
    "threading",
    "coffee",
    "stack",
)
EXPECTED_ROBOT = {task: ("Sawyer" if task == "pick_place" else "Panda") for task in TASKS}
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
SOURCE_ATTRS = (
    "bc_rnn_pi05_source",
    "humandemo_source",
    "source_dataset",
    "source_path",
    "lerobot_source",
    "source",
)
SOURCE_DEMO_ATTRS = ("bc_rnn_pi05_source_demo", "source_demo", "source_demo_key")


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _demo_keys(data: h5py.Group) -> list[str]:
    return sorted(
        (key for key in data if key.startswith("demo_") and isinstance(data[key], h5py.Group)),
        key=lambda key: (int(key[5:]) if key[5:].isdigit() else sys.maxsize, key),
    )


def task_from_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    for task in TASKS:
        if normalized.startswith(task):
            return task
    raise ValueError(f"Cannot infer task from {name!r}")


def discover_active_files(root: Path) -> list[tuple[str, Path]]:
    files = []
    for family in FAMILIES:
        directory = root / family
        files.extend((family, path) for path in sorted(directory.glob("*.hdf5")))
    return files


def _parse_env_args(data: h5py.Group) -> tuple[dict, list[str]]:
    reasons = []
    raw = data.attrs.get("env_args")
    if raw is None:
        return {}, ["missing_env_args"]
    try:
        value = _plain(raw)
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
        if not isinstance(parsed, dict):
            raise TypeError("env_args is not an object")
        return parsed, reasons
    except Exception:
        return {}, ["invalid_env_args"]


def _env_robot_reasons(env_args: dict, task: str) -> list[str]:
    reasons = []
    env_name = str(env_args.get("env_name", ""))
    expected_env = task.replace("_", "")
    if expected_env not in env_name.lower().replace("_", ""):
        reasons.append("task_env_mismatch")
    robots = (env_args.get("env_kwargs") or {}).get("robots")
    if isinstance(robots, str):
        robots = [robots]
    if not robots or EXPECTED_ROBOT[task].lower() not in {str(r).lower() for r in robots}:
        reasons.append("robot_mismatch")
    return reasons


def _dataset(group: h5py.Group, path: str) -> h5py.Dataset | None:
    try:
        value = group[path]
    except KeyError:
        return None
    return value if isinstance(value, h5py.Dataset) else None


def _finite(dataset: h5py.Dataset) -> bool:
    if dataset.dtype.kind not in "fc":
        return True
    return bool(np.isfinite(dataset[()]).all())


def demo_fingerprint(group: h5py.Group, family: str) -> str | None:
    names = ["actions"]
    if "states" in group:
        names.append("states")
    elif _dataset(group, "obs/state") is not None:
        names.append("obs/state")
    else:
        return None
    digest = hashlib.sha256()
    for name in names:
        dataset = _dataset(group, name)
        if dataset is None:
            return None
        values = np.ascontiguousarray(dataset[()])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _source_metadata(file_data: h5py.Group, group: h5py.Group) -> tuple[str | None, str | None]:
    source = next(
        (str(_plain(owner.attrs[key])) for owner in (group, file_data) for key in SOURCE_ATTRS if key in owner.attrs),
        None,
    )
    source_demo = next(
        (
            str(_plain(owner.attrs[key]))
            for owner in (group, file_data)
            for key in SOURCE_DEMO_ATTRS
            if key in owner.attrs
        ),
        None,
    )
    return source, source_demo


def _structural_reasons(
    group: h5py.Group,
    family: str,
    file_data: h5py.Group,
) -> tuple[list[str], int, str | None, dict]:
    reasons = []
    actions = _dataset(group, "actions")
    if actions is None:
        return ["missing_actions"], 0, None, {}
    if actions.ndim != 2 or actions.shape[0] == 0:
        reasons.append("invalid_action_shape")
    horizon = int(actions.shape[0]) if actions.ndim else 0
    if actions.ndim != 2 or actions.shape[-1] != 7:
        reasons.append("action_dimension_mismatch")
    if not _finite(actions):
        reasons.append("non_finite_actions")
    try:
        num_samples = int(group.attrs["num_samples"])
    except Exception:
        reasons.append("missing_or_invalid_num_samples")
    else:
        if num_samples != horizon:
            reasons.append("num_samples_mismatch")

    required = ["rewards", "dones"] if family == "bc_rnn_humandemo" else ["states"]
    for name in required:
        dataset = _dataset(group, name)
        if dataset is None:
            reasons.append(f"missing_{name}")
            continue
        if dataset.ndim == 0 or dataset.shape[0] != horizon:
            reasons.append(f"{name}_length_mismatch")
        if name == "states" and not _finite(dataset):
            reasons.append("non_finite_states")

    for name in ("rewards", "dones"):
        dataset = _dataset(group, name)
        if dataset is not None and (dataset.ndim == 0 or dataset.shape[0] != horizon):
            reasons.append(f"{name}_length_mismatch")

    obs = group.get("obs")
    if not isinstance(obs, h5py.Group):
        reasons.append("missing_obs")
    else:
        for image_key in IMAGE_KEYS:
            image = _dataset(group, f"obs/{image_key}")
            if image is None:
                reasons.append(f"missing_{image_key}")
            elif image.dtype != np.dtype("uint8") or image.shape != (horizon, 84, 84, 3):
                reasons.append(f"invalid_{image_key}_schema")
        for name, item in obs.items():
            if not isinstance(item, h5py.Dataset):
                continue
            if item.ndim == 0 or item.shape[0] != horizon:
                reasons.append(f"obs_length_mismatch:{name}")
            if not name.endswith("_image") and not _finite(item):
                reasons.append(f"non_finite_obs:{name}")

    if family == "bc_rnn_humandemo":
        state = _dataset(group, "obs/state")
        if state is None:
            reasons.append("missing_state_observation")
        elif state.ndim != 2 or state.shape != (horizon, 8):
            reasons.append("state_observation_shape_mismatch")
        rewards = _dataset(group, "rewards")
        dones = _dataset(group, "dones")
        # Terminal-success evidence follows the converter contract
        # (build_bc_rnn_humandemo_hdf5.py): rewards are all-zero placeholders
        # (BC-RNN consumes no reward), evidence is dones[-1] == 1. Requiring
        # rewards > 0 mislabelled 100% of legitimate converter output and
        # triggered the 2026-07-17 false-positive prune.
        if horizon and (
            rewards is None
            or dones is None
            or not bool(np.asarray(dones[-1]).item())
        ):
            reasons.append("missing_terminal_success_evidence")
        source, source_demo = _source_metadata(file_data, group)
        if not source:
            reasons.append("missing_source_metadata")
    else:
        source, source_demo = _source_metadata(file_data, group)
        if "model_file" not in group.attrs:
            reasons.append("missing_model_file")

    fingerprint = None
    try:
        fingerprint = demo_fingerprint(group, family)
    except Exception:
        reasons.append("fingerprint_error")
    details = {"source": source, "source_demo": source_demo}
    return sorted(set(reasons)), horizon, fingerprint, details


class ReplaySession:
    def __init__(self, dataset_path: Path, task: str):
        import yaml

        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.scripts.utils.script_utils import create_env

        config_path = PROJECT_ROOT / "error_benchmark" / "configs" / "tasks" / f"{task}.yaml"
        with config_path.open(encoding="utf-8") as handle:
            task_config = yaml.safe_load(handle)
        self.wrapper = EnvWrapper(
            create_env(task_config, str(dataset_path), enable_camera=False, has_renderer=False),
            task_config,
        )

    def check(self, group: h5py.Group) -> bool:
        actions = group["actions"][()]
        states = group["states"][()]
        self.wrapper.set_sim_state_flat(states[0])
        self.wrapper.forward()
        for action in actions:
            self.wrapper.step(action)
        return bool(self.wrapper.check_success())

    def close(self) -> None:
        try:
            self.wrapper._env.close()
        except Exception:
            pass


def audit_file(
    path: Path,
    root: Path,
    family: str,
    source_fingerprints: dict[str, list[dict]] | None = None,
    replay_factory: Callable[[Path, str], object] | None = ReplaySession,
    progress_every: int = 50,
) -> tuple[list[dict], dict]:
    relative = path.relative_to(root).as_posix()
    task = task_from_name(path.name)
    source_fingerprints = source_fingerprints or {}
    records = []
    file_errors = []
    replay = None
    total_samples = 0
    try:
        handle = h5py.File(path, "r")
    except Exception as exc:
        return [], {
            "path": relative,
            "family": family,
            "task": task,
            "demo_count": 0,
            "file_errors": ["hdf5_unreadable"],
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }

    with handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            return [], {
                "path": relative,
                "family": family,
                "task": task,
                "demo_count": 0,
                "file_errors": ["missing_data_group"],
            }
        data = handle["data"]
        env_args, env_reasons = _parse_env_args(data)
        env_reasons.extend(_env_robot_reasons(env_args, task) if env_args else [])
        family_task_attr = {
            "bc_rnn_pi05": "bc_rnn_pi05_task",
            "bc_rnn_humandemo": "humandemo_task",
        }.get(family)
        if family_task_attr:
            recorded_task = data.attrs.get(family_task_attr)
            if recorded_task is None:
                env_reasons.append("missing_family_task_metadata")
            elif str(_plain(recorded_task)) != task:
                env_reasons.append("family_task_mismatch")
        keys = _demo_keys(data)
        for index, key in enumerate(keys, 1):
            group = data[key]
            reasons, horizon, fingerprint, details = _structural_reasons(group, family, data)
            reasons.extend(env_reasons)
            total_samples += horizon
            records.append(
                {
                    "item_id": f"{relative}::{key}",
                    "path": relative,
                    "family": family,
                    "task": task,
                    "demo_key": key,
                    "status": "fail" if reasons else "pending",
                    "reasons": sorted(set(reasons)),
                    "num_samples": horizon,
                    "fingerprint": fingerprint,
                    **details,
                }
            )
            if progress_every and index % progress_every == 0:
                print(f"AUDIT PROGRESS {relative} demos={index}/{len(keys)}", flush=True)

        duplicates = defaultdict(list)
        for record in records:
            if record["fingerprint"]:
                duplicates[record["fingerprint"]].append(record)
        for duplicate_records in duplicates.values():
            if len(duplicate_records) > 1:
                for record in duplicate_records:
                    record["reasons"] = sorted(set(record["reasons"] + ["duplicate_within_file"]))
                    record["status"] = "fail"

        needs_replay = family in {"core", "mixed_2000", "bc_rnn_pi05"}
        for index, record in enumerate(records, 1):
            if record["status"] == "fail" or not needs_replay:
                if record["status"] == "pending":
                    record["status"] = "pass"
                    record["qualification_mode"] = "bc_training_schema_and_terminal_evidence"
                continue

            matches = source_fingerprints.get(record["fingerprint"], [])
            if family != "core" and matches:
                source = record.get("source")
                source_demo = record.get("source_demo")
                if source and source_demo:
                    named = [
                        match
                        for match in matches
                        if Path(source).name == Path(match["path"]).name
                        and source_demo == match["demo_key"]
                    ]
                    if not named:
                        record["status"] = "fail"
                        record["reasons"] = ["lineage_content_conflict"]
                        continue
                record["status"] = "pass"
                record["qualification_mode"] = "qualified_source_fingerprint"
                record["resolved_sources"] = [match["item_id"] for match in matches]
                continue

            if replay_factory is None:
                record["status"] = "unverifiable"
                record["reasons"] = ["semantic_replay_not_run"]
                continue
            try:
                if replay is None:
                    replay = replay_factory(path, task)
                replay_ok = bool(replay.check(data[record["demo_key"]]))
            except Exception as exc:
                record["status"] = "unverifiable"
                record["reasons"] = ["semantic_replay_error"]
                record["diagnostic"] = f"{type(exc).__name__}: {exc}"
            else:
                record["status"] = "pass" if replay_ok else "fail"
                record["reasons"] = [] if replay_ok else ["replay_not_successful"]
                record["qualification_mode"] = "simulator_action_replay"
            if progress_every and index % progress_every == 0:
                print(f"REPLAY PROGRESS {relative} demos={index}/{len(records)}", flush=True)

        if replay is not None:
            replay.close()
        stored_total = data.attrs.get("total")
        if stored_total is None:
            file_errors.append("missing_file_total")
        else:
            try:
                if int(stored_total) != total_samples:
                    file_errors.append("file_total_mismatch")
            except Exception:
                file_errors.append("invalid_file_total")

    counts = Counter(record["status"] for record in records)
    return records, {
        "path": relative,
        "family": family,
        "task": task,
        "demo_count": len(records),
        "total_samples": total_samples,
        "status_counts": dict(sorted(counts.items())),
        "file_errors": sorted(set(file_errors)),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _staging_inventory(root: Path, core_fingerprints: set[str]) -> list[dict]:
    active = {path.resolve() for _, path in discover_active_files(root)}
    rows = []
    for path in sorted((root / "core").rglob("*.hdf5")):
        if path.resolve() in active:
            continue
        row = {"path": path.relative_to(root).as_posix(), "readable": False, "demo_count": 0}
        try:
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                if not isinstance(data, h5py.Group):
                    row["reason"] = "missing_data_group"
                else:
                    fingerprints = []
                    errors = 0
                    for key in _demo_keys(data):
                        try:
                            fingerprints.append(demo_fingerprint(data[key], "core"))
                        except Exception:
                            errors += 1
                    row.update(
                        readable=True,
                        demo_count=len(fingerprints),
                        fingerprint_errors=errors,
                        merged_matches=sum(fp in core_fingerprints for fp in fingerprints if fp),
                        unique_fingerprints=len({fp for fp in fingerprints if fp}),
                    )
        except Exception as exc:
            row["reason"] = "hdf5_unreadable"
            row["diagnostic"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _audit_file_worker(payload: tuple[str, str, str, dict]) -> tuple[list[dict], dict]:
    path, root, family, source_fingerprints = payload
    return audit_file(Path(path), Path(root), family, source_fingerprints=source_fingerprints)


def run_audit(root: Path, report_dir: Path, workers: int = 1) -> dict:
    files = discover_active_files(root)
    report_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    file_records = []
    qualified_sources = defaultdict(list)
    for family in FAMILIES:
        family_paths = [path for discovered_family, path in files if discovered_family == family]
        source_index = dict(qualified_sources) if family in {"mixed_2000", "bc_rnn_pi05"} else {}
        payloads = [(str(path), str(root), family, source_index) for path in family_paths]
        if workers == 1:
            results = list(map(_audit_file_worker, payloads))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
                results = list(executor.map(_audit_file_worker, payloads))
        for records, file_record in results:
            all_records.extend(records)
            file_records.append(file_record)
            if family == "core":
                for record in records:
                    if record["status"] == "pass" and record["fingerprint"]:
                        qualified_sources[record["fingerprint"]].append(
                            {key: record[key] for key in ("item_id", "path", "demo_key")}
                        )
            print(
                f"AUDITED {file_record['path']} demos={file_record['demo_count']} "
                f"counts={file_record.get('status_counts', {})} errors={file_record['file_errors']}",
                flush=True,
            )

    all_records.sort(key=lambda row: row["item_id"])
    file_records.sort(key=lambda row: row["path"])
    core_fingerprints = {
        row["fingerprint"]
        for row in all_records
        if row["family"] == "core" and row["status"] == "pass" and row["fingerprint"]
    }
    staging = _staging_inventory(root, core_fingerprints)
    _write_jsonl(report_dir / "demos.jsonl", all_records)
    _write_jsonl(report_dir / "files.jsonl", file_records)
    _write_jsonl(report_dir / "staging.jsonl", staging)
    counts = Counter(row["status"] for row in all_records)
    by_family = {}
    for family in FAMILIES:
        subset = [row for row in all_records if row["family"] == family]
        by_family[family] = {
            "source_demos": len(subset),
            "status_counts": dict(sorted(Counter(row["status"] for row in subset).items())),
            "reason_counts": dict(
                sorted(Counter(reason for row in subset for reason in row["reasons"]).items())
            ),
        }
    summary = {
        "root": str(root),
        "files": len(file_records),
        "source_demos": len(all_records),
        "status_counts": dict(sorted(counts.items())),
        "by_family": by_family,
        "files_with_errors": sum(bool(row["file_errors"]) for row in file_records),
        "staging_files": len(staging),
        "staging_readable": sum(bool(row["readable"]) for row in staging),
        "staging_demos": sum(int(row["demo_count"]) for row in staging),
        "staging_merged_matches": sum(int(row.get("merged_matches", 0)) for row in staging),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MimicGen HDF5 qualification",
        "",
        f"- Active files: {summary['files']}",
        f"- Source demos: {summary['source_demos']}",
        f"- Status: {summary['status_counts']}",
        f"- Staging files: {summary['staging_files']} ({summary['staging_readable']} readable)",
        "",
        "| Family | Source demos | Status |",
        "|---|---:|---|",
    ]
    for family, values in by_family.items():
        lines.append(f"| {family} | {values['source_demos']} | `{values['status_counts']}` |")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MIMICGEN HDF5 AUDIT COMPLETE files={len(file_records)} source_demos={len(all_records)}")
    return summary


def check_report(report_dir: Path, expected_files: int, expected_demos: int) -> None:
    demos = _read_jsonl(report_dir / "demos.jsonl")
    files = _read_jsonl(report_dir / "files.jsonl")
    ids = [row.get("item_id") for row in demos]
    duplicate_records = len(ids) - len(set(ids))
    missing = sum(not item_id for item_id in ids)
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    missing += abs(len(files) - expected_files)
    missing += abs(len(demos) - expected_demos)
    missing += int(summary.get("files") != len(files))
    missing += int(summary.get("source_demos") != len(demos))
    if missing or duplicate_records:
        raise SystemExit(
            f"MIMICGEN HDF5 REPORT INVALID files={len(files)} source_demos={len(demos)} "
            f"missing={missing} duplicate_records={duplicate_records}"
        )
    print(
        f"MIMICGEN HDF5 REPORT VERIFIED files={len(files)} source_demos={len(demos)} "
        f"missing=0 duplicate_records=0"
    )


def _copy_attrs(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _build_filtered_file(source_path: Path, temp_path: Path, pass_records: list[dict]) -> None:
    expected = {row["demo_key"]: row for row in pass_records}
    with h5py.File(source_path, "r") as source, h5py.File(temp_path, "w") as target:
        _copy_attrs(source, target)
        for key in source:
            if key != "data":
                source.copy(key, target, name=key)
        source_data = source["data"]
        target_data = target.create_group("data")
        _copy_attrs(source_data, target_data)
        total = 0
        for key in _demo_keys(source_data):
            if key not in expected:
                continue
            source_data.copy(key, target_data, name=key)
            copied = target_data[key]
            fingerprint = demo_fingerprint(copied, expected[key]["family"])
            if fingerprint != expected[key]["fingerprint"]:
                raise RuntimeError(f"fingerprint changed while copying {key}")
            total += int(copied["actions"].shape[0])
        target_data.attrs["total"] = total
        for attr in ("num_demos", "total_demos"):
            if attr in target_data.attrs:
                target_data.attrs[attr] = len(expected)


def _verify_filtered_file(path: Path, pass_records: list[dict]) -> None:
    expected = {row["demo_key"]: row for row in pass_records}
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        actual = set(_demo_keys(data))
        if actual != set(expected):
            raise RuntimeError(f"filtered keys differ: expected={len(expected)} actual={len(actual)}")
        total = 0
        for key, record in expected.items():
            reasons, horizon, fingerprint, _ = _structural_reasons(data[key], record["family"], data)
            if reasons:
                raise RuntimeError(f"filtered demo {key} failed structural re-audit: {reasons}")
            if fingerprint != record["fingerprint"]:
                raise RuntimeError(f"filtered demo {key} fingerprint mismatch")
            total += horizon
        if int(data.attrs.get("total", -1)) != total:
            raise RuntimeError("filtered file total mismatch")


def clean_active(root: Path, report_dir: Path) -> dict:
    demos = _read_jsonl(report_dir / "demos.jsonl")
    files = _read_jsonl(report_dir / "files.jsonl")
    records_by_path = defaultdict(list)
    for row in demos:
        records_by_path[row["path"]].append(row)
    file_info = {row["path"]: row for row in files}
    cleanup = []
    quarantine_root = root / ".quarantine_20260714"
    for relative in sorted(records_by_path):
        records = records_by_path[relative]
        passing = [row for row in records if row["status"] == "pass"]
        needs_rebuild = len(passing) != len(records) or bool(file_info[relative]["file_errors"])
        if not needs_rebuild:
            cleanup.append({"path": relative, "action": "unchanged", "kept": len(passing), "removed": 0})
            continue
        active_path = root / relative
        temp_path = active_path.with_name(f".{active_path.name}.qualification_tmp")
        quarantine_path = quarantine_root / relative
        if temp_path.exists() or quarantine_path.exists():
            raise RuntimeError(f"refusing to overwrite temp or quarantine for {relative}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        _build_filtered_file(active_path, temp_path, passing)
        _verify_filtered_file(temp_path, passing)
        os.replace(active_path, quarantine_path)
        try:
            os.replace(temp_path, active_path)
        except Exception:
            os.replace(quarantine_path, active_path)
            raise
        cleanup.append(
            {
                "path": relative,
                "action": "filtered",
                "kept": len(passing),
                "removed": len(records) - len(passing),
                "quarantine_path": quarantine_path.relative_to(root).as_posix(),
            }
        )
        print(
            f"CLEANED {relative} kept={len(passing)} removed={len(records) - len(passing)}",
            flush=True,
        )
    _write_jsonl(report_dir / "cleanup.jsonl", cleanup)
    result = {
        "files_filtered": sum(row["action"] == "filtered" for row in cleanup),
        "files_unchanged": sum(row["action"] == "unchanged" for row in cleanup),
        "demos_removed": sum(row["removed"] for row in cleanup),
        "demos_kept": sum(row["kept"] for row in cleanup),
    }
    (report_dir / "cleanup_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"MIMICGEN CLEAN COMPLETE {result}")
    return result


def check_active(root: Path, report_dir: Path) -> None:
    records = _read_jsonl(report_dir / "demos.jsonl")
    status = {row["item_id"]: row["status"] for row in records}
    expected_pass = {item_id for item_id, value in status.items() if value == "pass"}
    actual = set()
    files = discover_active_files(root)
    for _, path in files:
        relative = path.relative_to(root).as_posix()
        with h5py.File(path, "r") as handle:
            actual.update(f"{relative}::{key}" for key in _demo_keys(handle["data"]))
    failed_active = sum(status.get(item_id) in {"fail", "unverifiable"} for item_id in actual)
    missing_pass = len(expected_pass - actual)
    unexpected_active = len(actual - set(status))
    if len(files) != 36 or failed_active or missing_pass or unexpected_active:
        raise SystemExit(
            f"MIMICGEN ACTIVE DATA INVALID files={len(files)} failed_active={failed_active} "
            f"missing_pass={missing_pass} unexpected_active={unexpected_active}"
        )
    print(
        f"MIMICGEN ACTIVE DATA VERIFIED files={len(files)} failed_active=0 "
        f"missing_pass=0 unexpected_active=0"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--audit", type=Path, metavar="ROOT")
    modes.add_argument("--clean", type=Path, metavar="ROOT")
    modes.add_argument("--check-report", type=Path, metavar="REPORT_DIR")
    modes.add_argument("--check-active", type=Path, metavar="ROOT")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--expected-files", type=int, default=36)
    parser.add_argument("--expected-demos", type=int, default=52846)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit:
        if not args.report_dir:
            raise SystemExit("--report-dir is required with --audit")
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        run_audit(args.audit.resolve(), args.report_dir.resolve(), workers=args.workers)
    elif args.clean:
        if not args.report_dir:
            raise SystemExit("--report-dir is required with --clean")
        clean_active(args.clean.resolve(), args.report_dir.resolve())
    elif args.check_report:
        check_report(args.check_report.resolve(), args.expected_files, args.expected_demos)
    elif args.check_active:
        if not args.report_dir:
            raise SystemExit("--report-dir is required with --check-active")
        check_active(args.check_active.resolve(), args.report_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
