#!/usr/bin/env python3
"""Audit human-labelled datasets and trace their derived artifacts.

This audit is deliberately read-only.  Dataset names are treated as claims;
episode fingerprints provide the evidence for normal-MimicGen versus human-
recovery origin.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


TASKS = (
    "coffee",
    "pick_place",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
)
CLAIMED_HUMANDEMO_COUNTS = {
    "coffee": 2190,
    "pick_place": 0,
    "stack": 2175,
    "stack_three": 2214,
    "threading": 1210,
    "three_piece_assembly": 2227,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _task_from_name(name: str) -> str | None:
    normalized = name.lower().replace("-", "_")
    return next((task for task in sorted(TASKS, key=len, reverse=True) if task in normalized), None)


def _action_hash(actions: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _trajectory_hash(actions: np.ndarray, state: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (actions, state):
        normalized = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        digest.update(str(normalized.shape).encode("ascii"))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def classify_origin(
    action_hash: str | None,
    core_refs: dict[str, list[str]],
    recovery_refs: dict[str, list[dict[str, Any]]],
) -> tuple[str, list[str]]:
    if not action_hash:
        return "unverifiable", []
    core = core_refs.get(action_hash, [])
    recovery = recovery_refs.get(action_hash, [])
    if core and recovery:
        return "ambiguous", [*core, *(row["item_id"] for row in recovery)]
    if recovery:
        return "human_recovery", [row["item_id"] for row in recovery]
    if core:
        return "normal_mimicgen", core
    return "unmatched", []


def _dataset_family(root: Path, task: str) -> str | None:
    name = root.name
    if name == f"{task}_humandemo":
        return "claimed_humandemo"
    if "merged_human_recovery" in name:
        return "merged_human_recovery"
    if "human_recovery" in name:
        return "human_recovery_only"
    return None


def discover_lerobot_datasets(root: Path) -> list[dict[str, Any]]:
    datasets = []
    seen = set()
    for info_path in sorted(root.rglob("meta/info.json")):
        dataset_root = info_path.parent.parent
        task = _task_from_name(dataset_root.name)
        if task is None:
            task = _task_from_name(str(dataset_root.relative_to(root)))
        if task is None:
            continue
        family = _dataset_family(dataset_root, task)
        if family is None:
            continue
        resolved = str(dataset_root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        datasets.append(
            {
                "root": dataset_root,
                "dataset_id": str(dataset_root.relative_to(root)),
                "task": task,
                "family": family,
            }
        )
    return datasets


def _image_present(value: Any, dataset_root: Path) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload:
            return True
        image_path = value.get("path")
        return bool(image_path and (dataset_root / image_path).exists())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) > 0
    array = np.asarray(value)
    return array.size > 0


def _decode_image(value: Any, dataset_root: Path) -> bool:
    from PIL import Image

    try:
        if isinstance(value, dict):
            if value.get("bytes"):
                source = io.BytesIO(value["bytes"])
            elif value.get("path"):
                source = dataset_root / value["path"]
            else:
                return False
            with Image.open(source) as image:
                image.load()
                return image.size == (84, 84)
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(value))) as image:
                image.load()
                return image.size == (84, 84)
        return np.asarray(value).shape[:2] == (84, 84)
    except Exception:
        return False


def _column_array(table, name: str, dtype=None) -> np.ndarray:
    values = table.column(name).combine_chunks().to_pylist()
    return np.asarray(values, dtype=dtype)


def audit_lerobot_episode(item: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    dataset_root = Path(item["dataset_root"])
    episode_index = int(item["episode_index"])
    info = item["info"]
    expected_length = int(item["expected_length"])
    chunk = episode_index // int(info.get("chunks_size", 1000))
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    parquet_path = dataset_root / pattern.format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    reasons = []
    warnings = []
    action_hash = None
    trajectory_hash = None
    length = 0
    decoded_samples = 0
    required = (
        "image",
        "wrist_image",
        "state",
        "actions",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    )
    try:
        table = pq.read_table(parquet_path, columns=list(required))
        length = table.num_rows
        missing = [name for name in required if name not in table.column_names]
        reasons.extend(f"missing_column:{name}" for name in missing)
        if missing:
            raise ValueError("required columns missing")
        actions = _column_array(table, "actions", np.float32)
        state = _column_array(table, "state", np.float32)
        timestamps = _column_array(table, "timestamp", np.float64).reshape(-1)
        frame_indices = _column_array(table, "frame_index", np.int64).reshape(-1)
        episode_indices = _column_array(table, "episode_index", np.int64).reshape(-1)
        if length == 0:
            reasons.append("empty_episode")
        if length != expected_length:
            reasons.append("episode_length_mismatch")
        if actions.shape != (length, 7):
            reasons.append("action_shape_mismatch")
        if state.shape != (length, 8):
            reasons.append("state_shape_mismatch")
        if not np.isfinite(actions).all():
            reasons.append("non_finite_actions")
        if not np.isfinite(state).all():
            reasons.append("non_finite_state")
        if not np.isfinite(timestamps).all():
            reasons.append("non_finite_timestamps")
        fps = int(info.get("fps", 0))
        if fps != 20:
            reasons.append("fps_mismatch")
        if length and not np.allclose(timestamps, np.arange(length) / max(fps, 1), atol=1e-4):
            reasons.append("timestamp_sequence_mismatch")
        if not np.array_equal(frame_indices, np.arange(length)):
            reasons.append("frame_index_sequence_mismatch")
        if length and not np.all(episode_indices == episode_index):
            reasons.append("episode_index_mismatch")
        action_hash = _action_hash(actions)
        trajectory_hash = _trajectory_hash(actions, state)

        for image_key in ("image", "wrist_image"):
            images = table.column(image_key).combine_chunks().to_pylist()
            if len(images) != length:
                reasons.append(f"{image_key}_length_mismatch")
                continue
            if any(not _image_present(value, dataset_root) for value in images):
                reasons.append(f"{image_key}_missing_payload")
            indices = sorted({0, length // 2, length - 1}) if length else []
            for index in indices:
                if _decode_image(images[index], dataset_root):
                    decoded_samples += 1
                else:
                    reasons.append(f"{image_key}_decode_failed")
                    break
    except Exception as exc:
        if not reasons:
            reasons.append("parquet_unreadable")
        warnings.append(f"{type(exc).__name__}: {exc}")

    return {
        "item_id": f"lerobot:{item['dataset_id']}:{episode_index}",
        "stage": "lerobot",
        "dataset_id": item["dataset_id"],
        "dataset_family": item["dataset_family"],
        "task": item["task"],
        "episode_index": episode_index,
        "source_path": str(parquet_path),
        "length": length,
        "action_hash": action_hash,
        "trajectory_hash": trajectory_hash,
        "structural_qualified": not reasons,
        "failure_reasons": sorted(set(reasons)),
        "warnings": warnings,
        "image_samples_decoded": decoded_samples,
        "terminal_success_evidence": "unavailable",
        "simulator_replay_available": False,
    }


def _episode_items(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    root = dataset["root"]
    with (root / "meta/info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = _load_jsonl(root / "meta/episodes.jsonl")
    return [
        {
            "dataset_root": str(root),
            "dataset_id": dataset["dataset_id"],
            "dataset_family": dataset["family"],
            "task": dataset["task"],
            "episode_index": row["episode_index"],
            "expected_length": row["length"],
            "info": info,
        }
        for row in episodes
    ]


def load_core_refs(root: Path) -> dict[str, dict[str, list[str]]]:
    refs = {task: defaultdict(list) for task in TASKS}
    for task in TASKS:
        for path in (root / f"{task}_d0.hdf5", root / f"{task}_d1.hdf5"):
            if not path.exists():
                continue
            with h5py.File(path, "r") as handle:
                for demo_key, group in handle.get("data", {}).items():
                    if "actions" not in group:
                        continue
                    refs[task][_action_hash(group["actions"][()])].append(f"{path.name}:{demo_key}")
    return {task: dict(values) for task, values in refs.items()}


def load_recovery_refs(report_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    refs = {task: defaultdict(list) for task in TASKS}
    for row in _load_jsonl(report_path):
        path = Path(row.get("source_path", ""))
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                action_hash = _action_hash(data["actions"])
        except Exception:
            continue
        refs[row["task"]][action_hash].append(row)
    return {task: dict(values) for task, values in refs.items()}


def apply_origin_and_usability(
    rows: list[dict[str, Any]],
    core_refs: dict[str, dict[str, list[str]]],
    recovery_refs: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    for row in rows:
        origin, source_ids = classify_origin(
            row.get("action_hash"),
            core_refs.get(row["task"], {}),
            recovery_refs.get(row["task"], {}),
        )
        matches = recovery_refs.get(row["task"], {}).get(row.get("action_hash"), [])
        recovery_qualified = bool(matches) and any(match.get("qualified") for match in matches)
        recovery_credit = bool(matches) and any(match.get("coverage_credit") for match in matches)
        structural = bool(row.get("structural_qualified"))
        row.update(
            {
                "origin_class": origin,
                "origin_source_ids": source_ids,
                "human_origin_verified": origin == "human_recovery",
                "recovery_source_qualified": recovery_qualified,
                "recovery_source_coverage_credit": recovery_credit,
                "usability": {
                    "nominal_policy_training": structural and origin == "normal_mimicgen",
                    "recovery_policy_training": structural and origin == "human_recovery" and recovery_qualified,
                    "error_scene_generation_source": False,
                    "canonical_training_scene": False,
                },
            }
        )


def _hdf5_task(path: Path) -> str | None:
    return _task_from_name(path.stem.removesuffix("_humandemo"))


def audit_hdf5_files(
    roots: list[Path],
    claimed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from error_benchmark.scripts.verification import audit_mimicgen_hdf5 as base

    claimed_by_hash = defaultdict(list)
    for row in claimed_rows:
        if row.get("trajectory_hash"):
            claimed_by_hash[(row["task"], row["trajectory_hash"])].append(row)
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*_humandemo.hdf5")):
            task = _hdf5_task(path)
            if task is None:
                continue
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                if not isinstance(data, h5py.Group):
                    rows.append(
                        {
                            "item_id": f"hdf5:{path}:missing_data",
                            "stage": "hdf5",
                            "task": task,
                            "source_path": str(path),
                            "structural_qualified": False,
                            "failure_reasons": ["missing_data_group"],
                        }
                    )
                    continue
                env_args, env_reasons = base._parse_env_args(data)
                env_reasons += base._env_robot_reasons(env_args, task)
                for demo_key in base._demo_keys(data):
                    group = data[demo_key]
                    reasons, horizon, _, details = base._structural_reasons(
                        group, "bc_rnn_humandemo", data
                    )
                    reasons.extend(env_reasons)
                    action_hash = None
                    trajectory_hash = None
                    image_samples_ok = True
                    try:
                        actions = group["actions"][()]
                        state = group["obs/state"][()]
                        action_hash = _action_hash(actions)
                        trajectory_hash = _trajectory_hash(actions, state)
                        for key in base.IMAGE_KEYS:
                            images = group[f"obs/{key}"]
                            for index in sorted({0, horizon // 2, horizon - 1}) if horizon else []:
                                if np.asarray(images[index]).shape != (84, 84, 3):
                                    image_samples_ok = False
                    except Exception:
                        reasons.append("fingerprint_or_image_sample_error")
                    if not image_samples_ok:
                        reasons.append("image_sample_shape_mismatch")
                    links = claimed_by_hash.get((task, trajectory_hash), [])
                    if not links:
                        reasons.append("orphan_hdf5_demo")
                    elif len(links) > 1:
                        reasons.append("ambiguous_lerobot_source")
                    source = links[0] if len(links) == 1 else None
                    source_usable = source.get("usability", {}) if source else {}
                    rows.append(
                        {
                            "item_id": f"hdf5:{path}:{demo_key}",
                            "stage": "hdf5",
                            "dataset_id": str(path),
                            "task": task,
                            "demo_key": demo_key,
                            "source_path": str(path),
                            "length": horizon,
                            "action_hash": action_hash,
                            "trajectory_hash": trajectory_hash,
                            "source_metadata": details,
                            "linked_lerobot_ids": [link["item_id"] for link in links],
                            "structural_qualified": not reasons,
                            "failure_reasons": sorted(set(reasons)),
                            "origin_class": source.get("origin_class", "unverifiable") if source else "unverifiable",
                            "human_origin_verified": bool(source and source.get("human_origin_verified")),
                            "recovery_source_qualified": bool(source and source.get("recovery_source_qualified")),
                            "usability": {
                                "nominal_policy_training": not reasons and bool(source_usable.get("nominal_policy_training")),
                                "recovery_policy_training": not reasons and bool(source_usable.get("recovery_policy_training")),
                                "error_scene_generation_source": False,
                                "canonical_training_scene": False,
                            },
                        }
                    )
    mark_logical_duplicates(rows)
    return rows


def mark_logical_duplicates(rows: list[dict[str, Any]]) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row.get("trajectory_hash"):
            groups[(row.get("task"), row["trajectory_hash"])].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["item_id"])
        for index, row in enumerate(ordered):
            row["logical_duplicate"] = index > 0
            row["logical_primary_id"] = ordered[0]["item_id"]


def audit_clean_trajectories(root: Path) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        task_root = root / task
        if not task_root.exists():
            continue
        manifest_path = task_root / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = {row["trajectory_id"]: row for row in json.load(handle).get("trajectories", [])}
        npz = {path.stem: path for path in task_root.glob("*.npz")}
        for trajectory_id in sorted(set(manifest) | set(npz)):
            entry = manifest.get(trajectory_id)
            path = npz.get(trajectory_id)
            reasons = []
            action_hash = None
            length = 0
            if entry is None:
                reasons.append("missing_manifest_entry")
            if path is None:
                reasons.append("missing_npz")
            else:
                try:
                    with np.load(path, allow_pickle=True) as data:
                        if "actions" not in data or "states" not in data:
                            reasons.append("missing_actions_or_states")
                        else:
                            actions = data["actions"]
                            states = data["states"]
                            length = len(actions)
                            action_hash = _action_hash(actions)
                            if actions.ndim != 2 or actions.shape[1] != 7 or len(states) != length:
                                reasons.append("trajectory_shape_mismatch")
                            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                                reasons.append("non_finite_trajectory")
                except Exception:
                    reasons.append("npz_unreadable")
            if entry and entry.get("num_steps") != length:
                reasons.append("manifest_length_mismatch")
            dataset_path = str((entry or {}).get("dataset_path", ""))
            human_origin = "humandemo" in dataset_path.lower()
            source_has_full_state = "core" in dataset_path.lower() or "datasets/source" in dataset_path.lower()
            if not human_origin:
                reasons.append("source_not_verified_human")
            if not source_has_full_state:
                reasons.append("source_full_sim_state_unverified")
            structural_reasons = [
                reason for reason in reasons if reason not in {"source_not_verified_human", "source_full_sim_state_unverified"}
            ]
            rows.append(
                {
                    "item_id": f"clean:{task}:{trajectory_id}",
                    "stage": "clean_trajectory",
                    "task": task,
                    "trajectory_id": trajectory_id,
                    "source_path": str(path or manifest_path),
                    "source_dataset_path": dataset_path,
                    "source_demo_key": (entry or {}).get("demo_key"),
                    "length": length,
                    "action_hash": action_hash,
                    "human_origin_verified": human_origin,
                    "source_full_sim_state_available": source_has_full_state,
                    "structural_qualified": not structural_reasons,
                    "failure_reasons": sorted(set(reasons)),
                    "usability": {
                        "nominal_policy_training": False,
                        "recovery_policy_training": False,
                        "error_scene_generation_source": not reasons,
                        "canonical_training_scene": False,
                    },
                }
            )
    return rows


def training_lineage_decision(
    scene: dict[str, Any],
    conformance: dict[str, Any] | None,
    clean: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons = []
    if not scene.get("qualified"):
        reasons.append("training_scene_not_qualified")
    status = (conformance or {}).get("generation_conformance", "unverifiable")
    if status != "pass":
        reasons.append(f"generation_conformance_{status}")
    if clean is None:
        reasons.append("clean_trajectory_not_found")
    else:
        if not clean.get("human_origin_verified"):
            reasons.append("clean_source_not_verified_human")
        if not clean.get("usability", {}).get("error_scene_generation_source"):
            reasons.append("clean_source_not_eligible")
    return not reasons, sorted(set(reasons))


def audit_training_lineage(
    report_dir: Path,
    training_root: Path,
    clean_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    training = _load_jsonl(report_dir / "training.jsonl")
    conformance_rows = _load_jsonl(report_dir / "training_filter/training_generation_conformance.jsonl")
    conformance = {row["item_id"]: row for row in conformance_rows}
    clean = {row["trajectory_id"]: row for row in clean_rows}
    rows = []
    for scene in training:
        task = scene.get("task")
        scene_id = scene.get("scene_id")
        metadata_path = training_root / str(task) / "scenes" / f"{scene_id}.json"
        clean_ref = None
        if metadata_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as handle:
                    meta = json.load(handle)
                clean_ref = meta.get("clean_trajectory_ref") or (meta.get("dataset") or {}).get("trajectory_id")
            except Exception:
                pass
        clean_row = clean.get(clean_ref)
        usable, reasons = training_lineage_decision(scene, conformance.get(scene["item_id"]), clean_row)
        rows.append(
            {
                "item_id": f"training_lineage:{scene_id}",
                "stage": "training_scene",
                "task": task,
                "scene_id": scene_id,
                "source_path": str(metadata_path),
                "clean_trajectory_ref": clean_ref,
                "linked_clean_id": clean_row.get("item_id") if clean_row else None,
                "scene_qualified": bool(scene.get("qualified")),
                "generation_conformance": (conformance.get(scene["item_id"]) or {}).get(
                    "generation_conformance", "unverifiable"
                ),
                "human_origin_verified": bool(clean_row and clean_row.get("human_origin_verified")),
                "failure_reasons": reasons,
                "usability": {
                    "nominal_policy_training": False,
                    "recovery_policy_training": False,
                    "error_scene_generation_source": False,
                    "canonical_training_scene": usable,
                },
            }
        )
    return rows


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        by_task[task] = {
            "items": len(task_rows),
            "structural_qualified": sum(bool(row.get("structural_qualified")) for row in task_rows),
            "human_origin_verified": sum(bool(row.get("human_origin_verified")) for row in task_rows),
            "nominal_policy_training": sum(bool(row.get("usability", {}).get("nominal_policy_training")) for row in task_rows),
            "recovery_policy_training": sum(bool(row.get("usability", {}).get("recovery_policy_training")) for row in task_rows),
            "error_scene_generation_source": sum(bool(row.get("usability", {}).get("error_scene_generation_source")) for row in task_rows),
            "canonical_training_scene": sum(bool(row.get("usability", {}).get("canonical_training_scene")) for row in task_rows),
        }
    failures = Counter(reason for row in rows for reason in row.get("failure_reasons", []))
    origins = Counter(row.get("origin_class", "not_applicable") for row in rows)
    return {
        "items": len(rows),
        "by_task": by_task,
        "failure_reasons": dict(sorted(failures.items())),
        "origin_classes": dict(sorted(origins.items())),
    }


def build_summary(
    lerobot_rows: list[dict[str, Any]],
    hdf5_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    claimed = [row for row in lerobot_rows if row.get("dataset_family") == "claimed_humandemo"]
    return {
        "audit_version": "1.0",
        "generated_at": _now(),
        "claimed_humandemo": len(claimed),
        "claimed_humandemo_expected": sum(CLAIMED_HUMANDEMO_COUNTS.values()),
        "pick_place_humandemo_missing": not any(row.get("task") == "pick_place" for row in claimed),
        "logical_hdf5_demos": sum(not row.get("logical_duplicate", False) for row in hdf5_rows),
        "physical_hdf5_demos": len(hdf5_rows),
        "stages": {
            "lerobot": _stage_summary(lerobot_rows),
            "hdf5": _stage_summary(hdf5_rows),
            "clean_trajectory": _stage_summary(clean_rows),
            "training_scene": _stage_summary(training_rows),
        },
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    lines = [
        "# Human-labelled and derived data audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Claimed humandemo episodes: {summary['claimed_humandemo']}",
        f"Pick-place humandemo missing: {summary['pick_place_humandemo_missing']}",
        "",
        "| Stage | Items | Structural pass | Human origin verified | Nominal train | Recovery train | Error-scene source | Canonical Training |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, data in summary["stages"].items():
        totals = {
            key: sum(task[key] for task in data["by_task"].values())
            for key in (
                "structural_qualified",
                "human_origin_verified",
                "nominal_policy_training",
                "recovery_policy_training",
                "error_scene_generation_source",
                "canonical_training_scene",
            )
        }
        lines.append(
            f"| {stage} | {data['items']} | {totals['structural_qualified']} | "
            f"{totals['human_origin_verified']} | {totals['nominal_policy_training']} | "
            f"{totals['recovery_policy_training']} | {totals['error_scene_generation_source']} | "
            f"{totals['canonical_training_scene']} |"
        )
    lines += ["", "## Claimed humandemo by task", "", "| Task | Episodes | Normal MimicGen | Human recovery | Unmatched/ambiguous | Recovery-train usable |", "|---|---:|---:|---:|---:|---:|"]
    claimed_rows = _load_jsonl(output_dir / "lerobot.jsonl")
    claimed_rows = [row for row in claimed_rows if row.get("dataset_family") == "claimed_humandemo"]
    for task in TASKS:
        rows = [row for row in claimed_rows if row.get("task") == task]
        normal = sum(row.get("origin_class") == "normal_mimicgen" for row in rows)
        recovery = sum(row.get("origin_class") == "human_recovery" for row in rows)
        usable = sum(row.get("usability", {}).get("recovery_policy_training", False) for row in rows)
        lines.append(f"| {task} | {len(rows)} | {normal} | {recovery} | {len(rows) - normal - recovery} | {usable} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_report(output_dir: Path, expected_claimed: int) -> int:
    names = ("lerobot", "hdf5", "clean", "training_lineage")
    rows = {name: _load_jsonl(output_dir / f"{name}.jsonl") for name in names}
    missing = sum(not (output_dir / f"{name}.jsonl").exists() for name in names)
    duplicate_records = 0
    for values in rows.values():
        ids = [row.get("item_id") for row in values]
        duplicate_records += len(ids) - len(set(ids))
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
        missing += 1
    claimed = sum(row.get("dataset_family") == "claimed_humandemo" for row in rows["lerobot"])
    missing += int(claimed != expected_claimed)
    missing += int(summary.get("claimed_humandemo") != claimed)
    for name, values in rows.items():
        stage_name = "training_scene" if name == "training_lineage" else ("clean_trajectory" if name == "clean" else name)
        missing += int(summary.get("stages", {}).get(stage_name, {}).get("items") != len(values))
    print(
        f"HUMAN-DERIVED REPORT COMPLETE claimed_humandemo={claimed} "
        f"missing={missing} duplicate_records={duplicate_records}"
    )
    return 0 if missing == 0 and duplicate_records == 0 else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--hdf5-root", type=Path, action="append")
    parser.add_argument("--human-recovery-report", type=Path)
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--training-report-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-report", type=Path)
    parser.add_argument("--expected-claimed-humandemo", type=int, default=10016)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_report:
        return check_report(args.check_report, args.expected_claimed_humandemo)
    required = (
        "lerobot_root",
        "core_root",
        "hdf5_root",
        "human_recovery_report",
        "clean_root",
        "training_root",
        "training_report_dir",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) in (None, [])]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = discover_lerobot_datasets(args.lerobot_root)
    episode_items = [item for dataset in datasets for item in _episode_items(dataset)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        lerobot_rows = list(pool.map(audit_lerobot_episode, episode_items))
    lerobot_rows.sort(key=lambda row: row["item_id"])

    core_refs = load_core_refs(args.core_root)
    recovery_refs = load_recovery_refs(args.human_recovery_report)
    apply_origin_and_usability(lerobot_rows, core_refs, recovery_refs)
    _write_jsonl(args.output_dir / "lerobot.jsonl", lerobot_rows)

    claimed_rows = [row for row in lerobot_rows if row["dataset_family"] == "claimed_humandemo"]
    hdf5_rows = audit_hdf5_files(args.hdf5_root, claimed_rows)
    _write_jsonl(args.output_dir / "hdf5.jsonl", hdf5_rows)

    clean_rows = audit_clean_trajectories(args.clean_root)
    _write_jsonl(args.output_dir / "clean.jsonl", clean_rows)
    training_rows = audit_training_lineage(
        args.training_report_dir, args.training_root, clean_rows
    )
    _write_jsonl(args.output_dir / "training_lineage.jsonl", training_rows)

    summary = build_summary(lerobot_rows, hdf5_rows, clean_rows, training_rows)
    write_summary(args.output_dir, summary)
    print(
        f"HUMAN-DERIVED AUDIT COMPLETE claimed_humandemo_episodes={summary['claimed_humandemo']} "
        f"lerobot={len(lerobot_rows)} hdf5={len(hdf5_rows)} clean={len(clean_rows)} "
        f"training={len(training_rows)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
