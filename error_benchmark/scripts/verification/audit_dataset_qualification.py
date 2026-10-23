#!/usr/bin/env python
"""Audit every Training scene, Eval scene, and Human recovery candidate.

The audit is read-only. Data failures are emitted as result rows; only an
incomplete or internally inconsistent audit is a process failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.env_wrapper import EnvWrapper
from error_benchmark.framework.context_replay import InjectionEngine
from error_benchmark.framework.error_skills import get_skill
from error_benchmark.framework.utils.rollout_utils import collect_rollout_stats
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry


AUDIT_VERSION = "1.1"
TASKS = (
    "coffee",
    "pick_place",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
)
EXPECTED_ROBOT = {task: ("Sawyer" if task == "pick_place" else "Panda") for task in TASKS}
HUMAN_TARGETS = {
    "coffee": 200,
    "pick_place": 240,
    "stack": 240,
    "stack_three": 240,
    "threading": 200,
    "three_piece_assembly": 240,
}
SCENE_QUALITY_CHECKS = (
    "pair_readable",
    "required_fields",
    "finite_post_state",
    "metadata_consistent",
    "robot_consistent",
    "environment_fingerprint",
    "state_restore",
    "not_success_at_start",
    "semantic_evidence",
    "unique_fingerprint",
)
HUMAN_QUALITY_CHECKS = (
    "npz_readable",
    "required_arrays",
    "finite_arrays",
    "lengths_consistent",
    "action_dimension",
    "recorded_success",
    "source_scene_found",
    "source_scene_eligible",
    "source_scene_consistent",
    "source_not_successful",
    "state_replay",
    "state_eef_error",
    "action_replay",
    "final_eef_error",
    "success_streak",
    "release_lift",
)
HUMAN_TRAINING_SUCCESS_GATES = (
    "npz_readable",
    "required_arrays",
    "finite_arrays",
    "lengths_consistent",
    "action_dimension",
    "recorded_success",
    "source_scene_found",
    "source_scene_eligible",
    "source_scene_consistent",
    "source_not_successful",
    "state_replay",
    "state_eef_error",
    "saved_final_state_success",
    "saved_start_action_replay",
    "saved_start_final_success",
    "saved_start_success_streak",
    "saved_start_release_lift",
)
TRAINING_SCENE_ID_RE = re.compile(
    r"^v5_train_(?P<subtype>.+)_(?P<seq>[0-9]{3})_(?P<hash>[0-9a-f]{8})$"
)


_WORKER_ENV = None
_WORKER_TASK = ""
_WORKER_TASK_CONFIG = {}
_WORKER_BENCHMARK_CONFIG = {}
_WORKER_ENGINE = None
_WORKER_ENV_ERROR = ""
_WORKER_TRAINING = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalar(value, default=""):
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _add_reason(row: dict, code: str, detail: str | None = None) -> None:
    if code not in row["failure_reasons"]:
        row["failure_reasons"].append(code)
    if detail:
        row.setdefault("failure_details", {})[code] = str(detail)


def _set_check(row: dict, name: str, passed: bool, reason: str, detail=None) -> None:
    row["checks"][name] = bool(passed)
    if not passed:
        _add_reason(row, reason, detail)


def _trajectory_lengths_ok(actions, states, eef_positions, recorded_steps: int) -> bool:
    """Match the collector layout: T actions and T+1 pre/post states."""
    return bool(
        actions is not None
        and states is not None
        and eef_positions is not None
        and actions.ndim == states.ndim == eef_positions.ndim == 2
        and len(actions) > 0
        and len(states) == len(actions) + 1
        and len(eef_positions) == len(actions) + 1
        and recorded_steps == len(actions)
    )


def _human_source_scene_id(arrays: dict) -> str:
    """Use scene_origin when available, with legacy scene_id fallback."""
    origin = str(_scalar(arrays.get("scene_origin"), "")).strip()
    return origin or str(_scalar(arrays.get("scene_id"), "")).strip()


def _base_row(dataset_type: str, item_id: str, source_path: Path, task="") -> dict:
    return {
        "audit_version": AUDIT_VERSION,
        "dataset_type": dataset_type,
        "item_id": item_id,
        "source_path": str(source_path),
        "task": task,
        "subtype": "",
        "checks": {},
        "qualified": False,
        "canonical_eligible": False,
        "selected": False,
        "coverage_credit": False,
        "failure_reasons": [],
        "warnings": [],
        "augmentation_eligible": "not_evaluated" if dataset_type == "human" else None,
    }


def _infer_task(path: Path, root: Path | None = None) -> str:
    parts = path.parts
    if root is not None:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            pass
    for part in parts:
        if part in TASKS:
            return part
    return ""


def _subtype(meta: dict) -> tuple[str, str, str]:
    labels = meta.get("labels") or {}
    spec = meta.get("error_spec") or {}
    error_name = str(labels.get("error_name") or spec.get("error_name") or "")
    degree = str(labels.get("degree") or spec.get("degree") or "")
    subtype = str(labels.get("subtype_id") or "")
    if not subtype and error_name and degree:
        subtype = f"{error_name}_{degree}"
    return error_name, degree, subtype


def _robot_from_config(config: dict) -> str:
    robots = (config.get("env_args") or {}).get("robots")
    if isinstance(robots, (list, tuple)):
        return str(robots[0]) if robots else ""
    return str(robots or "")


def _scene_fingerprint(task: str, subtype: str, state: np.ndarray) -> str:
    array = np.asarray(state, dtype="<f8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(task.encode("utf-8"))
    digest.update(b"\0")
    digest.update(subtype.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _contains_backfill_marker(obj) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "backfill" in str(key).lower() or _contains_backfill_marker(value):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(_contains_backfill_marker(value) for value in obj)
    return "backfill" in str(obj).lower()


def classify_scene_provenance(dataset_type: str, scene_id: str, meta: dict) -> dict:
    controlled_match = re.search(r"_([0-9]{3})_[0-9a-fA-F]+$", scene_id)
    controlled = scene_id.startswith("v5_train_") and controlled_match is not None
    backfilled = _contains_backfill_marker(meta) or (
        dataset_type == "training" and scene_id.startswith("v5_") and not controlled
    )
    if dataset_type == "training":
        kind = "controlled_training" if controlled else (
            "validation_backfill" if backfilled else "unknown"
        )
        valid = kind == "controlled_training"
    else:
        kind = "validation_generation" if not controlled else "controlled_training"
        valid = kind == "validation_generation"
    return {
        "kind": kind,
        "valid": valid,
        "sequence_index": int(controlled_match.group(1)) if controlled else None,
    }


def _sha256_marker(value) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def evaluate_training_generation(row: dict, meta: dict, task_meta_ids: set[str] | None) -> dict:
    """Classify strict controlled-magnitude evidence for one quality-eligible scene."""
    scene_id = str(meta.get("scene_id") or row.get("scene_id") or "")
    task = str(row.get("task") or "")
    subtype = str(row.get("subtype") or "")
    labels = meta.get("labels") or {}
    error_spec = meta.get("error_spec") or {}
    replay = meta.get("replay") or {}
    match = TRAINING_SCENE_ID_RE.fullmatch(scene_id)
    checks = {}
    failures = []
    missing = []

    def structural(name, passed, reason):
        checks[name] = bool(passed)
        if not passed:
            failures.append(reason)

    structural("quality_eligible", bool(row.get("canonical_eligible")), "not_quality_eligible")
    structural("regular_scene_id", match is not None, "non_regular_training_scene_id")
    structural(
        "scene_id_consistent",
        scene_id == str(row.get("scene_id") or "") == Path(row.get("source_path", "")).stem,
        "scene_id_mismatch",
    )
    structural(
        "subtype_consistent",
        bool(match)
        and match.group("subtype") == subtype
        and labels.get("subtype_id") == subtype,
        "subtype_mismatch",
    )

    seq = int(match.group("seq")) if match else None
    structural(
        "sequence_consistent",
        seq is not None and labels.get("training_seq") == seq,
        "training_sequence_mismatch",
    )
    clean_ref = meta.get("clean_trajectory_ref")
    injection_step = meta.get("injection_step")
    lineage_ok = bool(clean_ref) and (
        (meta.get("dataset") or {}).get("trajectory_id")
        == error_spec.get("source_trajectory")
        == clean_ref
    )
    structural("clean_trajectory_lineage", lineage_ok, "clean_trajectory_lineage_mismatch")
    step_ok = injection_step is not None and (
        replay.get("injection_frame")
        == error_spec.get("source_frame")
        == (error_spec.get("apply") or {}).get("t")
        == injection_step
    )
    structural("injection_step_lineage", step_ok, "injection_step_lineage_mismatch")

    expected_hash = None
    if match and clean_ref and injection_step is not None:
        hash_input = f"train_{task}_{subtype}_{seq:03d}_{clean_ref}_{injection_step}"
        expected_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:8]
    structural(
        "scene_id_hash",
        bool(match) and expected_hash == match.group("hash"),
        "scene_id_hash_mismatch",
    )
    if task_meta_ids is None:
        checks["task_meta_membership"] = None
        missing.append("task_meta_missing")
    else:
        structural(
            "task_meta_membership",
            scene_id in task_meta_ids,
            "task_meta_membership_missing",
        )
    structural(
        "not_coverage_repair",
        not labels.get("coverage_repair") and "_repair_" not in scene_id,
        "coverage_repair_uncontrolled",
    )

    evidence_fields = {
        "controlled_magnitude_mode": "generation_mode",
        "training_point_index": "training_point_index",
        "magnitude_fraction": "magnitude_fraction",
        "scenes_per_subtype": "scenes_per_subtype",
        "injection_seed": "injection_seed",
        "injection_attempt": "injection_attempt",
        "generator_version": "generator_version",
        "generator_sha256": "generator_sha256",
        "config_sha256": "config_sha256",
    }
    for check_name, field in evidence_fields.items():
        if field not in labels:
            checks[check_name] = None
            missing.append(f"missing_{field}")

    if "generation_mode" in labels:
        structural(
            "controlled_magnitude_mode",
            labels.get("generation_mode") == "controlled_magnitude",
            "generation_mode_mismatch",
        )
    n = labels.get("scenes_per_subtype")
    point_index = labels.get("training_point_index")
    if "scenes_per_subtype" in labels:
        structural(
            "scenes_per_subtype",
            isinstance(n, int) and not isinstance(n, bool) and n == 10,
            "scenes_per_subtype_mismatch",
        )
    if "training_point_index" in labels:
        structural(
            "training_point_index",
            isinstance(point_index, int)
            and not isinstance(point_index, bool)
            and isinstance(n, int)
            and 0 <= point_index < n,
            "training_point_index_invalid",
        )
    if "magnitude_fraction" in labels:
        expected_fraction = (
            point_index / max(n - 1, 1)
            if isinstance(point_index, int) and isinstance(n, int) and n > 0
            else None
        )
        actual_fraction = labels.get("magnitude_fraction")
        fraction_ok = (
            isinstance(actual_fraction, (int, float))
            and not isinstance(actual_fraction, bool)
            and expected_fraction is not None
            and abs(float(actual_fraction) - expected_fraction) <= 1e-12
        )
        structural("magnitude_fraction", fraction_ok, "magnitude_fraction_mismatch")
    for field in ("injection_seed", "injection_attempt"):
        if field in labels:
            value = labels.get(field)
            structural(
                field,
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{field}_invalid",
            )
    if "generator_version" in labels:
        structural("generator_version", bool(labels.get("generator_version")), "generator_version_invalid")
    if "generator_sha256" in labels:
        structural("generator_sha256", _sha256_marker(labels.get("generator_sha256")), "generator_sha256_invalid")
    if "config_sha256" in labels:
        structural("config_sha256", _sha256_marker(labels.get("config_sha256")), "config_sha256_invalid")

    status = "fail" if failures else ("unverifiable" if missing else "pass")
    return {
        "audit_version": AUDIT_VERSION,
        "dataset_type": "training",
        "item_id": row.get("item_id"),
        "scene_id": scene_id,
        "source_path": row.get("source_path"),
        "task": task,
        "subtype": subtype,
        "quality_eligible": bool(row.get("canonical_eligible")),
        "generation_conformance": status,
        "checks": checks,
        "failure_reasons": sorted(set(failures)),
        "unverifiable_reasons": sorted(set(missing)),
        "training_point_index": point_index,
        "magnitude_fraction": labels.get("magnitude_fraction"),
        "selected": False,
    }


def discover_scene_candidates(root: Path, dataset_type: str, tasks=None) -> list[dict]:
    root = root.resolve()
    allowed = set(tasks or TASKS)
    candidates = []
    skip_names = {"meta.json", "summary.json", "manifest.json"}
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name in skip_names or json_path.name.endswith("_manifest.json"):
            continue
        if not json_path.name.startswith("v5_"):
            continue
        task = _infer_task(json_path, root)
        if tasks and task not in allowed:
            continue
        relative = json_path.relative_to(root).as_posix()
        candidates.append({
            "dataset_type": dataset_type,
            "item_id": f"{dataset_type}:{relative[:-5]}",
            "json_path": str(json_path),
            "npz_path": str(json_path.with_suffix(".npz")),
            "task": task,
        })
    return candidates


def discover_human_candidates(root: Path, tasks=None) -> list[dict]:
    root = root.resolve()
    allowed = set(tasks or TASKS)
    candidates = []
    for npz_path in sorted(root.rglob("*.npz")):
        task = _infer_task(npz_path, root)
        if tasks and task not in allowed:
            continue
        relative = npz_path.relative_to(root).as_posix()
        candidates.append({
            "dataset_type": "human",
            "item_id": f"human:{relative[:-4]}",
            "npz_path": str(npz_path),
            "task": task,
        })
    return candidates


def _worker_init(task: str, training_lookup=None) -> None:
    global _WORKER_ENV, _WORKER_TASK, _WORKER_TASK_CONFIG
    global _WORKER_BENCHMARK_CONFIG, _WORKER_ENGINE
    global _WORKER_ENV_ERROR, _WORKER_TRAINING
    if _WORKER_ENV is not None:
        try:
            _WORKER_ENV._env.close()
        except Exception:
            pass
    _WORKER_ENV = None
    _WORKER_TASK = task
    _WORKER_TASK_CONFIG = {}
    _WORKER_BENCHMARK_CONFIG = {}
    _WORKER_ENGINE = None
    _WORKER_ENV_ERROR = ""
    _WORKER_TRAINING = training_lookup or {}
    try:
        import yaml

        info = load_task_registry(task)
        config_path = Path(info["task_config"])
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        with open(config_path, encoding="utf-8") as handle:
            _WORKER_TASK_CONFIG = yaml.safe_load(handle)
        with open(PROJECT_ROOT / "error_benchmark/configs/benchmark_v5.yaml", encoding="utf-8") as handle:
            _WORKER_BENCHMARK_CONFIG = yaml.safe_load(handle)
        env = create_env(
            _WORKER_TASK_CONFIG,
            info["dataset_path"],
            enable_camera=False,
            has_renderer=False,
        )
        _WORKER_ENV = EnvWrapper(env, _WORKER_TASK_CONFIG)
        _WORKER_ENGINE = InjectionEngine(_WORKER_ENV, _WORKER_TASK_CONFIG)
    except Exception as exc:
        _WORKER_ENV_ERROR = f"{type(exc).__name__}: {exc}"


def _recompute_scene_decision(row: dict) -> None:
    row["qualified"] = all(row["checks"].get(name) is True for name in SCENE_QUALITY_CHECKS)
    provenance_valid = bool((row.get("provenance") or {}).get("valid"))
    overlap = "overlap_validation_pool" in row["failure_reasons"]
    row["canonical_eligible"] = row["qualified"] and provenance_valid and not overlap


def _audit_scene_candidate(candidate: dict) -> dict:
    started = time.monotonic()
    dataset_type = candidate["dataset_type"]
    json_path = Path(candidate["json_path"])
    npz_path = Path(candidate["npz_path"])
    row = _base_row(dataset_type, candidate["item_id"], json_path, candidate.get("task", ""))
    row["npz_path"] = str(npz_path)
    meta = {}
    pre_state = None
    post_state = None
    try:
        try:
            with open(json_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            json_ok = isinstance(meta, dict)
        except Exception as exc:
            json_ok = False
            _add_reason(row, "json_unreadable", f"{type(exc).__name__}: {exc}")

        npz_ok = False
        if not npz_path.exists():
            _add_reason(row, "missing_pair", str(npz_path))
        else:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "sim_state" in data.files:
                        pre_state = np.asarray(data["sim_state"])
                    if "post_sim_state" in data.files:
                        post_state = np.asarray(data["post_sim_state"])
                    npz_ok = True
            except Exception as exc:
                _add_reason(row, "npz_unreadable", f"{type(exc).__name__}: {exc}")
        _set_check(row, "pair_readable", json_ok and npz_ok, "missing_pair")

        scene_id = str(meta.get("scene_id") or json_path.stem)
        error_name, degree, subtype = _subtype(meta)
        meta_task = str(meta.get("task") or meta.get("task_name") or row["task"])
        row.update({
            "scene_id": scene_id,
            "task": meta_task,
            "subtype": subtype,
            "error_name": error_name,
            "degree": degree,
        })
        scene_id_match = bool(meta.get("scene_id")) and scene_id == json_path.stem
        required = bool(scene_id and error_name and degree and subtype and meta.get("env_fingerprint"))
        if post_state is None:
            _add_reason(row, "missing_post_sim_state")
        _set_check(
            row,
            "required_fields",
            required and post_state is not None,
            "missing_required_field",
        )
        if not scene_id_match:
            _add_reason(row, "scene_id_mismatch", f"json={scene_id} file={json_path.stem}")

        finite_state = post_state is not None and post_state.size > 0 and bool(np.isfinite(post_state).all())
        _set_check(row, "finite_post_state", finite_state, "non_finite_array")
        if finite_state:
            row["fingerprint"] = _scene_fingerprint(meta_task, subtype, post_state)

        metadata_consistent = (
            scene_id_match
            and bool(meta_task)
            and meta_task == candidate.get("task")
            and subtype == f"{error_name}_{degree}"
        )
        _set_check(row, "metadata_consistent", metadata_consistent, "metadata_mismatch")
        if meta_task != candidate.get("task"):
            _add_reason(row, "task_mismatch", f"metadata={meta_task} path={candidate.get('task')}")
        if subtype != f"{error_name}_{degree}":
            _add_reason(row, "subtype_mismatch")

        provenance = classify_scene_provenance(dataset_type, scene_id, meta)
        row["provenance"] = provenance
        if not provenance["valid"]:
            _add_reason(
                row,
                "training_provenance_invalid" if dataset_type == "training" else "eval_provenance_invalid",
                provenance["kind"],
            )

        current_robot = _robot_from_config(_WORKER_TASK_CONFIG)
        expected_robot = EXPECTED_ROBOT.get(meta_task, "")
        robot_ok = bool(current_robot and current_robot == expected_robot)
        _set_check(row, "robot_consistent", robot_ok, "robot_mismatch", f"current={current_robot} expected={expected_robot}")

        if _WORKER_ENV is None:
            _set_check(row, "environment_fingerprint", False, "environment_init_error", _WORKER_ENV_ERROR)
            _set_check(row, "state_restore", False, "environment_init_error", _WORKER_ENV_ERROR)
            _set_check(row, "not_success_at_start", False, "environment_init_error", _WORKER_ENV_ERROR)
        else:
            try:
                EnvWrapper.verify_fingerprint(_WORKER_ENV.get_fingerprint(), meta.get("env_fingerprint") or {})
                fingerprint_ok = True
            except Exception as exc:
                fingerprint_ok = False
                fingerprint_detail = f"{type(exc).__name__}: {exc}"
            _set_check(
                row,
                "environment_fingerprint",
                fingerprint_ok,
                "fingerprint_mismatch",
                None if fingerprint_ok else fingerprint_detail,
            )
            restore_ok = False
            already_success = None
            if post_state is not None:
                try:
                    _WORKER_ENV.set_sim_state_flat(post_state)
                    _WORKER_ENV.forward()
                    restore_ok = True
                    already_success = bool(_WORKER_ENV.check_success())
                except Exception as exc:
                    restore_detail = f"{type(exc).__name__}: {exc}"
            _set_check(
                row,
                "state_restore",
                restore_ok,
                "state_restore_error",
                None if restore_ok else locals().get("restore_detail", "missing state"),
            )
            _set_check(
                row,
                "not_success_at_start",
                restore_ok and already_success is False,
                "scene_already_successful" if already_success else "state_restore_error",
            )

        validated_by = meta.get("validated_by") or {}
        recorded_semantic_ok = (
            isinstance(validated_by, dict)
            and validated_by.get("skill") == error_name
            and "reason" in validated_by
        )
        current_semantic_ok = False
        if _WORKER_ENV is not None and _WORKER_ENGINE is not None and pre_state is not None and post_state is not None:
            try:
                _WORKER_ENV.set_sim_state_flat(pre_state)
                _WORKER_ENV.forward()
                pre_state_info = _WORKER_ENGINE._extract_state_info()
                target_object = (
                    (meta.get("labels") or {}).get("target_object")
                    or ((meta.get("error_spec") or {}).get("target") or {}).get("body")
                    or ""
                )
                pre_state_info["target_object"] = target_object
                _WORKER_ENV.set_sim_state_flat(post_state)
                _WORKER_ENV.forward()
                object_names = _WORKER_ENV.get_all_object_names()
                stats_object = target_object if target_object in object_names else (
                    object_names[0] if object_names else None
                )
                post_stats = collect_rollout_stats(
                    env=_WORKER_ENV,
                    obj_name=stats_object,
                    steps=20,
                    pre_state_info=pre_state_info,
                ) if stats_object else None
                skill_config = (
                    (_WORKER_BENCHMARK_CONFIG.get("error_skills") or {}).get(error_name) or {}
                )
                validation = get_skill(error_name, skill_config).validate(
                    env=_WORKER_ENV,
                    pre_state=pre_state_info,
                    post_stats=post_stats,
                    degree=degree,
                )
                current_semantic_ok = bool(validation.ok)
                row["semantic_validation"] = {
                    "ok": current_semantic_ok,
                    "reason": validation.reason,
                    "metrics": validation.metrics,
                }
            except Exception as exc:
                row["semantic_validation"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        semantic_ok = recorded_semantic_ok and current_semantic_ok
        _set_check(
            row,
            "semantic_evidence",
            semantic_ok,
            "missing_semantic_evidence" if not recorded_semantic_ok else "semantic_revalidation_failed",
            (row.get("semantic_validation") or {}).get("reason")
            or (row.get("semantic_validation") or {}).get("error"),
        )
        if not (meta.get("labels") or {}).get("validation_metrics"):
            row["warnings"].append("missing_validation_metrics")
        row["checks"]["unique_fingerprint"] = True
        row["historical_compatible"] = bool(candidate.get("historical_compatible", False))
        _recompute_scene_decision(row)
    except Exception as exc:
        _add_reason(row, "unexpected_exception", f"{type(exc).__name__}: {exc}")
        row["qualified"] = False
        row["canonical_eligible"] = False
    row["failure_reasons"].sort()
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _recompute_human_decision(row: dict) -> None:
    row["qualified"] = all(row["checks"].get(name) is True for name in HUMAN_QUALITY_CHECKS)
    row["ground_truth_qualified"] = row["qualified"]
    row["canonical_eligible"] = row["qualified"]
    row["training_success_qualified"] = all(
        row["checks"].get(name) is True for name in HUMAN_TRAINING_SUCCESS_GATES
    )


def _audit_human_candidate(candidate: dict) -> dict:
    started = time.monotonic()
    npz_path = Path(candidate["npz_path"])
    row = _base_row("human", candidate["item_id"], npz_path, candidate.get("task", ""))
    arrays = {}
    try:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                for key in data.files:
                    arrays[key] = np.asarray(data[key])
            readable = True
        except Exception as exc:
            readable = False
            _add_reason(row, "npz_unreadable", f"{type(exc).__name__}: {exc}")
        _set_check(row, "npz_readable", readable, "npz_unreadable")

        task = str(_scalar(arrays.get("task_name"), row["task"]))
        error_name = str(_scalar(arrays.get("error_name"), ""))
        degree = str(_scalar(arrays.get("degree"), ""))
        subtype = f"{error_name}_{degree}" if error_name and degree else ""
        scene_id = _human_source_scene_id(arrays)
        demo_id = str(_scalar(arrays.get("demo_id"), npz_path.stem))
        row.update({
            "demo_id": demo_id,
            "task": task,
            "subtype": subtype,
            "error_name": error_name,
            "degree": degree,
            "source_scene_id": scene_id,
        })

        actions = arrays.get("actions")
        states = arrays.get("states")
        eef_positions = arrays.get("eef_positions")
        required_arrays = actions is not None and states is not None and eef_positions is not None
        _set_check(row, "required_arrays", required_arrays, "missing_required_field")
        numeric_arrays = [array for array in (actions, states, eef_positions) if array is not None]
        finite = required_arrays and all(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all() for array in numeric_arrays)
        _set_check(row, "finite_arrays", finite, "non_finite_array")

        action_count = len(actions) if actions is not None and actions.ndim >= 1 else 0
        state_count = len(states) if states is not None and states.ndim >= 1 else 0
        eef_count = len(eef_positions) if eef_positions is not None and eef_positions.ndim >= 1 else 0
        recorded_steps = int(_scalar(arrays.get("num_steps"), -1))
        lengths_ok = _trajectory_lengths_ok(actions, states, eef_positions, recorded_steps)
        row["metrics"] = {
            "action_count": action_count,
            "state_count": state_count,
            "eef_count": eef_count,
            "recorded_steps": recorded_steps,
        }
        _set_check(row, "lengths_consistent", lengths_ok, "length_mismatch")

        expected_action_dim = _WORKER_ENV.get_action_dim() if _WORKER_ENV is not None else None
        action_dim_ok = actions is not None and actions.ndim == 2 and expected_action_dim == actions.shape[1]
        _set_check(
            row,
            "action_dimension",
            action_dim_ok,
            "action_dimension_mismatch",
            f"recorded={actions.shape[1] if actions is not None and actions.ndim == 2 else None} expected={expected_action_dim}",
        )
        recorded_success = bool(_scalar(arrays.get("success"), False))
        _set_check(row, "recorded_success", recorded_success, "recorded_failure")

        source = _WORKER_TRAINING.get(scene_id)
        _set_check(row, "source_scene_found", source is not None, "source_scene_missing", scene_id)
        source_eligible = bool(source and source.get("qualified") and source.get("canonical_eligible"))
        _set_check(row, "source_scene_eligible", source_eligible, "source_scene_ineligible")
        source_consistent = bool(
            source
            and source.get("task") == task
            and source.get("subtype") == subtype
        )
        _set_check(row, "source_scene_consistent", source_consistent, "source_scene_mismatch")

        source_not_successful = False
        source_state = None
        source_npz_missing = False
        restore_ok = False
        if _WORKER_ENV is None:
            _add_reason(row, "environment_init_error", _WORKER_ENV_ERROR)
        elif source:
            source_npz = Path(source["npz_path"])
            if not source_npz.exists():
                # A vanished source scene is ONE root cause, not six demo faults
                # (2026-07-23: stale index + pruned scenes produced a 0/1360
                # all-fail run through exactly this cascade).
                source_npz_missing = True
                _add_reason(row, "source_scene_npz_missing", str(source_npz))
            else:
                try:
                    with np.load(source_npz, allow_pickle=False) as scene_data:
                        source_state = np.asarray(scene_data["post_sim_state"])
                    _WORKER_ENV.set_sim_state_flat(source_state)
                    _WORKER_ENV.forward()
                    source_not_successful = not bool(_WORKER_ENV.check_success())
                    restore_ok = True
                except Exception as exc:
                    source_state = None
                    _add_reason(row, "state_restore_error", f"{type(exc).__name__}: {exc}")
        if source is None:
            not_successful_reason = "source_scene_missing"
        elif source_npz_missing:
            not_successful_reason = "source_scene_npz_missing"
        elif not restore_ok:
            not_successful_reason = "state_restore_error"
        else:
            # Restore genuinely succeeded and check_success() was already true.
            not_successful_reason = "scene_already_successful"
        _set_check(
            row,
            "source_not_successful",
            source_not_successful,
            not_successful_reason,
        )

        state_replay_ok = False
        max_state_error = float("inf")
        saved_final_state_success = False
        if _WORKER_ENV is not None and required_arrays and finite and states.ndim == 2:
            errors = []
            try:
                for index, state in enumerate(states):
                    _WORKER_ENV.set_sim_state_flat(state)
                    _WORKER_ENV.forward()
                    errors.append(float(np.linalg.norm(_WORKER_ENV.get_eef_pos() - eef_positions[index])))
                    if index == len(states) - 1:
                        saved_final_state_success = bool(_WORKER_ENV.check_success())
                state_replay_ok = True
                max_state_error = max(errors) if errors else float("inf")
            except Exception as exc:
                _add_reason(row, "state_replay_error", f"{type(exc).__name__}: {exc}")
        _set_check(row, "state_replay", state_replay_ok, "state_replay_error")
        state_eef_ok = state_replay_ok and max_state_error <= 0.01
        _set_check(row, "state_eef_error", state_eef_ok, "state_eef_error", max_state_error)
        _set_check(
            row,
            "saved_final_state_success",
            state_replay_ok and saved_final_state_success,
            "saved_final_state_not_successful",
        )
        row["metrics"]["max_state_eef_error_m"] = max_state_error if np.isfinite(max_state_error) else None

        action_replay_ok = False
        final_error = float("inf")
        trailing_streak = 0
        max_streak = 0
        release_lift_ok = False
        replay_attempted = (
            _WORKER_ENV is not None and source_state is not None and lengths_ok and finite
        )
        if replay_attempted:
            try:
                # states[0] is pre-action; eef_positions[T] follows actions[T-1].
                # Replay the full action sequence from the qualified source scene.
                _WORKER_ENV.set_sim_state_flat(source_state)
                _WORKER_ENV.forward()
                for action in actions:
                    _WORKER_ENV.step(action)
                    if _WORKER_ENV.check_success():
                        trailing_streak += 1
                        max_streak = max(max_streak, trailing_streak)
                    else:
                        trailing_streak = 0
                final_error = float(np.linalg.norm(_WORKER_ENV.get_eef_pos() - eef_positions[-1]))
                action_replay_ok = True
                if trailing_streak >= 10:
                    _WORKER_ENV.set_gripper_state(1.0, steps=10)
                    _WORKER_ENV.apply_eef_offset(np.array([0.0, 0.0, 0.10]), max_steps=30)
                    release_lift_ok = bool(_WORKER_ENV.check_success())
            except Exception as exc:
                _add_reason(row, "action_replay_error", f"{type(exc).__name__}: {exc}")
        # When the source-based replay could not even be attempted (missing or
        # unrestorable source scene, bad lengths), the four gates still fail —
        # unverifiable is never pass — but they share ONE diagnostic reason
        # instead of fanning out into four fake independent failures.
        _skip = "action_replay_not_attempted"
        _set_check(row, "action_replay", action_replay_ok,
                   "action_replay_error" if replay_attempted else _skip)
        final_error_ok = action_replay_ok and final_error <= 0.05
        _set_check(row, "final_eef_error", final_error_ok,
                   "final_eef_error" if replay_attempted else _skip, final_error)
        _set_check(row, "success_streak", trailing_streak >= 10,
                   "success_streak_too_short" if replay_attempted else _skip, trailing_streak)
        _set_check(row, "release_lift", release_lift_ok,
                   "release_lift_failed" if replay_attempted else _skip)
        row["metrics"].update({
            "final_eef_error_m": final_error if np.isfinite(final_error) else None,
            "trailing_success_streak": trailing_streak,
            "max_success_streak": max_streak,
        })

        saved_action_replay_ok = False
        saved_final_success = False
        saved_final_error = float("inf")
        saved_trailing_streak = 0
        saved_max_streak = 0
        saved_release_lift_evaluated = False
        saved_release_lift_ok = False
        if _WORKER_ENV is not None and lengths_ok and finite and states is not None:
            try:
                _WORKER_ENV.set_sim_state_flat(states[0])
                _WORKER_ENV.forward()
                for action in actions:
                    _WORKER_ENV.step(action)
                    if _WORKER_ENV.check_success():
                        saved_trailing_streak += 1
                        saved_max_streak = max(saved_max_streak, saved_trailing_streak)
                    else:
                        saved_trailing_streak = 0
                saved_final_success = bool(_WORKER_ENV.check_success())
                saved_final_error = float(
                    np.linalg.norm(_WORKER_ENV.get_eef_pos() - eef_positions[-1])
                )
                saved_action_replay_ok = True
                if saved_trailing_streak >= 10:
                    saved_release_lift_evaluated = True
                    _WORKER_ENV.set_gripper_state(1.0, steps=10)
                    _WORKER_ENV.apply_eef_offset(
                        np.array([0.0, 0.0, 0.10]),
                        max_steps=30,
                    )
                    saved_release_lift_ok = bool(_WORKER_ENV.check_success())
            except Exception as exc:
                _add_reason(
                    row,
                    "saved_start_action_replay_error",
                    f"{type(exc).__name__}: {exc}",
                )
        _set_check(
            row,
            "saved_start_action_replay",
            saved_action_replay_ok,
            "saved_start_action_replay_error",
        )
        _set_check(
            row,
            "saved_start_final_success",
            saved_action_replay_ok and saved_final_success,
            "saved_start_final_not_successful",
        )
        _set_check(
            row,
            "saved_start_success_streak",
            saved_trailing_streak >= 10,
            "saved_start_success_streak_too_short",
            saved_trailing_streak,
        )
        row["checks"]["saved_start_release_lift_evaluated"] = saved_release_lift_evaluated
        if saved_release_lift_evaluated:
            _set_check(
                row,
                "saved_start_release_lift",
                saved_release_lift_ok,
                "saved_start_release_lift_failed",
            )
        else:
            row["checks"]["saved_start_release_lift"] = None
        row["metrics"].update({
            "saved_start_final_eef_error_m": (
                saved_final_error if np.isfinite(saved_final_error) else None
            ),
            "saved_start_trailing_success_streak": saved_trailing_streak,
            "saved_start_max_success_streak": saved_max_streak,
            "saved_start_release_lift_evaluated": saved_release_lift_evaluated,
        })
        _recompute_human_decision(row)
    except Exception as exc:
        _add_reason(row, "unexpected_exception", f"{type(exc).__name__}: {exc}")
        row["qualified"] = False
        row["ground_truth_qualified"] = False
        row["canonical_eligible"] = False
    row["failure_reasons"].sort()
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _load_jsonl(path: Path, required_version: str | None = AUDIT_VERSION) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                (required_version is None or row.get("audit_version") == required_version)
                and row.get("item_id")
            ):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: item["item_id"]):
            handle.write(json.dumps(row, sort_keys=True, default=_jsonable) + "\n")
    temp.replace(path)


def _audit_candidates(candidates, worker, output_path: Path, workers: int, training_lookup=None, resume=True):
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    existing_rows = _load_jsonl(output_path) if resume else []
    existing_rows += _load_jsonl(partial_path) if resume else []
    by_id = {row["item_id"]: row for row in existing_rows}
    pending = [candidate for candidate in candidates if candidate["item_id"] not in by_id]
    grouped = defaultdict(list)
    for candidate in pending:
        grouped[candidate.get("task") or "unknown"].append(candidate)

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, "a", encoding="utf-8", newline="\n") as partial:
        for task in sorted(grouped):
            task_candidates = grouped[task]
            if workers <= 1:
                _worker_init(task, training_lookup)
                iterator = map(worker, task_candidates)
                pool = None
            else:
                context = multiprocessing.get_context("spawn")
                pool = context.Pool(
                    processes=workers,
                    initializer=_worker_init,
                    initargs=(task, training_lookup),
                )
                iterator = pool.imap(worker, task_candidates, chunksize=16)
            try:
                for index, row in enumerate(iterator, 1):
                    by_id[row["item_id"]] = row
                    partial.write(json.dumps(row, sort_keys=True, default=_jsonable) + "\n")
                    if index % 100 == 0:
                        partial.flush()
                        print(
                            f"HUMAN AUDIT PROGRESS task={task} "
                            f"processed={index}/{len(task_candidates)}",
                            flush=True,
                        )
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
    rows = [by_id[candidate["item_id"]] for candidate in candidates if candidate["item_id"] in by_id]
    _write_jsonl(output_path, rows)
    partial_path.unlink(missing_ok=True)
    return rows


def _mark_duplicate_fingerprints(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row.get("fingerprint"):
            groups[row["fingerprint"]].append(row)
    for duplicates in groups.values():
        if len(duplicates) < 2:
            continue
        ids = sorted(row["item_id"] for row in duplicates)
        for row in duplicates:
            row["checks"]["unique_fingerprint"] = False
            _add_reason(row, "duplicate_fingerprint", ",".join(ids))
            _recompute_scene_decision(row)


def finalize_scene_records(training: list[dict], evaluation: list[dict]) -> None:
    _mark_duplicate_fingerprints(training)
    _mark_duplicate_fingerprints(evaluation)
    eval_fingerprints = {row.get("fingerprint") for row in evaluation if row.get("fingerprint")}
    for row in training:
        if row.get("fingerprint") in eval_fingerprints:
            _add_reason(row, "overlap_validation_pool")
        _recompute_scene_decision(row)
        row["failure_reasons"].sort()
    for row in evaluation:
        _recompute_scene_decision(row)
        row["failure_reasons"].sort()


def build_training_lookup(rows: list[dict]) -> dict:
    lookup = {}
    for row in sorted(rows, key=lambda item: item["item_id"]):
        for alias in {row.get("scene_id"), Path(row.get("npz_path", "")).stem}:
            if alias and alias not in lookup:
                lookup[str(alias)] = row
    return lookup


def assign_coverage_credit(rows: list[dict]) -> None:
    credited = set()
    for row in sorted(rows, key=lambda item: (item.get("source_scene_id", ""), item["item_id"])):
        row["coverage_credit"] = False
        source = row.get("source_scene_id")
        if row.get("qualified") and source and source not in credited:
            row["coverage_credit"] = True
            credited.add(source)
        elif row.get("qualified") and source:
            row["warnings"].append("duplicate_source_no_coverage")


def _collect_scene_ids(obj, result: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "scene_id" and isinstance(value, str):
                result.add(value)
            else:
                _collect_scene_ids(value, result)
    elif isinstance(obj, list):
        for value in obj:
            _collect_scene_ids(value, result)


def load_historical_scene_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    result = set()
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as handle:
                _collect_scene_ids(json.load(handle), result)
        except (OSError, ValueError):
            continue
    return result


def _dataset_summary(rows: list[dict]) -> dict:
    by_task_subtype = defaultdict(lambda: Counter())
    failures = Counter()
    for row in rows:
        cell = f"{row.get('task')}/{row.get('subtype')}"
        bucket = by_task_subtype[cell]
        bucket["discovered"] += 1
        bucket["qualified"] += int(bool(row.get("qualified")))
        bucket["canonical_eligible"] += int(bool(row.get("canonical_eligible")))
        bucket["coverage_credit"] += int(bool(row.get("coverage_credit")))
        failures.update(row.get("failure_reasons") or [])
    return {
        "discovered": len(rows),
        "processed": len(rows),
        "qualified": sum(bool(row.get("qualified")) for row in rows),
        "canonical_eligible": sum(bool(row.get("canonical_eligible")) for row in rows),
        "coverage_credit": sum(bool(row.get("coverage_credit")) for row in rows),
        "failure_reasons": dict(sorted(failures.items())),
        "by_task_subtype": {
            cell: dict(counts) for cell, counts in sorted(by_task_subtype.items())
        },
    }


def _cumulative_gate_counts(rows: list[dict]) -> dict:
    counts = {"total": len(rows)}
    remaining = list(rows)
    for gate in HUMAN_TRAINING_SUCCESS_GATES:
        remaining = [row for row in remaining if row.get("checks", {}).get(gate) is True]
        counts[gate] = len(remaining)
    return counts


def build_human_training_success_funnel(human: list[dict]) -> dict:
    by_task = {}
    for task in TASKS:
        rows = [row for row in human if row.get("task") == task]
        by_task[task] = _cumulative_gate_counts(rows)
    cells = sorted({
        f"{row.get('task')}/{row.get('subtype')}"
        for row in human
        if row.get("task") and row.get("subtype")
    })
    return {
        "gate_order": list(HUMAN_TRAINING_SUCCESS_GATES),
        "overall": _cumulative_gate_counts(human),
        "by_task": by_task,
        "by_task_subtype": {
            cell: _cumulative_gate_counts([
                row
                for row in human
                if f"{row.get('task')}/{row.get('subtype')}" == cell
            ])
            for cell in cells
        },
    }


def build_summary(training: list[dict], evaluation: list[dict], human: list[dict]) -> dict:
    datasets = {
        "training": _dataset_summary(training),
        "eval": _dataset_summary(evaluation),
        "human": _dataset_summary(human),
    }
    cells = sorted({
        f"{row.get('task')}/{row.get('subtype')}"
        for row in evaluation
        if row.get("task") and row.get("subtype")
    })
    gaps = {}
    for name in ("training", "eval", "human"):
        field = "coverage_credit" if name == "human" else "canonical_eligible"
        rows = {"training": training, "eval": evaluation, "human": human}[name]
        counts = Counter(
            f"{row.get('task')}/{row.get('subtype')}"
            for row in rows
            if row.get(field)
        )
        gaps[name] = {
            cell: max(0, 10 - counts[cell])
            for cell in cells
            if counts[cell] < 10
        }
    historical = [row for row in evaluation if row.get("historical_compatible")]
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": _now(),
        "cell_universe_count": len(cells),
        "datasets": datasets,
        "gaps_to_ten_per_cell": gaps,
        "historical_eval": {
            "identified": len(historical),
            "qualified": sum(bool(row.get("qualified")) for row in historical),
            "canonical_eligible": sum(bool(row.get("canonical_eligible")) for row in historical),
        },
    }


def write_summary(output_dir: Path, summary: dict) -> None:
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    lines = [
        "# Dataset qualification audit",
        "",
        f"Generated: {summary['generated_at']}",
        f"Observed task/subtype cells: {summary['cell_universe_count']}",
        "",
        "| Dataset | Processed | Qualified | Canonical eligible | Coverage credit |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("training", "eval", "human"):
        data = summary["datasets"][name]
        lines.append(
            f"| {name} | {data['processed']} | {data['qualified']} | "
            f"{data['canonical_eligible']} | {data['coverage_credit']} |"
        )
    lines += ["", "## Historical eval compatibility", ""]
    historical = summary["historical_eval"]
    lines.append(
        f"Identified {historical['identified']}; qualified {historical['qualified']}; "
        f"canonical eligible {historical['canonical_eligible']}."
    )
    lines += ["", "## Largest failure reasons", ""]
    failures = Counter()
    for data in summary["datasets"].values():
        failures.update(data["failure_reasons"])
    for reason, count in failures.most_common(20):
        lines.append(f"- `{reason}`: {count}")
    lines += ["", "## Gaps to ten per cell", ""]
    for name, gaps in summary["gaps_to_ten_per_cell"].items():
        lines.append(f"- {name}: {sum(gaps.values())} missing across {len(gaps)} cells")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_human_summary(training: list[dict], human: list[dict]) -> dict:
    target_cells = sorted({
        f"{row.get('task')}/{row.get('subtype')}"
        for row in training
        if row.get("task") and row.get("subtype")
    })
    coverage = Counter(
        f"{row.get('task')}/{row.get('subtype')}"
        for row in human
        if row.get("coverage_credit")
    )
    eligible_sources = defaultdict(set)
    for row in training:
        if row.get("canonical_eligible"):
            eligible_sources[f"{row.get('task')}/{row.get('subtype')}"].add(
                str(row.get("scene_id") or row.get("item_id"))
            )
    by_task = {}
    for task, target in HUMAN_TARGETS.items():
        task_rows = [row for row in human if row.get("task") == task]
        credited = sum(bool(row.get("coverage_credit")) for row in task_rows)
        by_task[task] = {
            "target": target,
            "candidates": len(task_rows),
            "readable": sum(bool(row.get("checks", {}).get("npz_readable")) for row in task_rows),
            "recorded_success": sum(
                bool(row.get("checks", {}).get("recorded_success")) for row in task_rows
            ),
            "qualified": sum(bool(row.get("qualified")) for row in task_rows),
            "coverage_credit": credited,
            "gap": max(0, target - credited),
        }
    summary = {
        "audit_version": AUDIT_VERSION,
        "generated_at": _now(),
        "target": sum(HUMAN_TARGETS.values()),
        "candidates": len(human),
        "readable": sum(bool(row.get("checks", {}).get("npz_readable")) for row in human),
        "recorded_success": sum(
            bool(row.get("checks", {}).get("recorded_success")) for row in human
        ),
        "qualified": sum(bool(row.get("qualified")) for row in human),
        "unique_source_candidates": len({
            row.get("source_scene_id") for row in human if row.get("source_scene_id")
        }),
        "coverage_credit": sum(bool(row.get("coverage_credit")) for row in human),
        "by_task": by_task,
        "by_task_subtype": _dataset_summary(human)["by_task_subtype"],
        "human_training_success_funnel": build_human_training_success_funnel(human),
        "gaps_to_ten_per_cell": {
            cell: max(0, 10 - coverage[cell])
            for cell in target_cells
            if coverage[cell] < 10
        },
        "training_source_capacity_gaps": {
            cell: max(0, 10 - len(eligible_sources[cell]))
            for cell in target_cells
            if len(eligible_sources[cell]) < 10
        },
        "failure_reasons": _dataset_summary(human)["failure_reasons"],
    }
    summary["gap"] = max(0, summary["target"] - summary["coverage_credit"])
    return summary


def write_human_summary(output_dir: Path, summary: dict) -> None:
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    lines = [
        "# Human Recovery qualification audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Task | Candidates | Readable | Recorded success | Qualified | Unique coverage | Target | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, data in summary["by_task"].items():
        lines.append(
            f"| {task} | {data['candidates']} | {data['readable']} | "
            f"{data['recorded_success']} | {data['qualified']} | "
            f"{data['coverage_credit']} | {data['target']} | {data['gap']} |"
        )
    lines += [
        "",
        f"Total qualified: {summary['qualified']}",
        f"Unique qualified source coverage: {summary['coverage_credit']}/{summary['target']}",
        f"Gap: {summary['gap']}",
        f"Training source-capacity gap: {sum(summary['training_source_capacity_gaps'].values())}",
        "",
        "## Failure reasons",
        "",
    ]
    for reason, count in sorted(summary["failure_reasons"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{reason}`: {count}")
    funnel = summary["human_training_success_funnel"]
    lines += [
        "",
        "## Saved-start training-success AND funnel",
        "",
        "| # | Added AND condition | Overall | Newly failed | "
        + " | ".join(TASKS)
        + " |",
        "|---:|---|---:|---:|" + "---:|" * len(TASKS),
    ]
    previous = funnel["overall"]["total"]
    for index, gate in enumerate(funnel["gate_order"], 1):
        passed = funnel["overall"][gate]
        lines.append(
            f"| {index} | `{gate}` | {passed} | {previous - passed} | "
            + " | ".join(str(funnel["by_task"][task][gate]) for task in TASKS)
            + " |"
        )
        previous = passed
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(output_dir / "success_funnel.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group_type",
                "group",
                "gate_index",
                "gate",
                "total",
                "cumulative_pass",
                "newly_failed",
            ),
        )
        writer.writeheader()
        groups = [
            ("overall", "all", funnel["overall"]),
            *(("task", task, counts) for task, counts in funnel["by_task"].items()),
            *(("task_subtype", cell, counts) for cell, counts in funnel["by_task_subtype"].items()),
        ]
        for group_type, group, counts in groups:
            previous = counts["total"]
            for index, gate in enumerate(funnel["gate_order"], 1):
                passed = counts[gate]
                writer.writerow({
                    "group_type": group_type,
                    "group": group,
                    "gate_index": index,
                    "gate": gate,
                    "total": counts["total"],
                    "cumulative_pass": passed,
                    "newly_failed": previous - passed,
                })
                previous = passed


def _task_meta_scene_ids(training_root: Path, task: str) -> set[str] | None:
    path = training_root / task / "meta.json"
    try:
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return None
    return {
        str(scene.get("scene_id"))
        for scene in meta.get("scenes", [])
        if isinstance(scene, dict) and scene.get("scene_id")
    }


def _mark_duplicate_training_points(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row.get("generation_conformance") != "pass":
            continue
        key = (row.get("task"), row.get("subtype"), row.get("training_point_index"))
        groups[key].append(row)
    for duplicates in groups.values():
        if len(duplicates) < 2:
            continue
        ids = sorted(row["item_id"] for row in duplicates)
        for row in duplicates:
            row["generation_conformance"] = "fail"
            row["checks"]["unique_training_point_index"] = False
            row["failure_reasons"].append("duplicate_training_point_index")
            row["failure_details"] = {"duplicate_training_point_index": ids}
            row["failure_reasons"] = sorted(set(row["failure_reasons"]))


def build_training_filter(report_dir: Path, training_root: Path, output_dir: Path) -> dict:
    """Build reversible Training manifests from the completed qualification audit."""
    source_rows = _load_jsonl(report_dir / "training.jsonl")
    quality_rows = sorted(
        (row for row in source_rows if row.get("canonical_eligible")),
        key=lambda row: row["item_id"],
    )
    task_meta = {task: _task_meta_scene_ids(training_root, task) for task in TASKS}
    conformance = []
    for row in quality_rows:
        try:
            with open(row["source_path"], encoding="utf-8") as handle:
                meta = json.load(handle)
            if not isinstance(meta, dict):
                raise ValueError("scene metadata is not an object")
            result = evaluate_training_generation(row, meta, task_meta.get(row.get("task")))
        except Exception as exc:
            result = {
                "audit_version": AUDIT_VERSION,
                "dataset_type": "training",
                "item_id": row.get("item_id"),
                "scene_id": row.get("scene_id"),
                "source_path": row.get("source_path"),
                "task": row.get("task"),
                "subtype": row.get("subtype"),
                "quality_eligible": True,
                "generation_conformance": "fail",
                "checks": {"artifact_json_readable": False},
                "failure_reasons": ["artifact_json_unreadable"],
                "failure_details": {
                    "artifact_json_unreadable": f"{type(exc).__name__}: {exc}"
                },
                "unverifiable_reasons": [],
                "selected": False,
            }
        conformance.append(result)

    _mark_duplicate_training_points(conformance)
    subsets = {
        status: [row for row in conformance if row["generation_conformance"] == status]
        for status in ("pass", "fail", "unverifiable")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "training_quality_eligible.jsonl", quality_rows)
    _write_jsonl(output_dir / "training_generation_conformance.jsonl", conformance)
    for status, rows in subsets.items():
        _write_jsonl(output_dir / f"training_generation_{status}.jsonl", rows)

    by_task = {}
    for task in TASKS:
        task_quality = [row for row in quality_rows if row.get("task") == task]
        task_conf = [row for row in conformance if row.get("task") == task]
        by_task[task] = {
            "quality_eligible": len(task_quality),
            "generation_pass": sum(row["generation_conformance"] == "pass" for row in task_conf),
            "generation_fail": sum(row["generation_conformance"] == "fail" for row in task_conf),
            "generation_unverifiable": sum(
                row["generation_conformance"] == "unverifiable" for row in task_conf
            ),
        }
    failures = Counter()
    unverifiable = Counter()
    for row in conformance:
        failures.update(row.get("failure_reasons") or [])
        unverifiable.update(row.get("unverifiable_reasons") or [])
    summary = {
        "audit_version": AUDIT_VERSION,
        "generated_at": _now(),
        "source": len(source_rows),
        "logically_excluded": len(source_rows) - len(quality_rows),
        "quality_eligible": len(quality_rows),
        "conformance_processed": len(conformance),
        "generation_pass": len(subsets["pass"]),
        "generation_fail": len(subsets["fail"]),
        "generation_unverifiable": len(subsets["unverifiable"]),
        "selected": 0,
        "by_task": by_task,
        "failure_reasons": dict(sorted(failures.items())),
        "unverifiable_reasons": dict(sorted(unverifiable.items())),
    }
    with open(output_dir / "training_filter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    lines = [
        "# Training quality filter and generation conformance",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Task | Quality eligible | Generation pass | Fail | Unverifiable |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, data in by_task.items():
        lines.append(
            f"| {task} | {data['quality_eligible']} | {data['generation_pass']} | "
            f"{data['generation_fail']} | {data['generation_unverifiable']} |"
        )
    lines += [
        "",
        f"Source candidates: {summary['source']}",
        f"Logically excluded: {summary['logically_excluded']}",
        f"Quality eligible: {summary['quality_eligible']}",
        f"Strict generation pass: {summary['generation_pass']}",
        f"Strict generation fail: {summary['generation_fail']}",
        f"Strict generation unverifiable: {summary['generation_unverifiable']}",
    ]
    (output_dir / "training_filter_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"TRAINING FILTER COMPLETE source={summary['source']} "
        f"quality_eligible={summary['quality_eligible']} "
        f"pass={summary['generation_pass']} fail={summary['generation_fail']} "
        f"unverifiable={summary['generation_unverifiable']} output={output_dir}"
    )
    return summary


def check_training_filter(output_dir: Path, expected_source: int, expected_quality: int) -> int:
    names = (
        "training_quality_eligible",
        "training_generation_conformance",
        "training_generation_pass",
        "training_generation_fail",
        "training_generation_unverifiable",
    )
    rows = {name: _load_jsonl(output_dir / f"{name}.jsonl") for name in names}
    missing = sum(not (output_dir / f"{name}.jsonl").exists() for name in names)
    summary_path = output_dir / "training_filter_summary.json"
    if not summary_path.exists():
        summary = {}
        missing += 1
    else:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    quality = rows["training_quality_eligible"]
    conformance = rows["training_generation_conformance"]
    duplicate_records = 0
    for dataset_rows in (quality, conformance):
        ids = [row.get("item_id") for row in dataset_rows]
        duplicate_records += len(ids) - len(set(ids))
    if {row.get("item_id") for row in quality} != {
        row.get("item_id") for row in conformance
    }:
        missing += 1
    subset_total = sum(
        len(rows[f"training_generation_{status}"])
        for status in ("pass", "fail", "unverifiable")
    )
    expected_values = {
        "source": expected_source,
        "quality_eligible": expected_quality,
        "conformance_processed": expected_quality,
    }
    missing += sum(summary.get(key) != value for key, value in expected_values.items())
    missing += int(len(quality) != expected_quality)
    missing += int(len(conformance) != expected_quality)
    missing += int(subset_total != len(conformance))
    print(
        "TRAINING FILTER REPORT COMPLETE "
        f"source={summary.get('source', 0)} quality_eligible={len(quality)} "
        f"conformance_processed={len(conformance)} missing={missing} "
        f"duplicate_records={duplicate_records}"
    )
    return 0 if missing == 0 and duplicate_records == 0 else 1


def check_report(report_dir: Path, expected: dict[str, int]) -> int:
    files = {name: report_dir / f"{name}.jsonl" for name in ("training", "eval", "human")}
    rows = {name: _load_jsonl(path) for name, path in files.items()}
    missing = sum(abs(expected[name] - len(rows[name])) for name in rows)
    duplicate_records = 0
    for dataset_rows in rows.values():
        ids = [row.get("item_id") for row in dataset_rows]
        duplicate_records += len(ids) - len(set(ids))
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        missing += 1
    else:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        for name in rows:
            if summary.get("datasets", {}).get(name, {}).get("processed") != len(rows[name]):
                missing += 1
    print(
        "AUDIT REPORT COMPLETE "
        f"training={len(rows['training'])} eval={len(rows['eval'])} "
        f"human={len(rows['human'])} missing={missing} "
        f"duplicate_records={duplicate_records}"
    )
    return 0 if (
        all(len(rows[name]) == expected[name] for name in rows)
        and missing == 0
        and duplicate_records == 0
    ) else 1


def check_human_report(report_dir: Path, expected_human: int) -> int:
    human_path = report_dir / "human.jsonl"
    rows = _load_jsonl(human_path)
    ids = [row.get("item_id") for row in rows]
    duplicate_records = len(ids) - len(set(ids))
    missing = abs(expected_human - len(rows)) + int(not human_path.exists())
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        summary = {}
        missing += 1
    else:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    coverage_ids = [
        row.get("source_scene_id") for row in rows if row.get("coverage_credit")
    ]
    missing += int(summary.get("candidates") != len(rows))
    missing += int(summary.get("target") != sum(HUMAN_TARGETS.values()))
    missing += int(summary.get("coverage_credit") != len(coverage_ids))
    missing += int(len(coverage_ids) != len(set(coverage_ids)))
    missing += int(any(row.get("coverage_credit") and not row.get("qualified") for row in rows))
    print(
        f"HUMAN RECOVERY REPORT COMPLETE candidates={len(rows)} "
        f"missing={missing} duplicate_records={duplicate_records}"
    )
    return 0 if len(rows) == expected_human and missing == 0 and duplicate_records == 0 else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=PROJECT_ROOT / "error_benchmark/outputs/v5_training")
    parser.add_argument("--eval-root", type=Path, default=PROJECT_ROOT / "error_benchmark/outputs/v5")
    parser.add_argument("--human-root", type=Path, default=PROJECT_ROOT / "error_benchmark/outputs/recovery/demos")
    parser.add_argument("--historical-eval", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qualification/canonical_candidates_20260713")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--check-report", type=Path, default=None)
    parser.add_argument("--check-human-report", type=Path, default=None)
    parser.add_argument("--human-only", action="store_true")
    parser.add_argument("--training-report", type=Path, default=None)
    parser.add_argument("--filter-training-report", type=Path, default=None)
    parser.add_argument("--training-filter-output", type=Path, default=None)
    parser.add_argument("--check-training-filter", type=Path, default=None)
    parser.add_argument("--expected-training", type=int, default=5499)
    parser.add_argument("--expected-eval", type=int, default=135454)
    parser.add_argument("--expected-human", type=int, default=973)
    parser.add_argument("--expected-source", type=int, default=5499)
    parser.add_argument("--expected-quality-eligible", type=int, default=1244)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_training_filter:
        return check_training_filter(
            args.check_training_filter,
            args.expected_source,
            args.expected_quality_eligible,
        )
    if args.check_report:
        return check_report(
            args.check_report,
            {
                "training": args.expected_training,
                "eval": args.expected_eval,
                "human": args.expected_human,
            },
        )
    if args.check_human_report:
        return check_human_report(args.check_human_report, args.expected_human)

    if args.filter_training_report:
        output_dir = (
            args.training_filter_output
            or args.filter_training_report / "training_filter"
        )
        build_training_filter(args.filter_training_report, args.training_root, output_dir)
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.human_only:
        if args.training_report is None:
            raise SystemExit("--training-report is required with --human-only")
        training = _load_jsonl(args.training_report, required_version=None)
        if not training:
            raise SystemExit(f"No Training qualification rows found in {args.training_report}")
        human_candidates = discover_human_candidates(args.human_root, args.tasks)
        human = _audit_candidates(
            human_candidates,
            _audit_human_candidate,
            args.output_dir / "human.jsonl",
            args.workers,
            training_lookup=build_training_lookup(training),
            resume=not args.no_resume,
        )
        assign_coverage_credit(human)
        _write_jsonl(args.output_dir / "human.jsonl", human)
        write_human_summary(args.output_dir, build_human_summary(training, human))
        print(f"HUMAN RECOVERY AUDIT COMPLETE candidates={len(human)} output={args.output_dir}")
        return 0

    historical_ids = load_historical_scene_ids(args.historical_eval)
    training_candidates = discover_scene_candidates(args.training_root, "training", args.tasks)
    eval_candidates = discover_scene_candidates(args.eval_root, "eval", args.tasks)
    for candidate in eval_candidates:
        candidate["historical_compatible"] = Path(candidate["json_path"]).stem in historical_ids
    human_candidates = discover_human_candidates(args.human_root, args.tasks)

    resume = not args.no_resume
    training = _audit_candidates(
        training_candidates,
        _audit_scene_candidate,
        args.output_dir / "training.jsonl",
        args.workers,
        resume=resume,
    )
    evaluation = _audit_candidates(
        eval_candidates,
        _audit_scene_candidate,
        args.output_dir / "eval.jsonl",
        args.workers,
        resume=resume,
    )
    finalize_scene_records(training, evaluation)
    _write_jsonl(args.output_dir / "training.jsonl", training)
    _write_jsonl(args.output_dir / "eval.jsonl", evaluation)

    training_lookup = build_training_lookup(training)
    human = _audit_candidates(
        human_candidates,
        _audit_human_candidate,
        args.output_dir / "human.jsonl",
        args.workers,
        training_lookup=training_lookup,
        resume=resume,
    )
    assign_coverage_credit(human)
    _write_jsonl(args.output_dir / "human.jsonl", human)

    summary = build_summary(training, evaluation, human)
    write_summary(args.output_dir, summary)
    print(
        f"AUDIT COMPLETE training={len(training)} eval={len(evaluation)} "
        f"human={len(human)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
