#!/usr/bin/env python
"""Helpers for validating MimicGen source demos collected from human teleop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import h5py
import numpy as np


DEFAULT_ENV_INTERFACE_TYPE = "robosuite"
TASK_ENV_INTERFACE_NAMES = {
    "pick_place": "MG_PickPlace",
    "coffee": "MG_Coffee",
    "stack": "MG_Stack",
    "stack_three": "MG_StackThree",
    "threading": "MG_Threading",
    "three_piece_assembly": "MG_ThreePieceAssembly",
}

STACK_ENV_INTERFACE_NAME = TASK_ENV_INTERFACE_NAMES["stack"]
STACK_ENV_INTERFACE_TYPE = DEFAULT_ENV_INTERFACE_TYPE

DEFAULT_STACK_SOURCE_VALIDATION = {
    "settle_linvel_threshold": 0.01,
    "settle_angvel_threshold": 0.05,
    "settle_hold_frames": 10,
    "max_settle_steps": 40,
    "min_pre_grasp_frames": 5,
    "min_post_grasp_frames": 21,
    "require_final_replay_success": True,
    "reject_if_success_at_start": True,
}


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy / bytes scalars to JSON-friendly Python objects."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_env_interface_name_for_task(task_name: str) -> str:
    """Return the MimicGen environment interface name for a task."""

    try:
        return TASK_ENV_INTERFACE_NAMES[task_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MimicGen task '{task_name}'. Known tasks: "
            f"{sorted(TASK_ENV_INTERFACE_NAMES)}"
        ) from exc


def ensure_env_interface_attrs(
    dataset_path: Path | str,
    *,
    task_name: Optional[str] = None,
    env_interface_name: Optional[str] = None,
    env_interface_type: str = DEFAULT_ENV_INTERFACE_TYPE,
) -> Dict[str, Any]:
    """Ensure each demo's ``datagen_info`` stores MimicGen interface metadata."""

    if env_interface_name is None:
        if task_name is None:
            raise ValueError("Specify task_name or env_interface_name")
        env_interface_name = get_env_interface_name_for_task(task_name)

    dataset_path = Path(dataset_path)
    updated_count = 0
    missing_datagen_info_count = 0
    demo_count = 0

    with h5py.File(dataset_path, "r+") as handle:
        if "data" not in handle:
            raise ValueError(f"Missing 'data' group in {dataset_path}")
        data_group = handle["data"]
        for demo_key in sorted(data_group.keys()):
            if not demo_key.startswith("demo_"):
                continue
            demo_count += 1
            demo_group = data_group[demo_key]
            if "datagen_info" not in demo_group:
                missing_datagen_info_count += 1
                continue

            datagen_info = demo_group["datagen_info"]
            changed = False

            current_name = (
                _to_python_scalar(datagen_info.attrs["env_interface_name"])
                if "env_interface_name" in datagen_info.attrs
                else None
            )
            current_type = (
                _to_python_scalar(datagen_info.attrs["env_interface_type"])
                if "env_interface_type" in datagen_info.attrs
                else None
            )

            if current_name != env_interface_name:
                datagen_info.attrs["env_interface_name"] = env_interface_name
                changed = True
            if current_type != env_interface_type:
                datagen_info.attrs["env_interface_type"] = env_interface_type
                changed = True

            if changed:
                updated_count += 1

    return {
        "dataset_path": str(dataset_path),
        "demo_count": demo_count,
        "updated_count": updated_count,
        "missing_datagen_info_count": missing_datagen_info_count,
        "env_interface_name": env_interface_name,
        "env_interface_type": env_interface_type,
    }


def get_stack_source_validation_config(task_config: Optional[dict]) -> Dict[str, Any]:
    """Return stack source-validation config with defaults applied."""

    merged = dict(DEFAULT_STACK_SOURCE_VALIDATION)
    if task_config:
        merged.update(task_config.get("mimicgen_source_validation", {}))
    return merged


def get_task_object_names(task_config: Optional[dict]) -> list[str]:
    """Return task object names in config order."""

    if not task_config:
        return []
    return [
        str(obj["name"])
        for obj in task_config.get("objects", [])
        if isinstance(obj, dict) and obj.get("name")
    ]


def collect_object_motion_metrics(env_wrapper, object_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Read per-object linear and angular velocity magnitudes."""

    metrics: Dict[str, Dict[str, float]] = {}
    for name in object_names:
        try:
            linvel, angvel = env_wrapper.get_object_velocity(name)
        except Exception:
            continue
        metrics[name] = {
            "linvel": float(np.linalg.norm(linvel)),
            "angvel": float(np.linalg.norm(angvel)),
        }
    return metrics


def flatten_motion_metrics(
    metrics: Dict[str, Dict[str, float]],
    prefix: str = "initial",
) -> Dict[str, float]:
    """Flatten motion metrics into saveable metadata keys."""

    flat: Dict[str, float] = {}
    for obj_name, obj_metrics in metrics.items():
        flat[f"{prefix}_{obj_name}_linvel"] = float(obj_metrics.get("linvel", 0.0))
        flat[f"{prefix}_{obj_name}_angvel"] = float(obj_metrics.get("angvel", 0.0))
    return flat


def build_open_gripper_action(env_wrapper) -> np.ndarray:
    """Build a neutral action that keeps the gripper open while settling."""

    action = env_wrapper.get_neutral_action().copy()
    if action.size > 0:
        action[-1] = getattr(env_wrapper, "_gripper_open_action", -1.0)
    return action


def _motion_is_stable(
    motion_metrics: Dict[str, Dict[str, float]],
    linvel_threshold: float,
    angvel_threshold: float,
) -> bool:
    """Return True when all tracked objects satisfy stability thresholds."""

    if not motion_metrics:
        return True
    return all(
        obj_metrics["linvel"] <= linvel_threshold
        and obj_metrics["angvel"] <= angvel_threshold
        for obj_metrics in motion_metrics.values()
    )


@dataclass(frozen=True)
class SceneSettleResult:
    """Outcome of the pre-teleop settle gate."""

    accepted: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "metadata": {k: _to_python_scalar(v) for k, v in self.metadata.items()},
        }


def settle_scene_for_source_collection(
    env_wrapper,
    task_config: dict,
    logger=None,
    render_fn: Optional[Callable[[], None]] = None,
) -> SceneSettleResult:
    """Wait for a stack scene to become physically stable before teleop starts."""

    object_names = get_task_object_names(task_config)
    config = get_stack_source_validation_config(task_config)
    linvel_threshold = float(config["settle_linvel_threshold"])
    angvel_threshold = float(config["settle_angvel_threshold"])
    hold_frames = int(config["settle_hold_frames"])
    max_settle_steps = int(config["max_settle_steps"])

    initial_metrics = collect_object_motion_metrics(env_wrapper, object_names)
    metadata: Dict[str, Any] = {
        "validation_status": "pending",
        "validation_reason": "settling",
        "settle_steps_used": 0,
        "initial_success_after_load": bool(env_wrapper.check_success()),
    }
    metadata.update(flatten_motion_metrics(initial_metrics, prefix="initial"))

    stable_frames = 0
    for step in range(max_settle_steps + 1):
        current_metrics = collect_object_motion_metrics(env_wrapper, object_names)
        if _motion_is_stable(current_metrics, linvel_threshold, angvel_threshold):
            stable_frames += 1
        else:
            stable_frames = 0

        if stable_frames >= hold_frames:
            metadata["validation_status"] = "accepted"
            metadata["validation_reason"] = "accepted"
            metadata["settle_steps_used"] = step
            metadata.update(flatten_motion_metrics(current_metrics, prefix="post_settle"))
            return SceneSettleResult(
                accepted=True,
                reason="accepted",
                metadata=metadata,
            )

        if step == max_settle_steps:
            break

        env_wrapper.step(build_open_gripper_action(env_wrapper))
        if render_fn is not None:
            render_fn()

    metadata["validation_status"] = "rejected"
    metadata["validation_reason"] = "unstable_start"
    metadata["settle_steps_used"] = max_settle_steps
    metadata.update(
        flatten_motion_metrics(
            collect_object_motion_metrics(env_wrapper, object_names),
            prefix="post_settle",
        )
    )
    if logger is not None:
        logger.warning(
            "Scene did not settle within %s steps (lin<=%.4f, ang<=%.4f)",
            max_settle_steps,
            linvel_threshold,
            angvel_threshold,
        )
    return SceneSettleResult(
        accepted=False,
        reason="unstable_start",
        metadata=metadata,
    )


def find_first_binary_transition(signal: np.ndarray, threshold: float = 0.5) -> Optional[int]:
    """Return the first 0->1 transition index or None if no transition exists."""

    values = np.asarray(signal)
    if values.size < 2:
        return None
    active = values >= threshold
    transitions = np.flatnonzero(np.logical_not(active[:-1]) & active[1:])
    if transitions.size == 0:
        return None
    return int(transitions[0] + 1)


@dataclass(frozen=True)
class SourceDemoValidationResult:
    """Validation result for one prepared MimicGen source demo."""

    accepted: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "details": {k: _to_python_scalar(v) for k, v in self.details.items()},
        }


def _base_validation_details(demo_key: str, demo_group: h5py.Group) -> Dict[str, Any]:
    details = {
        "demo_key": demo_key,
        "num_samples": int(
            demo_group.attrs.get(
                "num_samples",
                demo_group["actions"].shape[0] if "actions" in demo_group else 0,
            )
        ),
    }
    for attr_name in ("scene_id", "demo_origin", "validation_status", "validation_reason"):
        if attr_name in demo_group.attrs:
            details[attr_name] = _to_python_scalar(demo_group.attrs[attr_name])
    return details


def validate_stack_prepared_demo(
    demo_key: str,
    demo_group: h5py.Group,
    task_config: dict,
    replay_success_checker: Optional[Callable[[str], bool]] = None,
) -> SourceDemoValidationResult:
    """Validate one prepared stack demo before merging into source data."""

    config = get_stack_source_validation_config(task_config)
    details = _base_validation_details(demo_key, demo_group)
    raw_reason = str(demo_group.attrs.get("validation_reason", ""))
    raw_status = str(demo_group.attrs.get("validation_status", ""))

    if raw_reason == "unstable_start":
        return SourceDemoValidationResult(False, "unstable_start", details)
    if raw_reason == "already_success_after_settle" and bool(config["reject_if_success_at_start"]):
        return SourceDemoValidationResult(False, "already_success_after_settle", details)
    if raw_status and raw_status not in {"pending", "accepted"}:
        return SourceDemoValidationResult(False, raw_reason or raw_status, details)

    if "datagen_info" not in demo_group:
        return SourceDemoValidationResult(False, "missing_datagen_info", details)

    datagen_info = demo_group["datagen_info"]
    if "subtask_term_signals" not in datagen_info:
        return SourceDemoValidationResult(False, "missing_grasp_signal", details)
    if "grasp" not in datagen_info["subtask_term_signals"]:
        return SourceDemoValidationResult(False, "missing_grasp_signal", details)

    grasp_signal = np.asarray(datagen_info["subtask_term_signals"]["grasp"])
    first_grasp_frame = find_first_binary_transition(grasp_signal)
    details["grasp_signal_max"] = float(grasp_signal.max()) if grasp_signal.size else 0.0
    details["first_grasp_frame"] = first_grasp_frame

    if first_grasp_frame is None:
        return SourceDemoValidationResult(False, "no_grasp_transition", details)

    min_pre = int(config["min_pre_grasp_frames"])
    if first_grasp_frame < min_pre:
        return SourceDemoValidationResult(False, "grasp_transition_too_early", details)

    num_samples = int(details["num_samples"])
    min_post = int(config["min_post_grasp_frames"])
    if first_grasp_frame > num_samples - min_post:
        return SourceDemoValidationResult(False, "grasp_transition_too_late", details)

    if bool(config["require_final_replay_success"]):
        if replay_success_checker is None:
            raise ValueError("replay_success_checker is required for stack source validation")
        replay_ok = bool(replay_success_checker(demo_key))
        details["final_replay_success"] = replay_ok
        if not replay_ok:
            return SourceDemoValidationResult(False, "final_replay_not_success", details)

    return SourceDemoValidationResult(True, "accepted", details)


def build_demo_replay_success_checker(
    dataset_path: str,
    task_config: dict,
) -> Callable[[str], bool]:
    """Build a replay checker that verifies final success by replaying actions."""

    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.scripts.utils.script_utils import create_env

    env = create_env(task_config, dataset_path, enable_camera=False, has_renderer=False)
    env_wrapper = EnvWrapper(env, task_config)
    dataset = Path(dataset_path)

    def _check(demo_key: str) -> bool:
        with h5py.File(dataset, "r") as h5_file:
            actions = h5_file[f"data/{demo_key}/actions"][()]
            states = h5_file[f"data/{demo_key}/states"][()]

        if actions.size == 0 or states.size == 0:
            return False

        env_wrapper.set_sim_state_flat(states[0])
        env_wrapper.forward()
        for action in actions:
            env_wrapper.step(action)
        return bool(env_wrapper.check_success())

    return _check
