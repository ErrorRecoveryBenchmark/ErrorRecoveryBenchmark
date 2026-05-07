#!/usr/bin/env python
"""Helpers for faithful MimicGen-style recovery augmentation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from error_benchmark.framework.utils.pose_transforms import (
    _quat2mat,
    axisangle2quat,
    make_pose,
)


@dataclass(frozen=True)
class RecoverySubtaskSpec:
    """Execution settings for one recovery subtask."""

    object_ref_strategy: str
    selection_strategy: str
    selection_strategy_kwargs: Optional[Dict]
    num_interpolation_steps: int
    num_fixed_steps: int
    apply_noise_during_interpolation: bool
    action_noise: float


DEFAULT_RECOVERY_SUBTASK_SPECS = {
    # MimicGen-style 2-subtask structure (default for all tasks)
    "pre_grasp": {"object_ref_strategy": "target_object"},
    "post_grasp": {"object_ref_strategy": "placement_ref"},
    # Legacy fine-grained subtask labels (kept for backward compatibility)
    "retract": {"object_ref_strategy": "none"},
    "release": {"object_ref_strategy": "none"},
    "navigate_to_object": {"object_ref_strategy": "target_object"},
    "re_orient": {"object_ref_strategy": "target_object"},
    "re_grasp": {"object_ref_strategy": "target_object"},
    "re_lift": {"object_ref_strategy": "target_object"},
    "re_transport": {"object_ref_strategy": "placement_ref"},
    "re_place": {"object_ref_strategy": "placement_ref"},
    "correct_position": {"object_ref_strategy": "none"},
    "resume_task": {"object_ref_strategy": "none"},
    "re_acquire": {"object_ref_strategy": "target_object"},
    "re_deliver": {"object_ref_strategy": "placement_ref"},
}

_TARGET_PHASE_GROUPS = {"approach"}
_TARGET_TASK_PHASES = {"pre_reach", "reach", "pre_grasp", "grasp", "lift"}
_PLACEMENT_PHASE_GROUPS = {"transfer"}
_PLACEMENT_TASK_PHASES = {"transport", "place", "done"}


def build_recovery_subtask_specs(spec_config: Optional[dict]) -> Dict[str, RecoverySubtaskSpec]:
    """Build recovery subtask specs from config with sane MimicGen-style defaults."""

    spec_config = spec_config or {}
    defaults = spec_config.get("defaults", {})
    overrides = spec_config.get("subtasks", {})

    with_object_strategy = defaults.get("selection_strategy_with_object", "nearest_neighbor_object")
    with_object_kwargs = defaults.get("selection_strategy_kwargs_with_object", {"nn_k": 3})
    without_object_strategy = defaults.get("selection_strategy_without_object", "random")
    without_object_kwargs = defaults.get("selection_strategy_kwargs_without_object")
    num_interpolation_steps = int(defaults.get("num_interpolation_steps", 5))
    num_fixed_steps = int(defaults.get("num_fixed_steps", 0))
    apply_noise_during_interpolation = bool(
        defaults.get("apply_noise_during_interpolation", False)
    )
    action_noise = float(defaults.get("action_noise", 0.0))

    specs: Dict[str, RecoverySubtaskSpec] = {}
    known_labels = list(DEFAULT_RECOVERY_SUBTASK_SPECS.keys())
    for label in overrides:
        if label not in DEFAULT_RECOVERY_SUBTASK_SPECS:
            known_labels.append(label)

    for label in known_labels:
        base = DEFAULT_RECOVERY_SUBTASK_SPECS.get(label, {})
        merged = deepcopy(base)
        merged.update(overrides.get(label, {}))
        strategy = merged["object_ref_strategy"]
        has_object_ref = strategy != "none"
        specs[label] = RecoverySubtaskSpec(
            object_ref_strategy=strategy,
            selection_strategy=merged.get(
                "selection_strategy",
                with_object_strategy if has_object_ref else without_object_strategy,
            ),
            selection_strategy_kwargs=merged.get(
                "selection_strategy_kwargs",
                deepcopy(with_object_kwargs if has_object_ref else without_object_kwargs),
            ),
            num_interpolation_steps=int(
                merged.get("num_interpolation_steps", num_interpolation_steps)
            ),
            num_fixed_steps=int(merged.get("num_fixed_steps", num_fixed_steps)),
            apply_noise_during_interpolation=bool(
                merged.get(
                    "apply_noise_during_interpolation",
                    apply_noise_during_interpolation,
                )
            ),
            action_noise=float(merged.get("action_noise", action_noise)),
        )
    return specs


def make_pose_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 pose from position and a wxyz quaternion."""

    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = _quat2mat(xyzw)
    return make_pose(np.asarray(pos), rot)


def infer_target_pose_from_action(
    current_pose: np.ndarray,
    action: np.ndarray,
    max_dpos: float,
    max_drot: float,
    relative: bool = True,
) -> np.ndarray:
    """Inverse of target-pose control for OSC actions."""

    target_action = np.asarray(action[:6], dtype=np.float64)
    if not relative:
        target_pos = target_action[:3]
        target_rot = _quat2mat(axisangle2quat(target_action[3:6]))
        return make_pose(target_pos, target_rot)

    curr_pos = current_pose[:3, 3]
    curr_rot = current_pose[:3, :3]
    delta_pos = target_action[:3] * max_dpos
    delta_rot = target_action[3:6] * max_drot
    delta_angle = np.linalg.norm(delta_rot)
    if delta_angle > 1e-8:
        delta_axis = delta_rot / delta_angle
        delta_quat = axisangle2quat(delta_axis, delta_angle)
    else:
        delta_quat = axisangle2quat(np.array([1.0, 0.0, 0.0]), 0.0)
    delta_rot_mat = _quat2mat(delta_quat)
    target_pos = curr_pos + delta_pos
    target_rot = delta_rot_mat.dot(curr_rot)
    return make_pose(target_pos, target_rot)


def resolve_placement_ref(
    task_name: str,
    task_config: dict,
    target_object: str,
    scene_labels: Optional[dict] = None,
) -> Optional[str]:
    """Resolve the placement reference object for object-centric warping."""

    scene_labels = scene_labels or {}
    if scene_labels.get("placement_ref"):
        return scene_labels["placement_ref"]

    task_overrides = (
        task_config.get("recovery_augmentation", {}).get("placement_refs", {})
    )
    if target_object in task_overrides:
        return task_overrides[target_object]

    if task_name == "stack":
        return "cubeB" if target_object == "cubeA" else None
    if task_name == "stack_three":
        mapping = {"cubeA": "cubeB", "cubeC": "cubeA"}
        return mapping.get(target_object)
    if task_name == "coffee":
        return "coffee_machine" if target_object == "coffee_pod" else None
    if task_name == "threading":
        return "tripod" if target_object == "needle" else None
    if task_name == "three_piece_assembly":
        mapping = {"piece_1": "base", "piece_2": "piece_1"}
        return mapping.get(target_object)
    if task_name == "pick_place":
        return None
    return None


def resolve_subtask_object_ref(
    spec: RecoverySubtaskSpec,
    task_name: str,
    task_config: dict,
    target_object: str,
    scene_labels: Optional[dict] = None,
) -> Optional[str]:
    """Resolve the object reference for a recovery subtask."""

    scene_labels = scene_labels or {}
    strategy = spec.object_ref_strategy

    if strategy == "none":
        return None
    if strategy == "target_object":
        return target_object
    if strategy == "placement_ref":
        return resolve_placement_ref(task_name, task_config, target_object, scene_labels)
    if strategy == "phase_ref":
        phase_group = scene_labels.get("phase_group", "")
        task_phase = scene_labels.get("task_phase", "")
        if phase_group in _PLACEMENT_PHASE_GROUPS or task_phase in _PLACEMENT_TASK_PHASES:
            return resolve_placement_ref(task_name, task_config, target_object, scene_labels)
        if phase_group in _TARGET_PHASE_GROUPS or task_phase in _TARGET_TASK_PHASES:
            return target_object
        return target_object

    raise ValueError(f"Unknown recovery object_ref_strategy '{strategy}'")
