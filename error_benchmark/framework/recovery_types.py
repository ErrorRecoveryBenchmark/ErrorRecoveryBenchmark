#!/usr/bin/env python
"""
Recovery Data Types - Data structures for recovery demo collection and augmentation.

New data structures for the recovery pipeline:
    RecoveryDemo        -> A single human teleoperation recovery demonstration
    RecoverySubtask     -> A segmented subtask within a recovery demo
    AugmentedRecovery   -> An augmented recovery trajectory
    RecoveryCollectionStatus -> Tracks collection progress per (task, subtype)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import json


# ─── Recovery Subtask Labels ───

RECOVERY_SUBTASKS = [
    "pre_grasp",            # MimicGen-style: approach + grasp (target_object warp)
    "post_grasp",           # MimicGen-style: lift + transport + place (placement_ref warp)
    "retract",              # Back away from error state (no warp)
    "release",              # Open gripper to release wrong/misaligned object (no warp)
    "navigate_to_object",   # Move EEF toward target object
    "re_orient",            # Align gripper pose with object
    "re_grasp",             # Approach + close gripper on target object
    "re_lift",              # Lift recovered object after grasp
    "re_transport",         # Move recovered object toward placement reference
    "re_place",             # Place recovered object at target
    "correct_position",     # In-place position correction (RBG_E)
    "resume_task",          # Resume normal task execution (RBG_E)
    "re_acquire",           # Backward-compatible coarse alias
    "re_deliver",           # Backward-compatible coarse alias
]

# Recovery Behavior Group definitions (fine-grained recovery subtasks)
RBG_SUBTASK_SEQUENCES = {
    "RBG_A": ["retract", "re_orient", "re_grasp", "re_lift", "re_transport", "re_place"],
    "RBG_B": ["navigate_to_object", "re_grasp", "re_lift", "re_transport", "re_place"],
    "RBG_C": ["retract", "navigate_to_object"],
    "RBG_D": ["release", "navigate_to_object", "re_grasp", "re_lift", "re_transport", "re_place"],
    "RBG_E": ["correct_position", "resume_task"],
}

# Map subtypes to their RBG
SUBTYPE_TO_RBG = {
    "grasp_misalignment_D0": "RBG_A",
    "grasp_misalignment_D1": "RBG_A",
    "grasp_wrong_pose_D0": "RBG_A",
    "grasp_wrong_pose_D1": "RBG_A",
    "drop_in_transit_D0": "RBG_B",
    "drop_in_transit_D1": "RBG_B",
    "drop_at_wrong_place_D0": "RBG_B",
    "drop_at_wrong_place_D1": "RBG_B",
    "collision_holding_D0": "RBG_C",
    "collision_holding_D1": "RBG_C",
    "collision_empty_D0": "RBG_C",
    "collision_empty_D1": "RBG_C",
    "collision_eef_object_D0": "RBG_C",
    "collision_eef_object_D1": "RBG_C",
    "collision_self_D0": "RBG_C",
    "collision_self_D1": "RBG_C",
    "wrong_object_D0": "RBG_D",
    "wrong_object_D1": "RBG_D",
    "trajectory_regression_D0": "RBG_E",
    "trajectory_regression_D1": "RBG_E",
    "stuck_no_progress_D0": "RBG_E",
    "stuck_no_progress_D1": "RBG_E",
    "position_error_D0": "RBG_E",
    "position_error_D1": "RBG_E",
}

# MCM motion instruction templates per RBG
RBG_MOTION_TEMPLATES = {
    "RBG_A": "release object, adjust gripper position, then re-grasp",
    "RBG_B": "move to dropped object, pick it up, then continue to target",
    "RBG_C": "pull back from collision, then re-approach target",
    "RBG_D": "release wrong object, move to correct object, then pick it up",
    "RBG_E": "correct the position error and resume the task",
}


@dataclass
class RecoverySubtask:
    """A segmented subtask within a recovery demonstration."""
    label: str = ""                      # Subtask label (from RECOVERY_SUBTASKS)
    start_step: int = 0                  # Start step index (inclusive)
    end_step: int = 0                    # End step index (exclusive)
    duration: int = 0                    # Number of steps
    eef_displacement: float = 0.0        # Total EEF displacement during subtask
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'start_step': self.start_step,
            'end_step': self.end_step,
            'duration': self.duration,
            'eef_displacement': self.eef_displacement,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RecoverySubtask':
        return cls(**d)


def _serialize_subtask_entry(subtask) -> dict:
    """Best-effort subtask serialization for manifests and worker payloads."""
    if isinstance(subtask, RecoverySubtask):
        return subtask.to_dict()
    if isinstance(subtask, dict):
        return RecoverySubtask.from_dict(subtask).to_dict()
    if hasattr(subtask, "to_dict") and callable(subtask.to_dict):
        data = subtask.to_dict()
        if isinstance(data, dict):
            return RecoverySubtask.from_dict(data).to_dict()

    start_step = int(getattr(subtask, "start_step", 0) or 0)
    end_step = int(getattr(subtask, "end_step", 0) or 0)
    duration = int(getattr(subtask, "duration", max(end_step - start_step, 0)) or 0)
    eef_displacement = float(getattr(subtask, "eef_displacement", 0.0) or 0.0)
    metadata = getattr(subtask, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "label": str(getattr(subtask, "label", type(subtask).__name__)),
        "start_step": start_step,
        "end_step": end_step,
        "duration": duration,
        "eef_displacement": eef_displacement,
        "metadata": metadata,
    }


@dataclass
class RecoveryDemo:
    """
    A single human teleoperation recovery demonstration.

    Collected by loading an error scene (NPZ), then recording human teleop
    actions until task success or timeout.
    """
    demo_id: str = ""
    task_name: str = ""
    error_name: str = ""                 # Error skill name
    degree: str = ""                     # "D0" | "D1"
    subtype_id: str = ""                 # "grasp_misalignment_D0"
    rbg: str = ""                        # "RBG_A" through "RBG_E"
    scene_id: str = ""                   # Source error scene ID
    scene_npz_path: str = ""             # Path to error scene NPZ
    success: bool = False                # Did recovery succeed?
    num_steps: int = 0
    actions: Optional[np.ndarray] = field(default=None, repr=False)      # (T, action_dim)
    states: Optional[np.ndarray] = field(default=None, repr=False)       # (T+1, state_dim)
    camera_images: Optional[List[np.ndarray]] = field(default=None, repr=False)  # List of (H,W,3)
    eef_positions: Optional[np.ndarray] = field(default=None, repr=False)  # (T+1, 3)
    eef_orientations: Optional[np.ndarray] = field(default=None, repr=False)  # (T+1, 4) wxyz quaternion
    target_poses: Optional[np.ndarray] = field(default=None, repr=False)  # (T, 4, 4) controller target poses
    gripper_states: Optional[np.ndarray] = field(default=None, repr=False) # (T+1,)
    object_positions: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)  # name -> (T+1, 3)
    object_orientations: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)  # name -> (T+1, 4) wxyz
    subtasks: List[RecoverySubtask] = field(default_factory=list)
    created: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.subtype_id and self.error_name and self.degree:
            self.subtype_id = f"{self.error_name}_{self.degree}"
        if not self.rbg and self.subtype_id:
            self.rbg = SUBTYPE_TO_RBG.get(self.subtype_id, "")

    def to_dict(self) -> dict:
        """Serialize metadata (excluding large arrays)."""
        return {
            'demo_id': self.demo_id,
            'task_name': self.task_name,
            'error_name': self.error_name,
            'degree': self.degree,
            'subtype_id': self.subtype_id,
            'rbg': self.rbg,
            'scene_id': self.scene_id,
            'scene_npz_path': self.scene_npz_path,
            'success': self.success,
            'num_steps': self.num_steps,
            'subtasks': [_serialize_subtask_entry(s) for s in self.subtasks],
            'created': self.created,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RecoveryDemo':
        subtasks = [
            RecoverySubtask.from_dict(_serialize_subtask_entry(s))
            for s in d.get('subtasks', [])
        ]
        return cls(
            demo_id=d.get('demo_id', ''),
            task_name=d.get('task_name', ''),
            error_name=d.get('error_name', ''),
            degree=d.get('degree', ''),
            subtype_id=d.get('subtype_id', ''),
            rbg=d.get('rbg', ''),
            scene_id=d.get('scene_id', ''),
            scene_npz_path=d.get('scene_npz_path', ''),
            success=d.get('success', False),
            num_steps=d.get('num_steps', 0),
            subtasks=subtasks,
            created=d.get('created', ''),
            metadata=d.get('metadata', {}),
        )


@dataclass
class AugmentedRecovery:
    """
    An augmented recovery trajectory generated from a source RecoveryDemo.

    Produced by MimicGen-style scene augmentation, cross-degree, or cross-subtype.
    """
    augmented_id: str = ""
    source_demo_id: str = ""             # Original human demo
    augmentation_type: str = ""          # "scene" | "cross_degree" | "cross_subtype"
    task_name: str = ""
    error_name: str = ""
    degree: str = ""
    subtype_id: str = ""
    rbg: str = ""
    target_scene_id: str = ""            # Error scene used as new initial state
    success: bool = False
    num_steps: int = 0
    actions: Optional[np.ndarray] = field(default=None, repr=False)
    states: Optional[np.ndarray] = field(default=None, repr=False)
    camera_images: Optional[List[np.ndarray]] = field(default=None, repr=False)
    eef_positions: Optional[np.ndarray] = field(default=None, repr=False)      # (T+1, 3)
    target_poses: Optional[np.ndarray] = field(default=None, repr=False)       # (T, 4, 4)
    gripper_states: Optional[np.ndarray] = field(default=None, repr=False)     # (T+1,)
    object_positions: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    subtasks: List[RecoverySubtask] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.subtype_id and self.error_name and self.degree:
            self.subtype_id = f"{self.error_name}_{self.degree}"
        if not self.rbg and self.subtype_id:
            self.rbg = SUBTYPE_TO_RBG.get(self.subtype_id, "")

    def to_dict(self) -> dict:
        return {
            'augmented_id': self.augmented_id,
            'source_demo_id': self.source_demo_id,
            'augmentation_type': self.augmentation_type,
            'task_name': self.task_name,
            'error_name': self.error_name,
            'degree': self.degree,
            'subtype_id': self.subtype_id,
            'rbg': self.rbg,
            'target_scene_id': self.target_scene_id,
            'success': self.success,
            'num_steps': self.num_steps,
            'subtasks': [_serialize_subtask_entry(s) for s in self.subtasks],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AugmentedRecovery':
        subtasks = [
            RecoverySubtask.from_dict(_serialize_subtask_entry(s))
            for s in d.get('subtasks', [])
        ]
        return cls(
            augmented_id=d.get('augmented_id', ''),
            source_demo_id=d.get('source_demo_id', ''),
            augmentation_type=d.get('augmentation_type', ''),
            task_name=d.get('task_name', ''),
            error_name=d.get('error_name', ''),
            degree=d.get('degree', ''),
            subtype_id=d.get('subtype_id', ''),
            rbg=d.get('rbg', ''),
            target_scene_id=d.get('target_scene_id', ''),
            success=d.get('success', False),
            num_steps=d.get('num_steps', 0),
            subtasks=subtasks,
            metadata=d.get('metadata', {}),
        )


@dataclass
class RecoveryCollectionStatus:
    """
    Tracks collection progress per (task, subtype).
    """
    task_name: str = ""
    collection_mode: str = "quota"  # "quota" | "per_scene"
    target_demos: Dict[str, int] = field(default_factory=dict)   # subtype_id -> target count
    collected_demos: Dict[str, int] = field(default_factory=dict) # subtype_id -> collected count
    augmented_demos: Dict[str, int] = field(default_factory=dict) # subtype_id -> augmented count
    target_scene_ids: Dict[str, List[str]] = field(default_factory=dict)     # subtype_id -> target scene_ids
    collected_scene_ids: Dict[str, List[str]] = field(default_factory=dict)  # subtype_id -> covered scene_ids

    def _uses_scene_coverage(self, subtype_id: Optional[str] = None) -> bool:
        if self.collection_mode != "per_scene":
            return False
        if subtype_id is None:
            return bool(self.target_scene_ids or self.collected_scene_ids)
        return (
            subtype_id in self.target_scene_ids
            or subtype_id in self.collected_scene_ids
        )

    def get_target_scene_ids(self, subtype_id: str) -> List[str]:
        return list(self.target_scene_ids.get(subtype_id, []))

    def get_collected_scene_ids(self, subtype_id: str) -> List[str]:
        return list(self.collected_scene_ids.get(subtype_id, []))

    def get_collected(self, subtype_id: str) -> int:
        if self._uses_scene_coverage(subtype_id):
            return len(self.collected_scene_ids.get(subtype_id, []))
        return self.collected_demos.get(subtype_id, 0)

    def get_target(self, subtype_id: str) -> int:
        if self._uses_scene_coverage(subtype_id):
            return len(self.target_scene_ids.get(subtype_id, []))
        return self.target_demos.get(subtype_id, 0)

    def get_augmented(self, subtype_id: str) -> int:
        return self.augmented_demos.get(subtype_id, 0)

    def increment_collected(self, subtype_id: str, n: int = 1):
        self.collected_demos[subtype_id] = self.collected_demos.get(subtype_id, 0) + n

    def set_target_scene_ids(self, subtype_id: str, scene_ids: List[str]):
        self.target_scene_ids[subtype_id] = sorted(set(scene_ids))
        self.target_demos[subtype_id] = len(self.target_scene_ids[subtype_id])

    def set_collected_scene_ids(self, subtype_id: str, scene_ids: List[str]):
        self.collected_scene_ids[subtype_id] = sorted(set(scene_ids))
        self.collected_demos[subtype_id] = len(self.collected_scene_ids[subtype_id])

    def increment_augmented(self, subtype_id: str, n: int = 1):
        self.augmented_demos[subtype_id] = self.augmented_demos.get(subtype_id, 0) + n

    def is_collection_complete(self, subtype_id: str) -> bool:
        return self.get_collected(subtype_id) >= self.get_target(subtype_id)

    def remaining(self, subtype_id: str) -> int:
        return max(0, self.get_target(subtype_id) - self.get_collected(subtype_id))

    def total_collected(self) -> int:
        if self._uses_scene_coverage():
            keys = set(self.target_scene_ids) | set(self.collected_scene_ids)
            return sum(self.get_collected(subtype_id) for subtype_id in keys)
        return sum(self.collected_demos.values())

    def total_target(self) -> int:
        if self._uses_scene_coverage():
            keys = set(self.target_scene_ids) | set(self.target_demos)
            return sum(self.get_target(subtype_id) for subtype_id in keys)
        return sum(self.target_demos.values())

    def summary(self) -> Dict:
        complete = sum(1 for sid in self.target_demos if self.is_collection_complete(sid))
        return {
            'task_name': self.task_name,
            'collection_mode': self.collection_mode,
            'total_target': self.total_target(),
            'total_collected': self.total_collected(),
            'total_augmented': sum(self.augmented_demos.values()),
            'subtypes_complete': complete,
            'subtypes_total': len(self.target_demos),
        }

    def to_dict(self) -> dict:
        return {
            'task_name': self.task_name,
            'collection_mode': self.collection_mode,
            'target_demos': self.target_demos.copy(),
            'collected_demos': self.collected_demos.copy(),
            'augmented_demos': self.augmented_demos.copy(),
            'target_scene_ids': {k: list(v) for k, v in self.target_scene_ids.items()},
            'collected_scene_ids': {k: list(v) for k, v in self.collected_scene_ids.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RecoveryCollectionStatus':
        return cls(
            task_name=d.get('task_name', ''),
            collection_mode=d.get('collection_mode', 'quota'),
            target_demos=d.get('target_demos', {}),
            collected_demos=d.get('collected_demos', {}),
            augmented_demos=d.get('augmented_demos', {}),
            target_scene_ids=d.get('target_scene_ids', {}),
            collected_scene_ids=d.get('collected_scene_ids', {}),
        )
