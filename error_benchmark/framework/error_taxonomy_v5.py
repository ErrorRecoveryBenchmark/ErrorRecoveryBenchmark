#!/usr/bin/env python
"""
Error Taxonomy v5.0 - Error Skill-based Classification

Replaces the v4 three-level hierarchy (3 Family x 8 Category x 24 Type)
with a simpler, task-semantics-based taxonomy:

    12 Error Skills x D0/D1 degrees = 24 subtypes

Degrees (difficulty-based, not strict modality):
    D0 = Easy: small displacement (<10-15cm), prefer no rotation
    D1 = Hard: large displacement (>10-15cm), prefer rotation

Error Skills (all support D0 + D1):
    E1:  grasp_misalignment     D0/D1  (2 subtypes)
    E2:  drop_in_transit        D0/D1  (2 subtypes)
    E3:  drop_at_wrong_place    D0/D1  (2 subtypes)  [includes former E4 interaction mode]
    E4:  collision_holding      D0/D1  (2 subtypes)
    E5:  collision_empty        D0/D1  (2 subtypes)
    E6:  collision_eef_object   D0/D1  (2 subtypes)
    E7:  collision_self         D0/D1  (2 subtypes)
    E8:  wrong_object           D0/D1  (2 subtypes)
    E9:  trajectory_regression  D0/D1  (2 subtypes)
    E10: stuck_no_progress      D0/D1  (2 subtypes)
    E11: grasp_wrong_pose       D0/D1  (2 subtypes)
    E12: position_error         D0/D1  (2 subtypes)
    ───────────────────────────── Total: 24 subtypes
"""

from enum import Enum
from typing import Dict, List, Any, Tuple


# ─── Degree of error ─────────────────────────────────────

class ErrorDegree(str, Enum):
    """Error degree (difficulty-based)"""
    D0 = "D0"  # Easy: small displacement (<10-15cm), prefer no rotation
    D1 = "D1"  # Hard: large displacement (>10-15cm), prefer rotation


# ─── Error Skill Names ───────────────────────────────────

class ErrorSkillName(str, Enum):
    """12 error skill identifiers"""
    E1_GRASP_MISALIGNMENT = "grasp_misalignment"
    E2_DROP_IN_TRANSIT = "drop_in_transit"
    E3_DROP_AT_WRONG_PLACE = "drop_at_wrong_place"
    E4_COLLISION_HOLDING = "collision_holding"
    E5_COLLISION_EMPTY = "collision_empty"
    E6_COLLISION_EEF_OBJECT = "collision_eef_object"
    E7_COLLISION_SELF = "collision_self"
    E8_WRONG_OBJECT = "wrong_object"
    E9_TRAJECTORY_REGRESSION = "trajectory_regression"
    E10_STUCK_NO_PROGRESS = "stuck_no_progress"
    E11_GRASP_WRONG_POSE = "grasp_wrong_pose"
    E12_POSITION_ERROR = "position_error"


# ─── Error Skill Definitions ─────────────────────────────

ERROR_SKILL_DEFS: Dict[str, Dict[str, Any]] = {
    ErrorSkillName.E1_GRASP_MISALIGNMENT: {
        "id": "E1",
        "name": "grasp_misalignment",
        "description": "Gripper closes too early or with EEF offset, resulting in misaligned grasp",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["pre_grasp", "grasp"],
        "injection_method": "Move EEF to offset position before gripper closes, or close gripper early",
        "recovery_strategy": "Re-open gripper, adjust position, re-grasp",
    },
    ErrorSkillName.E2_DROP_IN_TRANSIT: {
        "id": "E2",
        "name": "drop_in_transit",
        "description": "Drop held object mid-transit, far from target position",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["lift", "transport"],
        "injection_method": "Open gripper to drop object when EEF is far from target",
        "recovery_strategy": "Move to dropped object, re-grasp, resume task",
    },
    ErrorSkillName.E3_DROP_AT_WRONG_PLACE: {
        "id": "E3",
        "name": "drop_at_wrong_place",
        "description": "Near target, drop object at wrong location (offset or onto another object)",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["transport", "place"],
        "injection_method": "Offset EEF away from target or move above non-target object, then release",
        "recovery_strategy": "Locate dropped object, re-grasp, navigate back to target",
    },
    ErrorSkillName.E4_COLLISION_HOLDING: {
        "id": "E4",
        "name": "collision_holding",
        "description": "While holding object, collide it with environment obstacle",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["lift", "transport"],
        "injection_method": "Move EEF so held object hits table edge, walls, or obstacles",
        "recovery_strategy": "Pull back, adjust trajectory, retry transport",
    },
    ErrorSkillName.E5_COLLISION_EMPTY: {
        "id": "E5",
        "name": "collision_empty",
        "description": "Empty gripper (not holding) hits objects on the table",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["pre_reach", "reach"],
        "injection_method": "Move EEF to collide with non-target objects",
        "recovery_strategy": "Retract, re-plan approach trajectory",
    },
    ErrorSkillName.E6_COLLISION_EEF_OBJECT: {
        "id": "E6",
        "name": "collision_eef_object",
        "description": "EEF hits non-target object, displacing it",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["reach", "pre_grasp", "transport"],
        "injection_method": "Divert EEF to hit non-target object body",
        "recovery_strategy": "Recover displaced object or adjust plan",
    },
    ErrorSkillName.E7_COLLISION_SELF: {
        "id": "E7",
        "name": "collision_self",
        "description": "EEF or arm link contacts own robot links",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["reach", "lift", "transport"],
        "injection_method": "Move EEF toward robot base or other arm links",
        "recovery_strategy": "Back away from self-collision configuration",
    },
    ErrorSkillName.E8_WRONG_OBJECT: {
        "id": "E8",
        "name": "wrong_object",
        "description": "Approach and grasp wrong object (multi-object tasks only)",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["reach", "pre_grasp", "grasp"],
        "injection_method": "Redirect EEF to non-target object and grasp it",
        "recovery_strategy": "Release wrong object, move to correct target",
    },
    ErrorSkillName.E9_TRAJECTORY_REGRESSION: {
        "id": "E9",
        "name": "trajectory_regression",
        "description": "Replay trajectory backwards, undoing progress",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["lift", "transport"],
        "injection_method": "Replay recent actions in reverse order",
        "recovery_strategy": "Detect regression, re-execute forward trajectory",
    },
    ErrorSkillName.E10_STUCK_NO_PROGRESS: {
        "id": "E10",
        "name": "stuck_no_progress",
        "description": "Freeze EEF in place, no task progress",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["reach", "pre_grasp", "grasp", "lift", "transport"],
        "injection_method": "Neutral actions (zero input) for freeze_steps, robot stays still",
        "recovery_strategy": "Detect stall, resume active control",
    },
    ErrorSkillName.E11_GRASP_WRONG_POSE: {
        "id": "E11",
        "name": "grasp_wrong_pose",
        "description": "Grasp object with wrong orientation (rotation error)",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["pre_grasp", "grasp"],
        "injection_method": "D0: small rotation; D1: large rotation before grasping",
        "recovery_strategy": "Release, re-orient gripper, re-grasp",
    },
    ErrorSkillName.E12_POSITION_ERROR: {
        "id": "E12",
        "name": "position_error",
        "description": "Move EEF to wrong position (general positioning error)",
        "valid_degrees": [ErrorDegree.D0, ErrorDegree.D1],
        "valid_phases": ["reach", "pre_grasp", "lift", "transport"],
        "injection_method": "D0: small offset; D1: large offset with rotation",
        "recovery_strategy": "Detect deviation, correct to intended trajectory",
    },
}


# ─── Subtype enumeration ─────────────────────────────────

def get_all_subtypes() -> List[Tuple[str, str]]:
    """
    Return all 24 (error_name, degree) subtypes.

    Returns:
        List of (error_skill_name, degree) tuples
    """
    subtypes = []
    for skill_name, skill_def in ERROR_SKILL_DEFS.items():
        for degree in skill_def["valid_degrees"]:
            subtypes.append((skill_name.value, degree.value))
    return subtypes


def get_subtype_id(error_name: str, degree: str) -> str:
    """
    Return a unique subtype identifier like 'grasp_misalignment_D0'.

    Args:
        error_name: Error skill name (e.g., 'grasp_misalignment')
        degree: Degree string (e.g., 'D0')

    Returns:
        Subtype ID string
    """
    return f"{error_name}_{degree}"


def get_skill_def(error_name: str) -> Dict[str, Any]:
    """
    Get the skill definition for an error name.

    Args:
        error_name: Error skill name string

    Returns:
        Skill definition dict
    """
    skill_enum = ErrorSkillName(error_name)
    return ERROR_SKILL_DEFS[skill_enum]


def is_valid_subtype(error_name: str, degree: str) -> bool:
    """
    Check if (error_name, degree) is a valid subtype.

    Args:
        error_name: Error skill name string
        degree: Degree string ('D0' or 'D1')

    Returns:
        True if the combination is valid
    """
    try:
        skill_def = get_skill_def(error_name)
        return ErrorDegree(degree) in skill_def["valid_degrees"]
    except (ValueError, KeyError):
        return False


def get_valid_degrees(error_name: str) -> List[str]:
    """
    Return valid degrees for an error skill.

    Args:
        error_name: Error skill name string

    Returns:
        List of degree strings
    """
    skill_def = get_skill_def(error_name)
    return [d.value for d in skill_def["valid_degrees"]]


def get_valid_phases(error_name: str) -> List[str]:
    """
    Return valid task phases for an error skill.

    Args:
        error_name: Error skill name string

    Returns:
        List of phase strings
    """
    skill_def = get_skill_def(error_name)
    return skill_def["valid_phases"]


# ─── Task phase ordering ─────────────────────────────────

PHASE_ORDER = {
    "pre_reach": 0,
    "reach": 1,
    "pre_grasp": 2,
    "grasp": 3,
    "lift": 4,
    "transport": 5,
    "place": 6,
    "done": 7,
}


# ─── Summary statistics ──────────────────────────────────

TOTAL_SKILLS = len(ERROR_SKILL_DEFS)  # 12
TOTAL_SUBTYPES = len(get_all_subtypes())  # 24
