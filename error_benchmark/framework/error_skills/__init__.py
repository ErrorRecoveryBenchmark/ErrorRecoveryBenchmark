"""
Error Skills Package - v5.0 Error Injection Framework

Each Error Skill bundles detection, injection, and validation into a single
self-contained unit, replacing the v4 detector + injector + validator pattern.

Registry:
    SKILL_REGISTRY maps error_name -> skill_class
    get_skill(name, config) -> BaseErrorSkill instance
    get_all_skills(config) -> list of all skill instances
"""

from typing import Dict, Type, Optional, List
from .base_skill import BaseErrorSkill, SkillConfig

from .e01_grasp_misalignment import GraspMisalignmentSkill
from .e02_drop_in_transit import DropInTransitSkill
from .e03_drop_at_wrong_place import DropAtWrongPlaceSkill
from .e04_collision_holding import CollisionHoldingSkill
from .e05_collision_empty import CollisionEmptySkill
from .e06_collision_eef_object import CollisionEefObjectSkill
from .e07_collision_self import CollisionSelfSkill
from .e08_wrong_object import WrongObjectSkill
from .e09_trajectory_regression import TrajectoryRegressionSkill
from .e10_stuck_no_progress import StuckNoProgressSkill
from .e11_grasp_wrong_pose import GraspWrongPoseSkill
from .e12_position_error import PositionErrorSkill

# Skill Registry
SKILL_REGISTRY: Dict[str, Type[BaseErrorSkill]] = {
    "grasp_misalignment": GraspMisalignmentSkill,
    "drop_in_transit": DropInTransitSkill,
    "drop_at_wrong_place": DropAtWrongPlaceSkill,
    "collision_holding": CollisionHoldingSkill,
    "collision_empty": CollisionEmptySkill,
    "collision_eef_object": CollisionEefObjectSkill,
    "collision_self": CollisionSelfSkill,
    "wrong_object": WrongObjectSkill,
    "trajectory_regression": TrajectoryRegressionSkill,
    "stuck_no_progress": StuckNoProgressSkill,
    "grasp_wrong_pose": GraspWrongPoseSkill,
    "position_error": PositionErrorSkill,
}


def get_skill(name: str, config: Optional[dict] = None) -> BaseErrorSkill:
    """
    Instantiate an error skill by name.

    Args:
        name: Skill name (e.g., 'drop_in_transit', 'stuck_no_progress')
        config: Skill-specific config dict from benchmark_v5.yaml

    Returns:
        BaseErrorSkill instance

    Raises:
        KeyError: If skill name not found in registry
    """
    if name not in SKILL_REGISTRY:
        raise KeyError(
            f"Unknown error skill: '{name}'. "
            f"Available: {list(SKILL_REGISTRY.keys())}"
        )

    skill_config = SkillConfig(params=config or {})
    return SKILL_REGISTRY[name](config=skill_config)


def get_all_skills(skills_config: Optional[dict] = None) -> List[BaseErrorSkill]:
    """
    Instantiate all registered error skills.

    Args:
        skills_config: Dict mapping skill_name -> config_dict

    Returns:
        List of all BaseErrorSkill instances
    """
    skills_config = skills_config or {}
    skills = []
    for name, cls in SKILL_REGISTRY.items():
        config = skills_config.get(name, {})
        skill_config = SkillConfig(params=config)
        skills.append(cls(config=skill_config))
    return skills


__all__ = [
    "BaseErrorSkill",
    "SkillConfig",
    "SKILL_REGISTRY",
    "get_skill",
    "get_all_skills",
    "GraspMisalignmentSkill",
    "DropInTransitSkill",
    "DropAtWrongPlaceSkill",
    "CollisionHoldingSkill",
    "CollisionEmptySkill",
    "CollisionEefObjectSkill",
    "CollisionSelfSkill",
    "WrongObjectSkill",
    "TrajectoryRegressionSkill",
    "StuckNoProgressSkill",
    "GraspWrongPoseSkill",
    "PositionErrorSkill",
]
