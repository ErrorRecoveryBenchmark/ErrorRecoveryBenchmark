#!/usr/bin/env python
"""
BaseErrorSkill - Abstract base class for v5 Error Skills.

Each Error Skill bundles:
    - can_inject(): Offline detection (which frames are injectable)
    - inject(): Action-based injection (move EEF, open gripper, etc.)
    - validate(): Post-injection verification (did the error actually occur)

This eliminates the v4 detector-injector-validator mismatch problem
that caused ~85% rejection rates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import (
        ErrorSpec, InjectionOpportunity, ValidationResult_v5, PostRolloutStats
    )

logger = logging.getLogger(__name__)


@dataclass
class SkillConfig:
    """Configuration for an error skill, loaded from benchmark_v5.yaml."""
    params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.params.get(key, default)

    def get_degree_params(self, degree: str) -> Dict[str, Any]:
        """Get degree-specific parameters (D0/D1)."""
        return self.params.get(degree, {})


class BaseErrorSkill(ABC):
    """
    Abstract base class for Error Skills.

    Each skill implements the three-phase pattern:
        1. can_inject() — Offline scan: is this frame injectable?
        2. inject() — Execute the injection using action-based control
        3. validate() — Verify the error actually occurred

    Subclasses must define:
        - name: Error skill identifier (matches ErrorSkillName enum)
        - valid_degrees: List of applicable degrees (D0/D1)
        - valid_phases: List of task phases where this error applies
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        """
        Args:
            config: Skill-specific configuration from benchmark_v5.yaml
        """
        self.config = config or SkillConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Error skill name (e.g., 'stuck_no_progress')."""
        ...

    @property
    @abstractmethod
    def valid_degrees(self) -> List[str]:
        """List of valid degrees for this skill (e.g., ['D0', 'D1'])."""
        ...

    @property
    @abstractmethod
    def valid_phases(self) -> List[str]:
        """List of task phases where this error can be injected."""
        ...

    @abstractmethod
    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Check if injection is possible at this frame. Called during offline scan.

        Args:
            frame_state: State info dict at the current frame (from StateExtractor)
            trajectory_context: Context about the full trajectory:
                - 'frame_index': int - current frame index
                - 'total_frames': int - total frames in trajectory
                - 'task_phase': str - current task phase
                - 'prev_phases': List[str] - phases of previous frames
                - 'objects': Dict - object states
                - 'eef_pos': np.ndarray - EEF position
                - 'gripper_closed_norm': float

        Returns:
            List of injectable degree strings (e.g., ['D0', 'D1']).
            Empty list means this frame is not injectable for this skill.
        """
        ...

    @abstractmethod
    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Execute the error injection using action-based control.

        The environment is already at the target frame state when this is called.
        The skill should use env.move_eef_to(), env.set_gripper_state(), etc.

        Args:
            env: EnvWrapper with environment at the target frame
            degree: Injection degree ('D0' or 'D1')
            frame_state: State info at the injection frame
            rng: Random state for reproducibility
            render_fn: Optional no-arg callback for video frame capture.
                Called after each env.step() during injection.

        Returns:
            ErrorSpec describing the injection that was performed
        """
        ...

    @abstractmethod
    def validate(
        self,
        env: 'EnvWrapper',
        pre_state: Dict[str, Any],
        post_stats: 'PostRolloutStats',
        degree: str,
    ) -> 'ValidationResult_v5':
        """
        Verify that the error actually occurred after injection.

        Args:
            env: EnvWrapper after injection + post-rollout
            pre_state: State info before injection
            post_stats: Trajectory statistics collected after injection
            degree: The degree that was injected

        Returns:
            ValidationResult_v5 with ok=True if error was confirmed
        """
        ...

    def supports_degree(self, degree: str) -> bool:
        """Check if this skill supports the given degree."""
        return degree in self.valid_degrees

    def supports_phase(self, phase: str) -> bool:
        """Check if this skill can inject during the given task phase."""
        return phase in self.valid_phases

    # ── Shared helpers used across multiple skills ──

    # Common thresholds used across error skills
    GRIPPER_CLOSED_THRESHOLD = 0.20   # gripper_closed_norm >= this means "closed"
    OBJECT_PROXIMITY_THRESHOLD = 0.08  # EEF-to-object distance for "near"
    MIN_OBJECT_HEIGHT = 0.93           # minimum z for "held above table"

    @staticmethod
    def _get_phase(
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> str:
        """Extract current task phase from frame_state or trajectory_context."""
        return trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

    def settle_physics(
        self,
        env: 'EnvWrapper',
        steps: int,
        gripper: str = 'close',
        render_fn=None,
    ) -> None:
        """Run neutral actions for physics settlement.

        Args:
            env: EnvWrapper instance
            steps: Number of settlement steps
            gripper: 'close' for closed gripper, 'open' for open gripper
            render_fn: Optional frame capture callback
        """
        if gripper == 'close':
            gripper_val = env.get_gripper_action_close()
        else:
            gripper_val = env.get_gripper_action_open()

        for _ in range(steps):
            action = env.get_neutral_action()
            action[-1] = gripper_val
            env.step(action)
            if render_fn is not None:
                render_fn()

    def find_held_object(
        self,
        objects: Dict[str, Any],
        eef_pos: 'np.ndarray',
        min_height: float = 0.93,
        proximity: float = 0.08,
        z_diff_threshold: float = 0.04,
    ) -> Optional[str]:
        """Find object held by gripper (above min_height AND near EEF).

        Checks are ordered cheapest-first for hot-path performance
        (called once per frame during offline scan).
        """
        for obj_name, obj_info in objects.items():
            obj_z = obj_info['pos'][2]
            if obj_z < min_height:
                continue
            if abs(eef_pos[2] - obj_z) >= z_diff_threshold:
                continue
            obj_pos = np.array(obj_info['pos'])
            if np.linalg.norm(eef_pos - obj_pos) < proximity:
                return obj_name
        return None

    def get_safe_object_pose(
        self,
        env: 'EnvWrapper',
        obj_name: str,
    ):
        """Get object pose with safe fallback on error.

        Returns:
            (pos_copy, quat_copy) or (zeros, identity_quat) on failure.
        """
        try:
            pos, quat = env.get_object_pose(obj_name)
            return pos.copy(), quat.copy()
        except (ValueError, KeyError, AttributeError, RuntimeError):
            return np.zeros(3), np.array([1, 0, 0, 0])

    def open_gripper_joints_direct(self, sim) -> None:
        """Set all gripper/finger joint positions to 0 (open) via direct MuJoCo access.

        Used by drop skills because the OSC controller compensates gripper commands.
        """
        for i in range(sim.model.njnt):
            jnt_name = sim.model.joint_id2name(i)
            if 'gripper' in jnt_name.lower() or 'finger' in jnt_name.lower():
                addr = sim.model.jnt_qposadr[i]
                sim.data.qpos[addr] = 0.0
                dof_addr = sim.model.jnt_dofadr[i]
                sim.data.qvel[dof_addr] = 0.0

    def zero_gripper_actuators(self, sim) -> None:
        """Set all gripper/finger actuator controls to 0."""
        for i in range(sim.model.nu):
            act_name = sim.model.actuator_id2name(i)
            if 'gripper' in act_name.lower() or 'finger' in act_name.lower():
                sim.data.ctrl[i] = 0.0

    def find_object_joint(self, env: 'EnvWrapper', obj_name: str):
        """Find the MuJoCo free joint for a named object.

        Returns (joint_id, qpos_addr, qvel_addr) or (None, None, None).
        """
        sim = env._env.sim
        body_name = None
        for obj_cfg in env._task_config.get('objects', []):
            if obj_cfg['name'] == obj_name:
                body_name = obj_cfg.get('body_name', obj_name)
                break
        if body_name is None:
            body_name = obj_name

        for i in range(sim.model.njnt):
            jnt_body_id = sim.model.jnt_bodyid[i]
            jnt_body_name = sim.model.body_id2name(jnt_body_id)
            if jnt_body_name == body_name and sim.model.jnt_type[i] == 0:  # free joint
                return i, sim.model.jnt_qposadr[i], sim.model.jnt_dofadr[i]
        return None, None, None

    def has_non_ground_contact(
        self,
        contacts,
        obj_name: str,
        ground_keywords=('table', 'floor', 'ground', 'bin'),
    ):
        """Check if object has contact with non-ground geometry.

        Returns:
            (has_contact: bool, interacting_geom: str)
        """
        for c in contacts:
            if obj_name not in c.geom1 and obj_name not in c.geom2:
                continue
            other_geom = c.geom2 if obj_name in c.geom1 else c.geom1
            if not any(kw in other_geom.lower() for kw in ground_keywords):
                return True, other_geom
        return False, ""

    # ── Refactoring helpers (shared patterns across skills) ──

    def check_remaining_frames(
        self,
        trajectory_context: Dict[str, Any],
        min_frames: int = 20,
        d1_min_frames: int = 30,
    ) -> List[str]:
        """Check remaining frame budget and return injectable degrees.

        Standard pattern: D0 needs >= min_frames remaining, D1 needs >= d1_min_frames.
        Skills with non-standard logic (e09, e10) should implement their own checks.
        """
        frame_idx = trajectory_context.get('frame_index', 0)
        total_frames = trajectory_context.get('total_frames', 0)
        remaining = total_frames - frame_idx
        if remaining < min_frames:
            return []
        injectable = ['D0']
        if remaining >= d1_min_frames:
            injectable.append('D1')
        return injectable

    @staticmethod
    def normalize_direction(vec: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Return a unit-length copy of vec. If near-zero, return a copy unchanged.

        Returns a NEW array — never mutates the input.
        """
        norm = np.linalg.norm(vec)
        if norm > epsilon:
            return vec / norm
        return vec.copy()

    @staticmethod
    def random_direction(rng: np.random.RandomState, dims: int = 2) -> np.ndarray:
        """Generate a random unit direction vector.

        Returns a new unit vector of the given dimensionality.
        """
        d = rng.randn(dims)
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            d = d / norm
        return d

    def create_error_spec(
        self,
        degree: str,
        target: Dict,
        params: Dict,
        frame_index: int,
        trajectory_id: str,
        seed: int,
    ) -> 'ErrorSpec':
        """
        Helper to create an ErrorSpec with v5 fields populated.

        Args:
            degree: 'D0' or 'D1'
            target: Target dict (e.g., {'body': 'cube_main'})
            params: Skill-specific parameters
            frame_index: Frame index in clean trajectory
            trajectory_id: Clean trajectory identifier
            seed: Random seed

        Returns:
            ErrorSpec with v5 fields set
        """
        from error_benchmark.framework.core import ErrorSpec

        return ErrorSpec(
            type="error_skill",
            family="action_based",
            target=target,
            params=params,
            apply={"mode": "at_step", "t": frame_index},
            seed=seed,
            direction_strategy="",
            error_name=self.name,
            degree=degree,
            source_frame=frame_index,
            source_trajectory=trajectory_id,
        )
