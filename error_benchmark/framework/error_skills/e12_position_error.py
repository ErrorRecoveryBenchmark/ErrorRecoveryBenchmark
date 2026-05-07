#!/usr/bin/env python
"""
E13: position_error - Move EEF to wrong position.

Degrees: D0 (translation only), D1 (larger translation + rotation)
Phases: reach, pre_grasp, lift, transport
Injection: Add offset to EEF target position and/or rotation
Recovery: Detect deviation, correct to intended trajectory

A general-purpose positioning error that displaces the robot
from its intended trajectory.
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class PositionErrorSkill(BaseErrorSkill):
    """
    Move the EEF to a wrong position/orientation by applying an offset.
    """

    @property
    def name(self) -> str:
        return "position_error"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["reach", "pre_grasp", "lift", "transport"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject at any frame in a valid phase, as long as there's room to move.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Need enough remaining frames
        return self.check_remaining_frames(trajectory_context)

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Apply position/rotation offset to EEF.
        D0: translation offset only (easy).
        D1: translation offset + rotation (hard).
        """
        from error_benchmark.framework.utils.math_utils import (
            random_translation, random_rotation_quat, euler_to_quat, quat_multiply
        )

        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()
        pre_eef_quat = env.get_eef_quat().copy()

        pos_offset = np.zeros(3)
        rot_offset = np.zeros(3)

        if degree == "D0":
            # D0: translation offset only (easy)
            offset_range = degree_params.get('offset_range', [0.02, 0.10])
            magnitude = rng.uniform(offset_range[0], offset_range[1])
            pos_offset = random_translation(rng, magnitude)

        if degree == "D1":
            # D1: larger translation offset + rotation (hard)
            offset_range = degree_params.get('offset_range', [0.02, 0.10])
            magnitude = rng.uniform(offset_range[0], offset_range[1])
            pos_offset = random_translation(rng, magnitude)

            rotation_range = degree_params.get('rotation_range', [0.2, 0.8])
            max_angle = rng.uniform(rotation_range[0], rotation_range[1])
            rot_offset = self.random_direction(rng, dims=3) * max_angle

        # Execute the offset movement with tight convergence thresholds
        steps_taken = env.apply_eef_offset(
            pos_offset=pos_offset,
            rot_offset=rot_offset if np.any(np.abs(rot_offset) > 1e-8) else None,
            max_steps=80,
            pos_threshold=0.003,
            rot_threshold=0.03,
            render_fn=render_fn,
        )

        return self.create_error_spec(
            degree=degree,
            target={"eef": True},
            params={
                "pos_offset": pos_offset.tolist(),
                "rot_offset": rot_offset.tolist(),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_eef_quat": pre_eef_quat.tolist(),
                "steps_taken": steps_taken,
            },
            frame_index=frame_state.get('step', 0),
            trajectory_id=frame_state.get('trajectory_id', ''),
            seed=int(rng.randint(0, 2**31)),
        )

    def validate(
        self,
        env: 'EnvWrapper',
        pre_state: Dict[str, Any],
        post_stats: 'PostRolloutStats',
        degree: str,
    ) -> 'ValidationResult_v5':
        """
        Verify that the EEF actually moved to a wrong position.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        displacement = np.linalg.norm(current_eef - pre_eef)

        # D0: check position displacement only
        pos_ok = True
        if degree == "D0":
            min_displacement = 0.01  # At least 1cm displacement
            pos_ok = displacement >= min_displacement

        # D1: check both position displacement and rotation change
        rot_ok = True
        rot_dist = 0.0
        if degree == "D1":
            min_displacement = 0.01  # At least 1cm displacement
            pos_ok = displacement >= min_displacement
            current_quat = env.get_eef_quat()
            pre_quat = np.array(pre_state.get('eef_quat', [1, 0, 0, 0]))
            dot = abs(np.dot(current_quat, pre_quat))
            rot_dist = 2 * np.arccos(np.clip(dot, -1, 1))
            min_rotation = 0.1  # At least ~5.7 degrees
            rot_ok = rot_dist >= min_rotation

        ok = pos_ok and rot_ok

        metrics = {
            'eef_displacement': float(displacement),
        }
        if degree == "D1":
            metrics['rotation_distance'] = float(rot_dist)

        reason = "Position error confirmed" if ok else "Insufficient displacement"

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
