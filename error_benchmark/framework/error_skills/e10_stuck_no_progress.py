#!/usr/bin/env python
"""
E10: stuck_no_progress - Freeze EEF in place, no task progress.

Degrees:
  D0: shorter freeze with limited disruption
  D1: longer freeze that stalls the task more severely
Phases: reach, pre_grasp, grasp, lift, transport
Injection: Output zero/neutral actions to freeze robot
Recovery: Detect stall, resume active control
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class StuckNoProgressSkill(BaseErrorSkill):
    """
    Freeze the EEF in place by outputting neutral actions.
    The robot makes no progress toward the task goal.
    """

    @property
    def name(self) -> str:
        return "stuck_no_progress"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["reach", "pre_grasp", "grasp", "lift", "transport"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject at any frame where the task is not yet done.
        This is the most broadly applicable skill.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if phase in ('done', 'place'):
            return []

        if not self.supports_phase(phase):
            return []

        # Must have at least some frames remaining
        frame_idx = trajectory_context.get('frame_index', 0)
        total_frames = trajectory_context.get('total_frames', 0)
        if total_frames - frame_idx < 10:
            return []

        return list(self.valid_degrees)

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Execute neutral actions for freeze_steps to simulate being stuck.
        Pure freeze — robot outputs zero/neutral actions and stays in place.
        """
        degree_params = self.config.get_degree_params(degree)
        freeze_steps = degree_params.get('freeze_steps', 20)

        # Record pre-injection EEF position
        pre_eef_pos = env.get_eef_pos().copy()

        for _ in range(freeze_steps):
            action = env.get_neutral_action()
            env.step(action)
            if render_fn is not None:
                render_fn()

        # Build ErrorSpec
        return self.create_error_spec(
            degree=degree,
            target={"eef": True},
            params={
                "degree": degree,
                "freeze_steps": freeze_steps,
                "pre_eef_pos": pre_eef_pos.tolist(),
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
        Validate that the robot made no progress during the freeze period.
        Check that EEF displacement is below threshold.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        degree_params = self.config.get_degree_params(degree)
        freeze_steps = degree_params.get('freeze_steps', 20)
        min_progress = self.config.get('min_progress_threshold', 0.002)
        max_displacement = min_progress * freeze_steps

        # Check EEF displacement during the freeze
        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        displacement = np.linalg.norm(current_eef - pre_eef)

        # The robot should have barely moved
        ok = displacement < max_displacement

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics={
                'eef_displacement': float(displacement),
                'threshold': float(max_displacement),
            },
            reason="EEF frozen in place" if ok else f"EEF moved {displacement:.4f}m (too much)",
            post_rollout_stats=post_stats,
        )
