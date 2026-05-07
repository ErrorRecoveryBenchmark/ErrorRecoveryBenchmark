#!/usr/bin/env python
"""
E6: collision_empty - Collide empty gripper with objects on the table.

Degrees: D0 (gentle approach), D1 (aggressive approach with overshoot)
Phases: pre_reach, reach
Injection: Move EEF into objects without grasping, displacing them
Recovery: Detect displaced objects, re-plan approach to avoid disturbance
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class CollisionEmptySkill(BaseErrorSkill):
    """
    Move the empty gripper into objects on the table, displacing them
    from their original positions.
    """

    @property
    def name(self) -> str:
        return "collision_empty"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["pre_reach", "reach"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when gripper is open and at least one object is on the table.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Gripper must be open (not holding anything)
        gripper_closed = frame_state.get('gripper_closed_norm', 0.0)
        if gripper_closed >= self.GRIPPER_CLOSED_THRESHOLD:
            return []

        # At least one object must exist
        objects = frame_state.get('objects', {})
        if not objects:
            return []

        # Need enough remaining frames for recovery
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
        Move EEF to collide with objects on the table.

        D0: Gentle nudge toward nearest object (easy).
        D1: Aggressive approach with overshoot (hard).
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        obj_names = list(objects.keys())

        # Pick a target object (collision target — use task target for consistency)
        target_obj = frame_state.get('target_object') or obj_names[0]
        if target_obj not in objects:
            target_obj = obj_names[0]
        target_obj_pos = np.array(objects[target_obj]['pos'])

        # Record all pre-injection object positions
        pre_obj_positions = {}
        for obj_name in obj_names:
            try:
                pos, _ = env.get_object_pose(obj_name)
                pre_obj_positions[obj_name] = pos.copy().tolist()
            except Exception:
                pre_obj_positions[obj_name] = objects[obj_name]['pos']

        # Compute approach: move EEF to object height then sweep through
        approach_height_offset = degree_params.get('approach_height_offset', 0.0)
        collision_target = target_obj_pos.copy()
        collision_target[2] += approach_height_offset  # Aim at object center height

        if degree == "D0":
            # D0: gentle, short approach (easy)
            speed_scale = degree_params.get('speed_scale', 1.0)
            max_steps = degree_params.get('max_steps', 30)
        elif degree == "D1":
            # D1: aggressive approach with overshoot (hard)
            speed_scale = degree_params.get('speed_scale', 2.0)
            max_steps = degree_params.get('max_steps', 40)
            overshoot = rng.uniform(0.04, 0.08)
            direction = collision_target[:2] - pre_eef_pos[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            collision_target[:2] += direction * overshoot

        # Move EEF to collision target (gripper stays open)
        steps_taken = env.move_eef_to(
            target_pos=collision_target,
            max_steps=int(max_steps),
            render_fn=render_fn,
        )

        # Let physics settle
        self.settle_physics(env, 5, gripper='open', render_fn=render_fn)

        # Record post-injection object positions
        post_obj_positions = {}
        for obj_name in obj_names:
            try:
                pos, _ = env.get_object_pose(obj_name)
                post_obj_positions[obj_name] = pos.copy().tolist()
            except Exception:
                post_obj_positions[obj_name] = [0, 0, 0]

        return self.create_error_spec(
            degree=degree,
            target={"body": target_obj},
            params={
                "pre_eef_pos": pre_eef_pos.tolist(),
                "collision_target": collision_target.tolist(),
                "pre_obj_positions": pre_obj_positions,
                "post_obj_positions": post_obj_positions,
                "target_object": target_obj,
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
        Verify that at least one object was displaced from its original position.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        objects = pre_state.get('objects', {})
        if not objects:
            return ValidationResult_v5(
                ok=False,
                error_name=self.name,
                degree=degree,
                reason="No objects in pre_state",
            )

        # Check displacement of each object
        max_displacement = 0.0
        displaced_obj = None

        for obj_name, obj_info in objects.items():
            pre_pos = np.array(obj_info.get('pos', [0, 0, 0]))
            try:
                current_pos, _ = env.get_object_pose(obj_name)
                displacement = np.linalg.norm(current_pos - pre_pos)
                if displacement > max_displacement:
                    max_displacement = displacement
                    displaced_obj = obj_name
            except Exception:
                continue

        min_displacement = self.config.get('min_object_displacement', 0.01)
        ok = max_displacement >= min_displacement

        metrics = {
            'max_object_displacement': float(max_displacement),
            'displaced_object': displaced_obj or "",
        }

        if ok:
            reason = (
                f"Collision confirmed: {displaced_obj} displaced "
                f"{max_displacement:.3f}m"
            )
        else:
            reason = (
                f"No significant displacement: max={max_displacement:.3f}m "
                f"(threshold={min_displacement})"
            )

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
