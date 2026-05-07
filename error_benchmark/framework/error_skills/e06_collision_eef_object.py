#!/usr/bin/env python
"""
E7: collision_eef_object - Collide EEF with a non-target object.

Degrees: D0 (gentle collision), D1 (aggressive collision with larger overshoot)
Phases: reach, pre_grasp, transport
Injection: Divert EEF toward a non-target object, striking it
Recovery: Detect non-target disturbance, re-plan to avoid displaced object
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class CollisionEefObjectSkill(BaseErrorSkill):
    """
    Divert the EEF to hit a non-target object, displacing it and
    potentially complicating the task.
    """

    @property
    def name(self) -> str:
        return "collision_eef_object"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["reach", "pre_grasp", "transport"]

    def _get_target_and_nontarget(
        self, objects: Dict[str, Any], trajectory_context: Dict[str, Any]
    ):
        """
        Identify the target object (being grasped/transported) and
        non-target objects from the object set.

        Returns:
            (target_name, list_of_nontarget_names) or (None, [])
        """
        obj_names = list(objects.keys())
        if len(obj_names) < 2:
            return None, []

        # The first object is assumed to be the primary manipulation target
        target = trajectory_context.get('target_object', obj_names[0])
        nontargets = [n for n in obj_names if n != target]
        return target, nontargets

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when multiple objects exist and EEF is not holding the target.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Multiple objects must exist
        objects = frame_state.get('objects', {})
        target, nontargets = self._get_target_and_nontarget(objects, trajectory_context)
        if not nontargets:
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
        Divert EEF to collide with a non-target object.

        D0: Move toward non-target with gentle contact (easy).
        D1: Aggressive hit with larger overshoot (hard).
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        _, nontargets = self._get_target_and_nontarget(objects, frame_state)

        # Pick a random non-target object (defensive: can_inject guards empty list)
        if not nontargets:
            return self.create_error_spec(
                degree=degree, target={"eef": True},
                params={"error": "no_nontargets"},
                frame_index=frame_state.get('step', 0),
                trajectory_id=frame_state.get('trajectory_id', ''),
                seed=int(rng.randint(0, 2**31)),
            )
        victim_name = nontargets[rng.randint(0, len(nontargets))]

        # Record pre-injection positions
        pre_obj_positions = {}
        for obj_name in objects:
            try:
                pos, _ = env.get_object_pose(obj_name)
                pre_obj_positions[obj_name] = pos.copy().tolist()
            except Exception:
                pre_obj_positions[obj_name] = objects[obj_name].get('pos', [0, 0, 0])

        # Get victim position from environment
        try:
            victim_pos, _ = env.get_object_pose(victim_name)
            victim_pos = victim_pos.copy()
        except Exception:
            victim_pos = np.array(objects[victim_name].get('pos', [0, 0, 0]))

        # Compute collision target: aim at victim center
        collision_target = victim_pos.copy()

        # Degree-specific overshoot
        if degree == "D0":
            # D0: gentle collision with small overshoot (easy)
            max_steps = degree_params.get('max_steps', 35)
            overshoot = rng.uniform(0.02, 0.04)
            direction = collision_target[:2] - pre_eef_pos[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            collision_target[:2] += direction * overshoot
        elif degree == "D1":
            # D1: aggressive collision with larger overshoot (hard)
            max_steps = degree_params.get('max_steps', 40)
            overshoot = rng.uniform(0.04, 0.08)
            direction = collision_target[:2] - pre_eef_pos[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            collision_target[:2] += direction * overshoot

        # Execute collision
        steps_taken = env.move_eef_to(
            target_pos=collision_target,
            max_steps=int(max_steps),
            render_fn=render_fn,
        )

        # Let physics settle
        for _ in range(10):
            action = env.get_neutral_action()
            env.step(action)
            if render_fn is not None:
                render_fn()

        # Record post-injection positions
        post_obj_positions = {}
        for obj_name in objects:
            try:
                pos, _ = env.get_object_pose(obj_name)
                post_obj_positions[obj_name] = pos.copy().tolist()
            except Exception:
                post_obj_positions[obj_name] = [0, 0, 0]

        return self.create_error_spec(
            degree=degree,
            target={"body": victim_name},
            params={
                "pre_eef_pos": pre_eef_pos.tolist(),
                "collision_target": collision_target.tolist(),
                "victim_object": victim_name,
                "pre_obj_positions": pre_obj_positions,
                "post_obj_positions": post_obj_positions,
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
        Verify that the non-target object was displaced from its original position.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        objects = pre_state.get('objects', {})
        if len(objects) < 2:
            return ValidationResult_v5(
                ok=False,
                error_name=self.name,
                degree=degree,
                reason="Fewer than 2 objects in pre_state",
            )

        # Identify non-target objects (use target_object from pre_state if available)
        obj_names = list(objects.keys())
        target_name = pre_state.get('target_object') or obj_names[0]
        if target_name not in obj_names:
            target_name = obj_names[0]
        nontarget_names = [n for n in obj_names if n != target_name]

        # Check displacement of non-target objects
        max_displacement = 0.0
        displaced_obj = None

        for obj_name in nontarget_names:
            pre_pos = np.array(objects[obj_name].get('pos', [0, 0, 0]))
            try:
                current_pos, _ = env.get_object_pose(obj_name)
                displacement = np.linalg.norm(current_pos - pre_pos)
                if displacement > max_displacement:
                    max_displacement = displacement
                    displaced_obj = obj_name
            except Exception:
                continue

        min_displacement = self.config.get('min_nontarget_displacement', 0.005)
        ok = max_displacement >= min_displacement

        metrics = {
            'max_nontarget_displacement': float(max_displacement),
            'displaced_object': displaced_obj or "",
            'target_object': target_name,
        }

        if ok:
            reason = (
                f"Non-target collision confirmed: {displaced_obj} displaced "
                f"{max_displacement:.3f}m"
            )
        else:
            reason = (
                f"No significant non-target displacement: "
                f"max={max_displacement:.3f}m (threshold={min_displacement})"
            )

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
