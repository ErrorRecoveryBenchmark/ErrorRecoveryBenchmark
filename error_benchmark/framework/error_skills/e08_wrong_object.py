#!/usr/bin/env python
"""
E9: wrong_object - Redirect EEF to a non-target object and attempt grasp.

Degrees: D0, D1
Phases: reach, pre_grasp, grasp
Injection: Divert EEF to a non-target object and close gripper
Recovery: Detect wrong object grasped, release, redirect to correct target
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class WrongObjectSkill(BaseErrorSkill):
    """
    Redirect the EEF to a non-target object and attempt to grasp it,
    simulating a target-selection error in multi-object tasks.
    """

    @property
    def name(self) -> str:
        return "wrong_object"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["reach", "pre_grasp", "grasp"]

    def _get_target_and_nontarget(
        self, objects: Dict[str, Any], trajectory_context: Dict[str, Any]
    ):
        """
        Identify target and non-target graspable objects.

        Only considers objects with grasp_geoms (excludes fixtures like
        coffee_machine, base, etc.) to ensure wrong targets are reachable.

        Returns:
            (target_name, list_of_nontarget_names) or (None, [])
        """
        # Use graspable_objects list if available (from scanner/context_replay),
        # otherwise fall back to all objects in the scene
        graspable = trajectory_context.get('graspable_objects')
        if graspable:
            obj_names = [n for n in graspable if n in objects]
        else:
            obj_names = list(objects.keys())

        if len(obj_names) < 2:
            return None, []

        target = trajectory_context.get('target_object', obj_names[0])
        nontargets = [n for n in obj_names if n != target]
        return target, nontargets

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when multiple objects exist (multi-object tasks only).
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Multiple objects must exist
        objects = frame_state.get('objects', {})
        target, nontargets = self._get_target_and_nontarget(objects, trajectory_context)
        if not nontargets:
            return []

        # Gripper should be open (not already grasping)
        gripper_closed = frame_state.get('gripper_closed_norm', 0.0)
        if gripper_closed >= self.GRIPPER_CLOSED_THRESHOLD:
            return []

        # Need enough remaining frames
        return self.check_remaining_frames(trajectory_context, min_frames=30, d1_min_frames=40)

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Redirect EEF to a non-target object and attempt grasp.

        D0: Move to wrong object and close gripper at its position.
        D1: Move to wrong object, close gripper, and attempt to lift.
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        target, nontargets = self._get_target_and_nontarget(objects, frame_state)

        # Pick a random non-target object (defensive: can_inject guards empty list)
        if not nontargets:
            return self.create_error_spec(
                degree=degree, target={"eef": True},
                params={"error": "no_nontargets"},
                frame_index=frame_state.get('step', 0),
                trajectory_id=frame_state.get('trajectory_id', ''),
                seed=int(rng.randint(0, 2**31)),
            )
        wrong_obj = nontargets[rng.randint(0, len(nontargets))]

        # Record pre-injection state
        pre_obj_positions = {}
        for obj_name in objects:
            try:
                pos, _ = env.get_object_pose(obj_name)
                pre_obj_positions[obj_name] = pos.copy().tolist()
            except Exception:
                pre_obj_positions[obj_name] = objects[obj_name].get('pos', [0, 0, 0])

        # Get wrong object position from environment
        try:
            wrong_obj_pos, _ = env.get_object_pose(wrong_obj)
            wrong_obj_pos = wrong_obj_pos.copy()
        except Exception:
            wrong_obj_pos = np.array(objects[wrong_obj].get('pos', [0, 0, 0]))

        # Move to slightly above wrong object for pre-grasp
        grasp_approach = wrong_obj_pos.copy()
        grasp_height_offset = degree_params.get('grasp_height_offset', 0.03)
        grasp_approach[2] += grasp_height_offset

        approach_steps = degree_params.get('approach_steps', 60)
        steps_to_approach = env.move_eef_to(
            target_pos=grasp_approach,
            max_steps=int(approach_steps),
            render_fn=render_fn,
        )

        # Move down to grasp position
        grasp_target = wrong_obj_pos.copy()
        grasp_target[2] += degree_params.get('grasp_descent_offset', 0.005)
        env.move_eef_to(target_pos=grasp_target, max_steps=20, render_fn=render_fn)

        # Close gripper to attempt grasp
        gripper_close_steps = degree_params.get('gripper_close_steps', 30)
        env.set_gripper_state(open_fraction=0.0, steps=gripper_close_steps, render_fn=render_fn)

        # D1: also attempt to lift the wrong object
        lift_steps = 0
        if degree == "D1":
            lift_height = degree_params.get('lift_height', 0.05)
            lift_target = env.get_eef_pos().copy()
            lift_target[2] += lift_height
            lift_steps = env.move_eef_to(
                target_pos=lift_target,
                max_steps=30,
                render_fn=render_fn,
            )

        # Let physics settle
        self.settle_physics(env, 15, gripper='close', render_fn=render_fn)

        return self.create_error_spec(
            degree=degree,
            target={"body": wrong_obj},
            params={
                "pre_eef_pos": pre_eef_pos.tolist(),
                "target_object": target,
                "wrong_object": wrong_obj,
                "wrong_obj_pos": wrong_obj_pos.tolist(),
                "pre_obj_positions": pre_obj_positions,
                "steps_to_approach": steps_to_approach,
                "gripper_close_steps": gripper_close_steps,
                "lift_steps": lift_steps,
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
        Verify that the EEF is near the wrong object (not the target).
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

        obj_names = list(objects.keys())
        target_name = pre_state.get('target_object') or obj_names[0]
        if target_name not in obj_names:
            target_name = obj_names[0]
        nontarget_names = [n for n in obj_names if n != target_name]

        current_eef = env.get_eef_pos()

        # Check distance to each non-target object
        min_dist_nontarget = float('inf')
        closest_nontarget = None

        for obj_name in nontarget_names:
            try:
                obj_pos, _ = env.get_object_pose(obj_name)
                dist = np.linalg.norm(current_eef - obj_pos)
                if dist < min_dist_nontarget:
                    min_dist_nontarget = dist
                    closest_nontarget = obj_name
            except Exception:
                continue

        # Check distance to target object
        try:
            target_pos, _ = env.get_object_pose(target_name)
            dist_to_target = np.linalg.norm(current_eef - target_pos)
        except Exception:
            dist_to_target = float('inf')

        # Check gripper is closed (attempted grasp)
        gripper_closed = env.get_gripper_closed_norm()

        # Validation: EEF is closer to wrong object than target, gripper closed
        max_dist_to_wrong = self.config.get('max_dist_to_wrong_object', 0.12)
        near_wrong = min_dist_nontarget < max_dist_to_wrong
        gripper_ok = gripper_closed > 0.10
        closer_to_wrong = min_dist_nontarget < dist_to_target

        ok = near_wrong and gripper_ok and closer_to_wrong

        metrics = {
            'dist_to_nearest_nontarget': float(min_dist_nontarget),
            'closest_nontarget': closest_nontarget or "",
            'dist_to_target': float(dist_to_target),
            'gripper_closed_norm': float(gripper_closed),
            'closer_to_wrong': closer_to_wrong,
        }

        if ok:
            reason = (
                f"Wrong object confirmed: EEF near {closest_nontarget} "
                f"(dist={min_dist_nontarget:.3f}m), "
                f"gripper closed ({gripper_closed:.2f})"
            )
        else:
            parts = []
            if not near_wrong:
                parts.append(
                    f"EEF not near wrong object "
                    f"(dist={min_dist_nontarget:.3f}m > {max_dist_to_wrong})"
                )
            if not gripper_ok:
                parts.append(f"gripper not closed ({gripper_closed:.2f})")
            reason = "Wrong object not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
