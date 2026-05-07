#!/usr/bin/env python
"""
E12: grasp_wrong_pose - Rotate EEF significantly then close gripper.

Degrees: D0 (small rotation), D1 (large rotation)
Phases: pre_grasp, grasp
Injection: Apply large rotation to EEF orientation, then close gripper
Recovery: Detect bad orientation, open gripper, correct rotation, re-grasp

This is a rotation-only variant of grasp misalignment (E1).
While E1-D1 uses moderate rotation, E10 uses significantly larger
rotations that make the grasp clearly wrong-oriented.
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class GraspWrongPoseSkill(BaseErrorSkill):
    """
    Rotate the EEF significantly from its ideal orientation before closing
    the gripper, resulting in a clearly wrong grasp pose.
    """

    @property
    def name(self) -> str:
        return "grasp_wrong_pose"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["pre_grasp", "grasp"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when gripper is open and EEF is near an object.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Gripper must be open (not yet closed)
        gripper_closed = frame_state.get('gripper_closed_norm', 0.0)
        if gripper_closed >= self.GRIPPER_CLOSED_THRESHOLD:
            return []

        # EEF must be near an object
        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        objects = frame_state.get('objects', {})
        if not objects:
            return []

        near_object = False
        for obj_name, obj_info in objects.items():
            obj_pos = np.array(obj_info.get('pos', [0, 0, 0]))
            dist = np.linalg.norm(eef_pos - obj_pos)
            if dist < 0.1:
                near_object = True
                break

        if not near_object:
            return []

        # Need enough remaining frames for recovery
        return self.check_remaining_frames(trajectory_context, min_frames=25, d1_min_frames=30)

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Apply a large rotation to EEF orientation, then close gripper.
        The rotation is significantly larger than E1-D1 to create a
        clearly wrong grasp pose.
        """
        from error_benchmark.framework.utils.math_utils import (
            random_rotation_quat, quat_multiply
        )

        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()
        pre_eef_quat = env.get_eef_quat().copy()

        objects = frame_state.get('objects', {})
        obj_name = frame_state.get('target_object') or (next(iter(objects)) if objects else "unknown")

        # Record pre-injection object pose
        try:
            pre_obj_pos, pre_obj_quat = env.get_object_pose(obj_name)
            pre_obj_pos = pre_obj_pos.copy()
            pre_obj_quat = pre_obj_quat.copy()
        except Exception:
            pre_obj_pos = np.zeros(3)
            pre_obj_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # D0: small rotation [0.1, 0.25] radians (~6-14 degrees)
        # D1: large rotation [0.3, 1.0] radians (~17-57 degrees)
        default_range = [0.1, 0.25] if degree == "D0" else [0.3, 1.0]
        rotation_range = degree_params.get('rotation_offset', degree_params.get('rotation_range', default_range))
        max_angle = rng.uniform(rotation_range[0], rotation_range[1])

        # Generate rotation as Euler angles
        rot_offset = self.random_direction(rng, dims=3) * max_angle

        # Apply rotation offset
        rotation_steps = degree_params.get('rotation_steps', 40)
        steps_taken = env.apply_eef_offset(
            pos_offset=np.zeros(3),
            rot_offset=rot_offset,
            max_steps=int(rotation_steps),
            render_fn=render_fn,
        )

        # Close gripper with wrong orientation
        gripper_close_steps = degree_params.get('gripper_close_steps', 30)
        env.set_gripper_state(open_fraction=0.0, steps=gripper_close_steps, render_fn=render_fn)

        # Let physics settle
        self.settle_physics(env, 20, gripper='close', render_fn=render_fn)

        post_eef_quat = env.get_eef_quat().copy()

        # Compute actual rotation applied
        dot = abs(np.dot(post_eef_quat, pre_eef_quat))
        actual_rot_dist = 2 * np.arccos(np.clip(dot, -1, 1))

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name},
            params={
                "rot_offset": rot_offset.tolist(),
                "target_angle": float(max_angle),
                "actual_rotation": float(actual_rot_dist),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_eef_quat": pre_eef_quat.tolist(),
                "post_eef_quat": post_eef_quat.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_obj_quat": pre_obj_quat.tolist(),
                "gripper_close_steps": gripper_close_steps,
                "steps_taken": steps_taken,
                "object_name": obj_name,
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
        Verify that the gripper closed with a large rotation deviation
        from the ideal grasp orientation.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        # Check gripper is closed
        gripper_closed = env.get_gripper_closed_norm()
        gripper_ok = gripper_closed > 0.10

        # Check rotation deviation from pre-injection orientation
        current_quat = env.get_eef_quat()
        pre_quat = np.array(pre_state.get('eef_quat', [1, 0, 0, 0]))
        dot = abs(np.dot(current_quat, pre_quat))
        rot_dist = 2 * np.arccos(np.clip(dot, -1, 1))

        # D0: lower threshold for small rotation, D1: higher threshold for large rotation
        degree_params = self.config.get_degree_params(degree)
        default_min_rot = 0.08 if degree == "D0" else 0.15
        min_rotation = degree_params.get('min_rotation_deviation', default_min_rot)
        rot_ok = rot_dist >= min_rotation

        ok = gripper_ok and rot_ok

        metrics = {
            'gripper_closed_norm': float(gripper_closed),
            'rotation_distance': float(rot_dist),
            'rotation_degrees': float(np.degrees(rot_dist)),
            'min_rotation_threshold': float(min_rotation),
        }

        if ok:
            reason = (
                f"Wrong pose grasp confirmed: rotation={np.degrees(rot_dist):.1f} deg, "
                f"gripper closed ({gripper_closed:.2f})"
            )
        else:
            parts = []
            if not gripper_ok:
                parts.append(f"gripper not closed ({gripper_closed:.2f})")
            if not rot_ok:
                parts.append(
                    f"rotation too small ({np.degrees(rot_dist):.1f} deg "
                    f"< {np.degrees(min_rotation):.1f} deg)"
                )
            reason = "Wrong pose not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
