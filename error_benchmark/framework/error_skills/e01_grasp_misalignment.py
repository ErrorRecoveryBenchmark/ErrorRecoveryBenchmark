#!/usr/bin/env python
"""
E1: grasp_misalignment - Close gripper with misaligned EEF pose.

Degrees: D0 (lateral offset), D1 (larger offset + rotation)
Phases: pre_grasp, grasp
Injection: Offset/rotate EEF from ideal grasp pose, then close gripper
Recovery: Detect poor grasp, re-open gripper, re-approach, re-grasp
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class GraspMisalignmentSkill(BaseErrorSkill):
    """
    Close the gripper while the EEF is misaligned relative to the object,
    resulting in a poor or unstable grasp.
    """

    @property
    def name(self) -> str:
        return "grasp_misalignment"

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

        # EEF must be near an object (within 0.1m)
        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        objects = frame_state.get('objects', {})
        if not objects:
            return []

        near_object = False
        for obj_name, obj_info in objects.items():
            obj_pos = np.array(obj_info.get('pos', [0, 0, 0]))
            dist = np.linalg.norm(eef_pos[:2] - obj_pos[:2])  # XY distance
            if dist < 0.1:
                near_object = True
                break

        if not near_object:
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
        Offset and/or rotate EEF from ideal grasp pose, then close gripper.
        D0: small lateral offset then close (easy).
        D1: larger lateral offset + rotation then close (hard).
        """
        from error_benchmark.framework.utils.math_utils import random_translation

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

        pos_offset = np.zeros(3)
        rot_offset = np.zeros(3)

        if degree == "D0":
            # D0: small lateral offset only (easy)
            offset_range = degree_params.get('lateral_offset', degree_params.get('lateral_offset_range', [0.005, 0.04]))
            magnitude = rng.uniform(offset_range[0], offset_range[1])
            direction = self.random_direction(rng, dims=2)
            pos_offset[:2] = direction * magnitude
            env.apply_eef_offset(pos_offset=pos_offset, max_steps=30, render_fn=render_fn)

        if degree == "D1":
            # D1: larger lateral offset + rotation (hard)
            offset_range = degree_params.get('lateral_offset', degree_params.get('lateral_offset_range', [0.005, 0.04]))
            magnitude = rng.uniform(offset_range[0], offset_range[1])
            direction = self.random_direction(rng, dims=2)
            pos_offset[:2] = direction * magnitude
            env.apply_eef_offset(pos_offset=pos_offset, max_steps=30, render_fn=render_fn)

            rotation_range = degree_params.get('rotation_offset', degree_params.get('rotation_range', [0.08, 0.5]))
            max_angle = rng.uniform(rotation_range[0], rotation_range[1])
            rot_offset = self.random_direction(rng, dims=3) * max_angle
            env.apply_eef_offset(
                pos_offset=np.zeros(3),
                rot_offset=rot_offset,
                max_steps=30,
                render_fn=render_fn,
            )

        # Close gripper to attempt grasp with misalignment
        gripper_close_steps = degree_params.get('gripper_close_steps', 30)
        env.set_gripper_state(open_fraction=0.0, steps=gripper_close_steps, render_fn=render_fn)

        # Let physics settle
        self.settle_physics(env, 25, gripper='close', render_fn=render_fn)

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name},
            params={
                "pos_offset": pos_offset.tolist(),
                "rot_offset": rot_offset.tolist(),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_eef_quat": pre_eef_quat.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_obj_quat": pre_obj_quat.tolist(),
                "gripper_close_steps": gripper_close_steps,
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
        Verify that the gripper closed but the object is offset from ideal grasp pose.
        A successful misalignment means gripper is closed AND the grasp is imperfect
        (object position deviates from where it would be in a clean grasp).
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

        obj_name = pre_state.get('target_object') or next(iter(objects))

        # Check gripper is closed (attempted grasp)
        gripper_closed = env.get_gripper_closed_norm()
        gripper_ok = gripper_closed > 0.10

        # Check EEF displacement from pre-injection position
        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        eef_displacement = np.linalg.norm(current_eef - pre_eef)

        # D0: check lateral offset only
        pos_ok = True
        if degree == "D0":
            min_displacement = 0.01  # At least 1cm lateral offset
            pos_ok = eef_displacement >= min_displacement

        # D1: check both lateral offset and rotation deviation
        rot_ok = True
        rot_dist = 0.0
        if degree == "D1":
            min_displacement = 0.01  # At least 1cm lateral offset
            pos_ok = eef_displacement >= min_displacement
            current_quat = env.get_eef_quat()
            pre_quat = np.array(pre_state.get('eef_quat', [1, 0, 0, 0]))
            dot = abs(np.dot(current_quat, pre_quat))
            rot_dist = 2 * np.arccos(np.clip(dot, -1, 1))
            min_rotation = 0.08  # ~4.6 degrees
            rot_ok = rot_dist >= min_rotation

        ok = gripper_ok and pos_ok and rot_ok

        metrics = {
            'gripper_closed_norm': float(gripper_closed),
            'eef_displacement': float(eef_displacement),
        }
        if degree == "D1":
            metrics['rotation_distance'] = float(rot_dist)

        if ok:
            reason = f"Grasp misalignment confirmed: displacement={eef_displacement:.3f}m"
        else:
            parts = []
            if not gripper_ok:
                parts.append(f"gripper not closed ({gripper_closed:.2f})")
            if not pos_ok:
                parts.append(f"insufficient offset ({eef_displacement:.3f}m)")
            if not rot_ok:
                parts.append(f"insufficient rotation ({rot_dist:.3f} rad)")
            reason = "Misalignment not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
