#!/usr/bin/env python
"""
E10: trajectory_regression - Replay recent actions in reverse to regress progress.

Degrees: D0, D1
Phases: lift, transport
Injection: Reverse the trajectory by moving EEF back along recent path
Recovery: Detect regression, re-plan forward to recover lost progress
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class TrajectoryRegressionSkill(BaseErrorSkill):
    """
    Replay recent trajectory in reverse, causing the robot to regress
    to an earlier task phase (e.g., from transport back toward lift).
    """

    @property
    def name(self) -> str:
        return "trajectory_regression"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["lift", "transport"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when at least 10 frames of history are available
        and the gripper is holding an object.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Object must be above table and near EEF (indicating held)
        objects = frame_state.get('objects', {})
        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        min_height = 0.85

        held = False
        for obj_info in objects.values():
            obj_pos = np.array(obj_info['pos'])
            if obj_pos[2] >= min_height and np.linalg.norm(eef_pos - obj_pos) < 0.08:
                held = True
                break
        if not held:
            return []

        # At least 10 frames of history must be available
        frame_idx = trajectory_context.get('frame_index', 0)
        if frame_idx < 10:
            return []

        # Need remaining frames for recovery
        total_frames = trajectory_context.get('total_frames', 0)
        if total_frames - frame_idx < 20:
            return []

        # Need enough trajectory history before injection
        frame_idx = trajectory_context.get('frame_index', 0)
        if frame_idx < 10:
            return []

        injectable = ["D0"]
        if frame_idx >= 20 and total_frames - frame_idx >= 30:
            injectable.append("D1")

        return injectable

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Move EEF backward along the recent trajectory to regress task progress.

        D0: Move partway back (5-10 frames worth of regression).
        D1: Move significantly back (15-25 frames worth of regression).
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        obj_name = frame_state.get('target_object') or (next(iter(objects)) if objects else "unknown")

        try:
            pre_obj_pos, _ = env.get_object_pose(obj_name)
            pre_obj_pos = pre_obj_pos.copy()
        except Exception:
            pre_obj_pos = np.zeros(3)

        # Get the current task phase
        pre_phase = frame_state.get('task_phase', 'unknown')

        # Compute regression target: move EEF backward
        # The regression is implemented as a relative offset that reverses
        # the direction of recent progress
        # Read regression_steps from config: try regression_steps, then reverse_steps
        regression_steps_cfg = degree_params.get('regression_steps', None)
        if regression_steps_cfg is None:
            rs = degree_params.get('reverse_steps', None)
            if isinstance(rs, list) and len(rs) >= 2:
                regression_steps = int(rng.uniform(rs[0], rs[1]))
            elif rs is not None:
                regression_steps = int(rs)
            else:
                regression_steps = 30 if degree == "D0" else 50
        else:
            regression_steps = int(regression_steps_cfg)

        if degree == "D0":
            # Moderate regression
            regression_range = degree_params.get('regression_range', [0.04, 0.08])
            regression_mag = rng.uniform(regression_range[0], regression_range[1])
            # Reverse recent direction: move down and backward
            # In lift/transport, progress is typically up and forward
            pos_offset = np.zeros(3)
            pos_offset[2] = -regression_mag * 0.7  # Primarily downward
            # Add some lateral regression
            lateral = self.random_direction(rng, dims=2)
            pos_offset[:2] = lateral * regression_mag * 0.3

        else:  # D1
            # Aggressive regression
            regression_range = degree_params.get('regression_range', [0.08, 0.15])
            regression_mag = rng.uniform(regression_range[0], regression_range[1])
            pos_offset = np.zeros(3)
            pos_offset[2] = -regression_mag * 0.8  # Strong downward
            lateral = self.random_direction(rng, dims=2)
            pos_offset[:2] = lateral * regression_mag * 0.4

        # Execute regression movement (keep gripper closed)
        rot_offset = None
        if degree == "D1":
            rot_noise_cfg = degree_params.get('rotation_noise')
            if rot_noise_cfg and isinstance(rot_noise_cfg, list) and len(rot_noise_cfg) >= 2:
                rot_mag = rng.uniform(rot_noise_cfg[0], rot_noise_cfg[1])
                rot_axis = rng.randn(3)
                rot_axis /= max(np.linalg.norm(rot_axis), 1e-8)
                rot_offset = rot_axis * rot_mag

        kwargs = dict(pos_offset=pos_offset, max_steps=int(regression_steps), render_fn=render_fn)
        if rot_offset is not None:
            kwargs['rot_offset'] = rot_offset
        steps_taken = env.apply_eef_offset(**kwargs)

        # Hold in regressed position briefly to let physics settle
        self.settle_physics(env, 10, gripper='close', render_fn=render_fn)

        # Check post-regression task phase
        post_eef_pos = env.get_eef_pos().copy()

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name, "eef": True},
            params={
                "pos_offset": pos_offset.tolist(),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "post_eef_pos": post_eef_pos.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_phase": pre_phase,
                "regression_magnitude": float(regression_mag),
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
        Verify that the task phase regressed (e.g., from transport back to lift,
        or from lift to a lower position).
        """
        from error_benchmark.framework.core import ValidationResult_v5

        # Check EEF position regression
        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        eef_displacement = np.linalg.norm(current_eef - pre_eef)

        # EEF should have moved downward (Z decreased)
        z_regression = pre_eef[2] - current_eef[2]

        # Check if the held object also moved down
        objects = pre_state.get('objects', {})
        obj_z_regression = 0.0
        if objects:
            obj_name = pre_state.get('target_object') or next(iter(objects))
            if obj_name not in objects:
                obj_name = next(iter(objects))
            pre_obj_z = objects[obj_name]['pos'][2]
            try:
                current_obj_pos, _ = env.get_object_pose(obj_name)
                obj_z_regression = pre_obj_z - current_obj_pos[2]
            except Exception:
                pass

        # Try to get current task phase for comparison
        pre_phase = pre_state.get('task_phase', '')
        try:
            state_info = {'eef_pos': current_eef}
            post_phase = env.get_task_phase(state_info)
        except Exception:
            post_phase = "unknown"

        # Phase ordering for regression detection
        phase_order = {
            'pre_reach': 0, 'reach': 1, 'pre_grasp': 2,
            'grasp': 3, 'lift': 4, 'transport': 5, 'done': 6,
        }
        pre_idx = phase_order.get(pre_phase, -1)
        post_idx = phase_order.get(post_phase, -1)
        phase_regressed = post_idx < pre_idx if pre_idx >= 0 and post_idx >= 0 else False

        # Validation criteria:
        # 1. Significant downward movement (regression)
        # 2. Phase actually regressed, OR significant Z regression
        min_z_regression = 0.03  # At least 3cm downward
        z_ok = z_regression >= min_z_regression
        displacement_ok = eef_displacement > 0.02

        ok = (z_ok or phase_regressed) and displacement_ok

        metrics = {
            'eef_displacement': float(eef_displacement),
            'z_regression': float(z_regression),
            'obj_z_regression': float(obj_z_regression),
            'pre_phase': pre_phase,
            'post_phase': post_phase,
            'phase_regressed': phase_regressed,
        }

        if ok:
            parts = []
            if z_ok:
                parts.append(f"z_regression={z_regression:.3f}m")
            if phase_regressed:
                parts.append(f"phase: {pre_phase} -> {post_phase}")
            reason = "Trajectory regression confirmed: " + ", ".join(parts)
        else:
            parts = []
            if not z_ok:
                parts.append(f"z_regression={z_regression:.3f}m < {min_z_regression}")
            if not displacement_ok:
                parts.append(f"eef_displacement={eef_displacement:.3f}m too small")
            if not phase_regressed:
                parts.append(f"phase unchanged: {pre_phase} -> {post_phase}")
            reason = "Regression not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
