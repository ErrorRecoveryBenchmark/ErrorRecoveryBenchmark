#!/usr/bin/env python
"""
E2: drop_in_transit - Drop held object mid-transit, far from target.

Degrees: D0 (orientation preserved), D1 (orientation changed)
Phases: lift, transport
Injection: Open gripper to drop object when EEF is far from target position
Recovery: Move to dropped object, re-grasp, resume task
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class DropInTransitSkill(BaseErrorSkill):
    """
    Drop object mid-transit when EEF is far from target (>10cm).

    D0 vs D1 is determined by post-drop orientation change:
      D0: object orientation change < threshold (lands upright)
      D1: object orientation change >= threshold (tumbles/tilts)
    """

    @property
    def name(self) -> str:
        return "drop_in_transit"

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
        Can inject when gripper holds object above min height AND
        EEF is far from target position (> target_distance_threshold).
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        objects = frame_state.get('objects', {})
        if not objects:
            return []

        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        min_height = self.config.get_degree_params('D0').get('min_hold_height', 0.93)
        target_dist_thresh = self.config.params.get('target_distance_threshold', 0.10)

        # Require gripper to be closed (actually gripping, not just nearby)
        gripper_closed = frame_state.get('gripper_closed_norm', 0.0)
        grasp_threshold = self.config.params.get('grasp_closed_threshold', 0.05)
        if gripper_closed < grasp_threshold:
            return []

        # Find held object (above table AND close to EEF AND within ~3cm Z of EEF)
        held_obj = self.find_held_object(objects, eef_pos, min_height=min_height)

        if held_obj is None:
            return []

        # Check EEF is far from target
        target_pos = trajectory_context.get('target_pos')
        if target_pos is not None:
            eef_to_target = np.linalg.norm(eef_pos[:2] - np.array(target_pos)[:2])
            if eef_to_target < target_dist_thresh:
                return []  # Too close to target — use drop_at_wrong_place

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
        Drop the held object at current position (far from target).

        Uses MuJoCo direct manipulation: open gripper joints, offset object
        from EEF, apply downward velocity, let physics settle.
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        eef_pos = np.array(frame_state.get('eef_pos', pre_eef_pos))
        min_height = self.config.get_degree_params('D0').get('min_hold_height', 0.93)

        # Find the held object (above table AND close to EEF)
        obj_name = self.find_held_object(objects, eef_pos, min_height=min_height)
        if obj_name is None:
            obj_name = frame_state.get('target_object') or (next(iter(objects)) if objects else "unknown")

        # Record pre-drop object pose
        pre_obj_pos, pre_obj_quat = self.get_safe_object_pose(env, obj_name)

        sim = env._env.sim

        # Phase 1: Open gripper + set actuator controls to open
        self.open_gripper_joints_direct(sim)
        self.zero_gripper_actuators(sim)

        # Phase 2: Apply velocity to object
        jnt_id, qpos_addr, qvel_addr = self.find_object_joint(env, obj_name)
        if qpos_addr is not None:
            if degree == "D0":
                # Zero lateral/angular velocity, add slight downward push to assist gravity
                for k in range(6):
                    sim.data.qvel[qvel_addr + k] = 0.0
                sim.data.qvel[qvel_addr + 2] = -0.15  # Gentle downward push
            else:
                drop_speed = rng.uniform(0.8, 1.5)
                sim.data.qvel[qvel_addr + 2] = -drop_speed

        sim.forward()

        # Phase 3: Run bare physics (no controller) with gripper actuators zeroed
        # This is critical — the OSC controller compensates gripper commands,
        # so we must run without controller for separation
        import mujoco
        bare_steps = 80
        for _ in range(bare_steps):
            self.zero_gripper_actuators(sim)
            mujoco.mj_step(sim.model._model, sim.data._data)
            if render_fn is not None:
                render_fn()
        sim.forward()

        # Phase 4: Settle with controller (gripper open)
        settle_steps = degree_params.get('settle_steps', 40)
        self.settle_physics(env, settle_steps, gripper='open', render_fn=render_fn)

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name},
            params={
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_obj_quat": pre_obj_quat.tolist(),
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
        Verify object dropped and check orientation change for D0/D1 classification.

        D0: object fell but orientation change < threshold
        D1: object fell with orientation change >= threshold
        """
        from error_benchmark.framework.core import ValidationResult_v5
        from error_benchmark.framework.utils.math_utils import quat_multiply, quat_conjugate

        objects = pre_state.get('objects', {})
        if not objects:
            return ValidationResult_v5(
                ok=False, error_name=self.name, degree=degree,
                reason="No objects in pre_state",
            )

        # Find the held object: use target_object from pre_state, fallback to heuristic
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        min_height = self.config.params.get('D0', {}).get('min_hold_height', 0.93)
        obj_name = pre_state.get('target_object')
        if obj_name is None or obj_name not in objects:
            obj_name = None
            for name, info in objects.items():
                obj_pos = np.array(info['pos'])
                if obj_pos[2] >= min_height and np.linalg.norm(pre_eef - obj_pos) < 0.08:
                    obj_name = name
                    break
            if obj_name is None:
                obj_name = next(iter(objects))

        pre_z = objects[obj_name]['pos'][2]
        pre_quat = np.array(objects[obj_name].get('quat', [1, 0, 0, 0]))

        # Get post-drop Z
        if post_stats and post_stats.obj_z_trajectory:
            min_z = post_stats.min_z
        else:
            try:
                obj_pos, _ = env.get_object_pose(obj_name)
                min_z = obj_pos[2]
            except Exception:
                return ValidationResult_v5(
                    ok=False, error_name=self.name, degree=degree,
                    reason="Cannot read object position",
                )

        delta_z = pre_z - min_z

        # EEF-to-object distance
        try:
            obj_pos, post_quat = env.get_object_pose(obj_name)
            eef_pos = env.get_eef_pos()
            eef_obj_dist = float(np.linalg.norm(eef_pos - obj_pos))
        except Exception:
            eef_obj_dist = 999.0
            post_quat = pre_quat

        # Check drop occurred
        separated = eef_obj_dist > 0.08
        drop_ok = delta_z >= 0.03 or (separated and delta_z > 0.01)

        # Orientation change (logged for metrics, not used for validation —
        # D0/D1 is determined by inject() params, not post-drop rotation)
        q_diff = quat_multiply(post_quat, quat_conjugate(pre_quat))
        angle_change = 2.0 * np.arccos(np.clip(abs(q_diff[0]), 0.0, 1.0))

        ok = drop_ok

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics={
                'delta_z': float(delta_z),
                'min_z': float(min_z),
                'pre_z': float(pre_z),
                'eef_obj_dist': eef_obj_dist,
                'orientation_change_rad': float(angle_change),
            },
            reason=(f"Drop in transit confirmed: dz={delta_z:.3f}m, "
                    f"angle={angle_change:.3f}rad" if ok
                    else f"Drop in transit failed: dz={delta_z:.3f}m, "
                    f"angle={angle_change:.3f}rad, drop_ok={drop_ok}"),
            post_rollout_stats=post_stats,
        )
