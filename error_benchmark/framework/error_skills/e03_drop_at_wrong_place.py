#!/usr/bin/env python
"""
E3: drop_at_wrong_place - Drop object at wrong location near target.

Degrees: D0 (small offset), D1 (large offset)
Phases: transport, place

Two injection modes (chosen automatically based on scene context):
  - "offset": Offset EEF away from target, then release.
  - "interaction": Move EEF above nearest non-target object, then release.

Recovery: Locate dropped object, re-grasp, navigate back to target.
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class DropAtWrongPlaceSkill(BaseErrorSkill):
    """
    Near target, drop object at wrong location.

    Two injection modes:
      - "offset": Offset EEF far away then drop (object lands alone).
      - "interaction": Move EEF above a non-target object then drop
        (object falls onto another object).

    Mode is chosen randomly when non-target objects exist; offset-only
    when no non-target objects are available.

    D0 = small displacement, D1 = large displacement.
    """

    @property
    def name(self) -> str:
        return "drop_at_wrong_place"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["transport", "place"]

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when gripper holds object AND EEF is near target (<10cm).
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

        # Find held object
        held_obj = self.find_held_object(objects, eef_pos, min_height=min_height)

        if held_obj is None:
            return []

        # EEF must be near target
        target_pos = trajectory_context.get('target_pos')
        if target_pos is not None:
            eef_to_target = np.linalg.norm(eef_pos[:2] - np.array(target_pos)[:2])
            if eef_to_target >= target_dist_thresh:
                return []  # Too far from target — use drop_in_transit

        return self.check_remaining_frames(trajectory_context)

    def _inject_offset(
        self,
        env: 'EnvWrapper',
        degree_params: Dict[str, Any],
        obj_name: str,
        rng: np.random.RandomState,
        render_fn,
    ) -> Dict[str, Any]:
        """Offset mode: move EEF away from target, apply lateral offset + drop velocity."""
        from error_benchmark.framework.utils.math_utils import random_translation

        eef_offset_range = degree_params.get('eef_offset_range', [0.05, 0.10])
        magnitude = rng.uniform(eef_offset_range[0], eef_offset_range[1])
        pos_offset = random_translation(rng, magnitude)
        env.apply_eef_offset(pos_offset=pos_offset, max_steps=20, render_fn=render_fn)

        sim = env._env.sim

        # Open gripper joints
        self.open_gripper_joints_direct(sim)

        # Offset object from EEF and apply drop velocity
        jnt_id, qpos_addr, qvel_addr = self.find_object_joint(env, obj_name)
        if qpos_addr is not None:
            lateral_mag = rng.uniform(0.05, 0.08)
            lateral_dir = rng.randn(2)
            lateral_dir /= max(np.linalg.norm(lateral_dir), 1e-8)
            sim.data.qpos[qpos_addr] += lateral_dir[0] * lateral_mag
            sim.data.qpos[qpos_addr + 1] += lateral_dir[1] * lateral_mag

            drop_speed = rng.uniform(0.5, 1.5)
            sim.data.qvel[qvel_addr + 2] = -drop_speed

        sim.forward()

        # Bare physics steps to let object separate from gripper
        import mujoco
        for _ in range(15):
            mujoco.mj_step(sim.model._model, sim.data._data)
            if render_fn is not None:
                render_fn()
        sim.forward()

        return {"pos_offset": pos_offset.tolist()}

    def _inject_interaction(
        self,
        env: 'EnvWrapper',
        degree_params: Dict[str, Any],
        obj_name: str,
        objects: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn,
    ) -> Dict[str, Any]:
        """Interaction mode: move EEF above nearest non-target object, then drop."""
        pre_eef_pos = env.get_eef_pos().copy()

        # Find nearest non-target object
        nontarget_names = [n for n in objects if n != obj_name]
        nearest_nt = None
        nearest_dist = float('inf')
        nearest_nt_pos = None
        for nt_name in nontarget_names:
            try:
                nt_pos, _ = env.get_object_pose(nt_name)
                dist = np.linalg.norm(pre_eef_pos[:2] - nt_pos[:2])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_nt = nt_name
                    nearest_nt_pos = nt_pos.copy()
            except Exception:
                continue

        # Compute target position: directly above the non-target object
        drop_height_range = degree_params.get('drop_height_above', [0.05, 0.10])
        drop_height = rng.uniform(drop_height_range[0], drop_height_range[1])

        if nearest_nt_pos is not None:
            target_above = nearest_nt_pos.copy()
            target_above[2] = nearest_nt_pos[2] + drop_height
            target_above[2] = max(target_above[2], 0.90)
        else:
            target_above = pre_eef_pos.copy()
            target_above[2] = max(pre_eef_pos[2], 0.90)

        # Move EEF (with held object) to above non-target
        approach_steps = degree_params.get('approach_steps', 60)
        steps_to_approach = env.move_eef_to(
            target_pos=target_above,
            max_steps=int(approach_steps),
            render_fn=render_fn,
        )

        sim = env._env.sim

        # Open gripper joints
        self.open_gripper_joints_direct(sim)

        sim.forward()

        # Bare physics steps to let object separate from gripper
        import mujoco
        for _ in range(30):
            mujoco.mj_step(sim.model._model, sim.data._data)
            if render_fn is not None:
                render_fn()
        sim.forward()

        return {
            "nontarget_object": nearest_nt or "",
            "nontarget_pos": nearest_nt_pos.tolist() if nearest_nt_pos is not None else [],
            "drop_height": drop_height,
            "steps_to_approach": steps_to_approach,
        }

    def inject(
        self,
        env: 'EnvWrapper',
        degree: str,
        frame_state: Dict[str, Any],
        rng: np.random.RandomState,
        render_fn=None,
    ) -> 'ErrorSpec':
        """
        Drop object at wrong location. Chooses between offset and interaction
        modes based on scene context.
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        obj_name = frame_state.get('target_object') or (next(iter(objects)) if objects else "unknown")

        pre_obj_pos, pre_obj_quat = self.get_safe_object_pose(env, obj_name)

        # Choose injection mode: interaction if non-target objects exist (50% chance)
        nontarget_names = [n for n in objects if n != obj_name]
        if nontarget_names and rng.random() < 0.5:
            injection_mode = "interaction"
        else:
            injection_mode = "offset"

        # Execute chosen mode
        if injection_mode == "interaction":
            mode_params = self._inject_interaction(
                env, degree_params, obj_name, objects, rng, render_fn,
            )
        else:
            mode_params = self._inject_offset(
                env, degree_params, obj_name, rng, render_fn,
            )

        # Settle with open gripper
        settle_steps = degree_params.get('settle_steps', 40)
        self.settle_physics(env, settle_steps, gripper='open', render_fn=render_fn)

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name},
            params={
                "injection_mode": injection_mode,
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_obj_quat": pre_obj_quat.tolist(),
                "object_name": obj_name,
                **mode_params,
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
        Verify: object dropped (dz >= 3cm).
        Object interaction is recorded as a metric but does not affect validation.
        """
        from error_benchmark.framework.core import ValidationResult_v5
        from error_benchmark.framework.utils.math_utils import quat_multiply, quat_conjugate

        objects = pre_state.get('objects', {})
        if not objects:
            return ValidationResult_v5(
                ok=False, error_name=self.name, degree=degree,
                reason="No objects in pre_state",
            )

        obj_name = pre_state.get('target_object') or next(iter(objects))
        if obj_name not in objects:
            obj_name = next(iter(objects))
        pre_z = objects[obj_name]['pos'][2]
        pre_quat = np.array(objects[obj_name].get('quat', [1, 0, 0, 0]))

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

        # Record interaction info as metric (not used for validation)
        contacts = env.get_contact_summary()
        has_obj_interaction, interacting_geom = self.has_non_ground_contact(contacts, obj_name)

        # Also check historical contacts from post-rollout stats
        if not has_obj_interaction and post_stats and post_stats.obj_contact_geoms:
            has_obj_interaction = True
            interacting_geom = post_stats.obj_contact_geoms[0]

        # Orientation change (logged for metrics)
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
                'has_obj_interaction': has_obj_interaction,
                'interacting_geom': interacting_geom,
                'orientation_change_rad': float(angle_change),
                'historical_contact_geoms': post_stats.obj_contact_geoms if post_stats else [],
            },
            reason=(f"Drop at wrong place confirmed: dz={delta_z:.3f}m, "
                    f"interaction={has_obj_interaction}" if ok
                    else f"Drop at wrong place failed: dz={delta_z:.3f}m, "
                    f"drop_ok={drop_ok}"),
            post_rollout_stats=post_stats,
        )
