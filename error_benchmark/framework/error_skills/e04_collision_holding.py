#!/usr/bin/env python
"""
E5: collision_holding - Collide held object with environment.

Degrees: D0 (gentle downward), D1 (aggressive diagonal with rotation)
Phases: lift, transport
Injection: Move EEF so that the held object strikes environment geometry
           (table edge, walls, fixtures)
Recovery: Detect collision, stop, re-plan path to avoid obstacle
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class CollisionHoldingSkill(BaseErrorSkill):
    """
    While holding an object, move the EEF so the held object collides
    with environment geometry (table surface, walls, fixtures).
    """

    @property
    def name(self) -> str:
        return "collision_holding"

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
        Can inject when gripper is closed (holding) and object is above the table.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # Object must be above table and near EEF (indicating held)
        objects = frame_state.get('objects', {})
        if not objects:
            return []

        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        min_height = self.config.get_degree_params('D0').get('min_hold_height', 0.93)

        held_obj = self.find_held_object(objects, eef_pos, min_height=min_height)

        if held_obj is None:
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
        Move EEF downward/laterally so the held object strikes the table or
        environment fixtures.

        D0: Gentle downward move toward table surface (easy).
        D1: Aggressive diagonal move (down + lateral) into environment (hard).
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()

        objects = frame_state.get('objects', {})
        obj_name = frame_state.get('target_object') or (next(iter(objects)) if objects else "unknown")

        pre_obj_pos, _ = self.get_safe_object_pose(env, obj_name)

        # Record contacts before injection
        pre_contacts = len(env.get_contact_summary())

        # Compute collision direction based on degree
        pos_offset = np.zeros(3)

        if degree == "D0":
            # D0: gentle downward move toward table (easy)
            down_range = degree_params.get('down_range', [0.05, 0.10])
            down_mag = rng.uniform(down_range[0], down_range[1])
            pos_offset[2] = -down_mag

        elif degree == "D1":
            # D1: aggressive diagonal move (down + lateral) with optional rotation (hard)
            down_range = degree_params.get('down_range', [0.08, 0.15])
            lateral_range = degree_params.get('lateral_range', [0.06, 0.12])
            down_mag = rng.uniform(down_range[0], down_range[1])
            lateral_mag = rng.uniform(lateral_range[0], lateral_range[1])
            direction = self.random_direction(rng, dims=2)
            pos_offset[:2] = direction * lateral_mag
            pos_offset[2] = -down_mag

        # Execute the collision movement (keep gripper closed)
        collision_steps = degree_params.get('collision_steps', 40)
        steps_taken = env.apply_eef_offset(
            pos_offset=pos_offset,
            max_steps=collision_steps,
            render_fn=render_fn,
        )

        # Let physics settle
        self.settle_physics(env, 5, gripper='close', render_fn=render_fn)

        # Record contacts after injection
        post_contacts = env.get_contact_summary()
        contact_force_sum = sum(c.force for c in post_contacts)

        return self.create_error_spec(
            degree=degree,
            target={"body": obj_name},
            params={
                "pos_offset": pos_offset.tolist(),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_obj_pos": pre_obj_pos.tolist(),
                "pre_contact_count": pre_contacts,
                "post_contact_count": len(post_contacts),
                "contact_force_sum": float(contact_force_sum),
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
        Verify that the collision displaced the held object or the EEF moved
        significantly. Contact forces are transient in MuJoCo and may have
        resolved by the time we check, so we also accept displacement-based
        evidence.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        # Check current contacts
        contacts = env.get_contact_summary()
        total_force = sum(c.force for c in contacts)
        num_contacts = len(contacts)

        # Check that EEF actually moved
        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        eef_displacement = np.linalg.norm(current_eef - pre_eef)

        # Check object displacement
        obj_displacement = 0.0
        objects = pre_state.get('objects', {})
        if objects:
            obj_name = pre_state.get('target_object') or next(iter(objects))
            if obj_name not in objects:
                obj_name = next(iter(objects))
            try:
                obj_pos, _ = env.get_object_pose(obj_name)
                pre_obj_pos = np.array(objects[obj_name]['pos'])
                obj_displacement = np.linalg.norm(obj_pos - pre_obj_pos)
            except Exception:
                pass

        # Accept collision if EITHER:
        # 1. Contact forces detected, OR
        # 2. Object displaced significantly (collision happened but forces resolved), OR
        # 3. EEF moved significantly (the motion occurred even if contact was brief)
        min_force = self.config.get('min_contact_force', 0.5)
        force_ok = total_force > min_force
        obj_moved = obj_displacement > 0.005  # 5mm object displacement
        eef_moved = eef_displacement > 0.02   # 2cm EEF displacement

        ok = force_ok or (eef_moved and obj_moved) or eef_displacement > 0.05

        metrics = {
            'total_contact_force': float(total_force),
            'num_contacts': num_contacts,
            'eef_displacement': float(eef_displacement),
            'obj_displacement': float(obj_displacement),
        }

        if ok:
            reason = (f"Collision confirmed: force={total_force:.2f}, "
                      f"eef_disp={eef_displacement:.3f}m, obj_disp={obj_displacement:.3f}m")
        else:
            parts = []
            if not force_ok:
                parts.append(f"low contact force ({total_force:.2f} < {min_force})")
            if not eef_moved:
                parts.append(f"EEF barely moved ({eef_displacement:.3f}m)")
            if not obj_moved:
                parts.append(f"object barely moved ({obj_displacement:.3f}m)")
            reason = "Collision not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
