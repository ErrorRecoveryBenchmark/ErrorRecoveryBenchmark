#!/usr/bin/env python
"""
E8: collision_self - Move EEF toward robot body to approach self-collision.

Degrees: D0, D1
Phases: reach, lift, transport
Injection: Command EEF toward robot base/links to approach joint limits
           or trigger self-contact detection
Recovery: Detect joint limit / self-collision, backtrack to safe configuration
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .base_skill import BaseErrorSkill, SkillConfig

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import ErrorSpec, PostRolloutStats, ValidationResult_v5


class CollisionSelfSkill(BaseErrorSkill):
    """
    Move the EEF toward the robot's own body (base/links) to approach
    joint limits or trigger self-contact conditions.
    """

    @property
    def name(self) -> str:
        return "collision_self"

    @property
    def valid_degrees(self) -> List[str]:
        return ["D0", "D1"]

    @property
    def valid_phases(self) -> List[str]:
        return ["reach", "lift", "transport"]

    def _compute_joint_limit_proximity(self, env: 'EnvWrapper') -> float:
        """
        Compute how close any joint is to its limits.
        Returns the maximum proximity ratio (0 = center, 1 = at limit).
        """
        qpos = env.get_robot_qpos()
        joint_ranges = env.get_joint_ranges()

        max_proximity = 0.0
        # Only check the first N robot joints (skip gripper/object joints)
        n_robot_joints = min(7, len(joint_ranges))  # 7-DOF arm typical

        for i in range(n_robot_joints):
            lo, hi = joint_ranges[i]
            if hi - lo < 1e-8:
                continue  # Skip fixed joints
            # Normalize position to [0, 1] within joint range
            pos_norm = (qpos[i] - lo) / (hi - lo)
            # Proximity to nearest limit
            proximity = max(pos_norm, 1.0 - pos_norm)
            # proximity > 0.5 means in outer half of range
            # Map to [0, 1] where 1 = at limit
            limit_ratio = max(0.0, (proximity - 0.5) * 2.0)
            max_proximity = max(max_proximity, limit_ratio)

        return max_proximity

    def can_inject(
        self,
        frame_state: Dict[str, Any],
        trajectory_context: Dict[str, Any],
    ) -> List[str]:
        """
        Can inject when EEF is not already at extreme positions.
        """
        phase = trajectory_context.get('task_phase', frame_state.get('task_phase', ''))

        if not self.supports_phase(phase):
            return []

        # EEF should not already be at extreme positions
        eef_pos = np.array(frame_state.get('eef_pos', [0, 0, 0]))
        # Typical robot base is around (0, 0, ~0.9) in robosuite
        eef_z = eef_pos[2]
        if eef_z < 0.75:  # Already very low, risky
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
        Move EEF toward the robot base/links to approach self-collision.

        D0: Moderate retraction toward robot base.
        D1: Aggressive retraction with downward component toward base.
        """
        degree_params = self.config.get_degree_params(degree)
        pre_eef_pos = env.get_eef_pos().copy()
        pre_qpos = env.get_robot_qpos().copy()

        # Compute joint limit proximity before injection
        pre_joint_proximity = self._compute_joint_limit_proximity(env)

        # Robot base is approximately at origin in XY, around Z=0.9
        # Move EEF toward base: reduce XY distance and lower Z
        robot_base_xy = np.array([0.0, 0.0])  # Approximate base XY
        eef_xy = pre_eef_pos[:2]

        # Direction: toward robot base
        direction_xy = robot_base_xy - eef_xy
        norm_xy = np.linalg.norm(direction_xy)
        if norm_xy > 1e-8:
            direction_xy /= norm_xy

        pos_offset = np.zeros(3)

        if degree == "D0":
            retract_range = degree_params.get('retract_range', [0.05, 0.10])
            retract_mag = rng.uniform(retract_range[0], retract_range[1])
            pos_offset[:2] = direction_xy * retract_mag
            # Slight downward to stress joints
            pos_offset[2] = rng.uniform(-0.03, 0.0)

        elif degree == "D1":
            retract_range = degree_params.get('retract_range', [0.08, 0.15])
            retract_mag = rng.uniform(retract_range[0], retract_range[1])
            pos_offset[:2] = direction_xy * retract_mag
            # More aggressive downward component
            down_range = degree_params.get('down_range', [0.03, 0.08])
            pos_offset[2] = -rng.uniform(down_range[0], down_range[1])

        # Execute the movement
        max_steps = degree_params.get('max_steps', 40)
        steps_taken = env.apply_eef_offset(
            pos_offset=pos_offset,
            max_steps=int(max_steps),
            render_fn=render_fn,
        )

        # Let physics settle
        for _ in range(5):
            action = env.get_neutral_action()
            env.step(action)
            if render_fn is not None:
                render_fn()

        # Compute joint limit proximity after injection
        post_joint_proximity = self._compute_joint_limit_proximity(env)
        post_qpos = env.get_robot_qpos()

        # Check for self-contacts in current contact list
        contacts = env.get_contact_summary()
        robot_self_contacts = []
        for c in contacts:
            # Self-collision: both geoms belong to the robot
            g1, g2 = c.geom1 or "", c.geom2 or ""
            if ('robot' in g1.lower() or 'gripper' in g1.lower()) and \
               ('robot' in g2.lower() or 'gripper' in g2.lower()):
                robot_self_contacts.append(c)

        return self.create_error_spec(
            degree=degree,
            target={"eef": True, "robot_base": True},
            params={
                "pos_offset": pos_offset.tolist(),
                "pre_eef_pos": pre_eef_pos.tolist(),
                "pre_joint_proximity": float(pre_joint_proximity),
                "post_joint_proximity": float(post_joint_proximity),
                "self_contact_count": len(robot_self_contacts),
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
        Verify that joint limits were approached or self-contact occurred.
        """
        from error_benchmark.framework.core import ValidationResult_v5

        # Check joint limit proximity
        post_joint_proximity = self._compute_joint_limit_proximity(env)

        # Check for self-contacts
        contacts = env.get_contact_summary()
        robot_self_contacts = 0
        for c in contacts:
            g1, g2 = c.geom1 or "", c.geom2 or ""
            if ('robot' in g1.lower() or 'gripper' in g1.lower()) and \
               ('robot' in g2.lower() or 'gripper' in g2.lower()):
                robot_self_contacts += 1

        # Check EEF displacement toward base
        current_eef = env.get_eef_pos()
        pre_eef = np.array(pre_state.get('eef_pos', [0, 0, 0]))
        eef_displacement = np.linalg.norm(current_eef - pre_eef)

        # Validation criteria: joint limit approached OR self-contact detected
        min_proximity = self.config.get('min_joint_limit_proximity', 0.6)
        joint_limit_ok = post_joint_proximity >= min_proximity
        self_contact_ok = robot_self_contacts > 0
        displacement_ok = eef_displacement > 0.02

        ok = (joint_limit_ok or self_contact_ok) and displacement_ok

        metrics = {
            'joint_limit_proximity': float(post_joint_proximity),
            'robot_self_contacts': robot_self_contacts,
            'eef_displacement': float(eef_displacement),
        }

        if ok:
            parts = []
            if joint_limit_ok:
                parts.append(f"joint_proximity={post_joint_proximity:.2f}")
            if self_contact_ok:
                parts.append(f"self_contacts={robot_self_contacts}")
            reason = "Self-collision condition confirmed: " + ", ".join(parts)
        else:
            parts = []
            if not joint_limit_ok and not self_contact_ok:
                parts.append(
                    f"joint_proximity={post_joint_proximity:.2f} "
                    f"(need >={min_proximity}), no self-contacts"
                )
            if not displacement_ok:
                parts.append(f"EEF barely moved ({eef_displacement:.3f}m)")
            reason = "Self-collision not confirmed: " + ", ".join(parts)

        return ValidationResult_v5(
            ok=ok,
            error_name=self.name,
            degree=degree,
            metrics=metrics,
            reason=reason,
            post_rollout_stats=post_stats,
        )
