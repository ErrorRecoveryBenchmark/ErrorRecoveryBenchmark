#!/usr/bin/env python
"""
Rollout Utilities - Post-injection trajectory tracking

Extracted from v4 DropValidator's collect_rollout_stats() for reuse.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import PostRolloutStats


def collect_rollout_stats(
    env: 'EnvWrapper',
    obj_name: str,
    steps: int,
    pre_state_info: Dict,
    step_callback=None,
) -> 'PostRolloutStats':
    """
    Execute K steps of neutral action and collect trajectory statistics.

    Used to verify that an injection actually caused a meaningful error
    (object dropped, displaced, tipped over, etc.).

    Args:
        env: EnvWrapper instance
        obj_name: Object name to track
        steps: Number of steps to collect
        pre_state_info: Pre-injection state info dict
        step_callback: Optional callback(t) called each step for rendering

    Returns:
        PostRolloutStats with trajectory data
    """
    from error_benchmark.framework.core import PostRolloutStats

    stats = PostRolloutStats()
    stats.total_steps = steps

    initial_z = pre_state_info['objects'][obj_name]['pos'][2]
    stats.min_z = initial_z
    stats.max_z = initial_z

    z_trajectory = [initial_z]
    pos_trajectory = [pre_state_info['objects'][obj_name]['pos'].copy()]
    quat_trajectory = [pre_state_info['objects'][obj_name]['quat'].copy()]

    up_vector = np.array([0, 0, 1])
    max_offset = 0.0
    contact_geoms_seen = set()

    for t in range(steps):
        action = env.get_neutral_action()
        env.step(action)

        pos, quat = env.get_object_pose(obj_name)

        current_z = pos[2]
        z_trajectory.append(current_z)
        stats.min_z = min(stats.min_z, current_z)
        stats.max_z = max(stats.max_z, current_z)

        pos_trajectory.append(pos.copy())
        quat_trajectory.append(quat.copy())

        # Tip angle
        try:
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            obj_z_axis = rot.as_matrix()[:, 2]
            cos_angle = np.dot(obj_z_axis, up_vector)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            stats.max_tip_angle = max(stats.max_tip_angle, angle)
        except Exception:
            pass

        # XY offset from initial position
        xy_offset = np.linalg.norm(
            pos[:2] - pre_state_info['objects'][obj_name]['pos'][:2]
        )
        max_offset = max(max_offset, xy_offset)

        # Gripper contact heuristic
        if env.get_gripper_closed_norm() > 0.15 and current_z > stats.min_z + 0.02:
            stats.gripper_contact_frames += 1

        # Time to min z
        if current_z == stats.min_z:
            stats.time_to_min_z = t + 1

        # Track object-object contacts (for drop_at_wrong_place interaction mode)
        try:
            contacts = env.get_contact_summary()
            table_kw = ['table', 'floor', 'ground', 'bin']
            for c in contacts:
                geom_pair = [c.geom1, c.geom2]
                involves_obj = any(obj_name in g for g in geom_pair)
                if involves_obj:
                    other_geom = c.geom1 if obj_name in c.geom2 else c.geom2
                    if not any(kw in other_geom.lower() for kw in table_kw):
                        contact_geoms_seen.add(other_geom)
        except Exception:
            pass

        if step_callback:
            step_callback(t)

    stats.obj_contact_geoms = list(contact_geoms_seen)
    stats.max_offset_xy = max_offset
    stats.obj_z_trajectory = z_trajectory
    stats.obj_pos_trajectory = pos_trajectory
    stats.obj_quat_trajectory = quat_trajectory

    return stats
