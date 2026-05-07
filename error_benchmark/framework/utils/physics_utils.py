#!/usr/bin/env python
"""
Physics Utilities - Force direction strategies

Extracted from v4 ImpulseInjector for reuse by Error Skills.
"""

import numpy as np
from typing import Optional


DIRECTION_STRATEGIES = ["random_unit", "gravity_aligned", "lateral_to_motion"]


def compute_force_direction(
    strategy: str,
    rng: np.random.RandomState,
    object_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute a force direction vector based on the given strategy.

    Args:
        strategy: One of 'random_unit', 'gravity_aligned', 'lateral_to_motion'
        rng: Random state for reproducibility
        object_velocity: Object linear velocity (3,), needed for 'lateral_to_motion'

    Returns:
        Unit direction vector (3,)
    """
    if strategy == "random_unit":
        return _dir_random_unit(rng)
    elif strategy == "gravity_aligned":
        return _dir_gravity_aligned(rng)
    elif strategy == "lateral_to_motion":
        return _dir_lateral_to_motion(rng, object_velocity)
    else:
        return _dir_random_unit(rng)


def _dir_random_unit(rng: np.random.RandomState) -> np.ndarray:
    """Random unit sphere sampling."""
    vec = rng.randn(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0.0, 0.0, -1.0])
    return vec / norm


def _dir_gravity_aligned(rng: np.random.RandomState) -> np.ndarray:
    """Downward-dominant direction (simulates gravity-induced drops)."""
    lateral_noise = rng.randn(2) * 0.1
    direction = np.array([lateral_noise[0], lateral_noise[1], -1.0])
    return direction / np.linalg.norm(direction)


def _dir_lateral_to_motion(
    rng: np.random.RandomState,
    object_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Perpendicular to object's motion direction (simulates side collision).
    Falls back to random_unit if object is stationary or velocity not provided.
    """
    if object_velocity is None:
        return _dir_random_unit(rng)

    speed = np.linalg.norm(object_velocity)
    if speed < 1e-4:
        return _dir_random_unit(rng)

    # Gram-Schmidt orthogonalization
    motion_dir = object_velocity / speed
    rand_vec = rng.randn(3)
    lateral = rand_vec - np.dot(rand_vec, motion_dir) * motion_dir

    norm = np.linalg.norm(lateral)
    if norm < 1e-8:
        return _dir_random_unit(rng)

    return lateral / norm
