#!/usr/bin/env python
"""
Math Utilities - Quaternion operations and geometric helpers

Extracted from v4 PosePerturbInjector for reuse by Error Skills.
All quaternions use (w, x, y, z) format.
"""

import numpy as np


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles [roll, pitch, yaw] to quaternion [w, x, y, z].
    Uses ZYX intrinsic rotation order.

    Args:
        euler: (3,) array [roll, pitch, yaw] in radians

    Returns:
        (4,) quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication q1 * q2 (Hamilton product).
    Format: [w, x, y, z].

    Args:
        q1: (4,) quaternion
        q2: (4,) quaternion

    Returns:
        (4,) normalized quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = np.array([w, x, y, z])
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result /= norm
    return result


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate (inverse for unit quaternions).

    Args:
        q: (4,) quaternion [w, x, y, z]

    Returns:
        (4,) conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply_raw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Raw quaternion multiplication without normalization."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q: v' = q * v * q_conj.

    Args:
        q: (4,) unit quaternion [w, x, y, z]
        v: (3,) vector

    Returns:
        (3,) rotated vector
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conjugate(q)
    # Use raw multiply (no normalization) since intermediate is not unit
    rotated = _quat_multiply_raw(_quat_multiply_raw(q, v_quat), q_conj)
    return rotated[1:]  # Extract xyz


def random_rotation_quat(rng: np.random.RandomState, max_angle: float) -> np.ndarray:
    """
    Generate a random rotation quaternion with angle up to max_angle.

    Args:
        rng: Random state
        max_angle: Maximum rotation angle in radians

    Returns:
        (4,) unit quaternion [w, x, y, z]
    """
    # Random axis on unit sphere
    axis = rng.randn(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= norm

    # Random angle up to max_angle
    angle = rng.uniform(0, max_angle)

    # Axis-angle to quaternion
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def random_translation(rng: np.random.RandomState, max_offset: float) -> np.ndarray:
    """
    Generate a random translation vector with magnitude up to max_offset.

    Args:
        rng: Random state
        max_offset: Maximum offset magnitude in meters

    Returns:
        (3,) translation vector
    """
    direction = rng.randn(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.zeros(3)
    direction /= norm

    magnitude = rng.uniform(0, max_offset)
    return direction * magnitude
