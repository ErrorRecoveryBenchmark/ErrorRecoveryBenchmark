#!/usr/bin/env python
"""
Unit tests for v5.0 Utility modules
"""

import pytest
import numpy as np
from error_benchmark.framework.utils.math_utils import (
    euler_to_quat,
    quat_multiply,
    quat_conjugate,
    quat_rotate_vector,
    random_rotation_quat,
    random_translation,
)
from error_benchmark.framework.utils.physics_utils import (
    compute_force_direction,
    DIRECTION_STRATEGIES,
)


class TestMathUtils:
    """Tests for quaternion math utilities."""

    def test_identity_quaternion(self):
        """Euler [0,0,0] -> identity quaternion."""
        q = euler_to_quat(np.array([0, 0, 0]))
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_euler_to_quat_90_roll(self):
        """90 degree roll rotation."""
        q = euler_to_quat(np.array([np.pi/2, 0, 0]))
        # Expected: [cos(45), sin(45), 0, 0]
        expected = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_quat_multiply_identity(self):
        """Multiplying by identity gives same quaternion."""
        identity = np.array([1, 0, 0, 0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quat_multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quat_multiply_self_conjugate(self):
        """q * q_conjugate = identity."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = quat_conjugate(q)
        result = quat_multiply(q, q_conj)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-10)

    def test_quat_conjugate(self):
        """Conjugate negates xyz components."""
        q = np.array([0.5, 0.3, -0.1, 0.2])
        q_conj = quat_conjugate(q)
        assert q_conj[0] == q[0]
        assert q_conj[1] == -q[1]
        assert q_conj[2] == -q[2]
        assert q_conj[3] == -q[3]

    def test_quat_rotate_vector_identity(self):
        """Identity quaternion doesn't change vector."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_vector(q, v)
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_quat_rotate_vector_90_z(self):
        """90 degree rotation around z-axis rotates x to y."""
        # 90 deg around z: q = [cos(45), 0, 0, sin(45)]
        q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        v = np.array([1, 0, 0])
        result = quat_rotate_vector(q, v)
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_random_rotation_quat(self):
        """Random rotation quaternion is unit quaternion."""
        rng = np.random.RandomState(42)
        q = random_rotation_quat(rng, max_angle=1.0)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_random_rotation_quat_bounded(self):
        """Random rotation angle is within bounds."""
        rng = np.random.RandomState(42)
        max_angle = 0.5
        for _ in range(100):
            q = random_rotation_quat(rng, max_angle)
            angle = 2 * np.arccos(np.clip(abs(q[0]), 0, 1))
            assert angle <= max_angle + 1e-10

    def test_random_translation(self):
        """Random translation magnitude is within bounds."""
        rng = np.random.RandomState(42)
        max_offset = 0.1
        for _ in range(100):
            t = random_translation(rng, max_offset)
            assert np.linalg.norm(t) <= max_offset + 1e-10


class TestPhysicsUtils:
    """Tests for physics utilities."""

    def test_direction_strategies_list(self):
        """All 3 strategies are defined."""
        assert len(DIRECTION_STRATEGIES) == 3
        assert "random_unit" in DIRECTION_STRATEGIES
        assert "gravity_aligned" in DIRECTION_STRATEGIES
        assert "lateral_to_motion" in DIRECTION_STRATEGIES

    def test_random_unit_direction(self):
        """random_unit returns unit vector."""
        rng = np.random.RandomState(42)
        d = compute_force_direction("random_unit", rng)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_gravity_aligned_direction(self):
        """gravity_aligned points mostly downward."""
        rng = np.random.RandomState(42)
        d = compute_force_direction("gravity_aligned", rng)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)
        assert d[2] < 0  # Points downward

    def test_lateral_to_motion_with_velocity(self):
        """lateral_to_motion is perpendicular to velocity."""
        rng = np.random.RandomState(42)
        velocity = np.array([1.0, 0.0, 0.0])
        d = compute_force_direction("lateral_to_motion", rng, velocity)
        # Should be roughly perpendicular to velocity
        dot = abs(np.dot(d, velocity / np.linalg.norm(velocity)))
        assert dot < 0.1  # Nearly perpendicular

    def test_lateral_to_motion_no_velocity(self):
        """lateral_to_motion falls back to random when no velocity."""
        rng = np.random.RandomState(42)
        d = compute_force_direction("lateral_to_motion", rng, None)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_lateral_to_motion_zero_velocity(self):
        """lateral_to_motion falls back to random for zero velocity."""
        rng = np.random.RandomState(42)
        d = compute_force_direction("lateral_to_motion", rng, np.zeros(3))
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)

    def test_unknown_strategy_fallback(self):
        """Unknown strategy falls back to random_unit."""
        rng = np.random.RandomState(42)
        d = compute_force_direction("nonexistent", rng)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)
