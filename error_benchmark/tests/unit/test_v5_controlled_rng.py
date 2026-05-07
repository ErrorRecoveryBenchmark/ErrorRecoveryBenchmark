"""Unit tests for ControlledRng magnitude-controlled RNG wrapper."""

import numpy as np
import pytest

from error_benchmark.framework.utils.controlled_rng import ControlledRng


class TestControlledRng:
    """Tests for ControlledRng."""

    def test_uniform_interpolation_midpoint(self):
        """uniform() returns linearly interpolated value at fraction=0.5."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.5)
        result = rng.uniform(0.0, 1.0)
        assert result == pytest.approx(0.5)

    def test_uniform_interpolation_min(self):
        """fraction=0.0 returns low."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.0)
        assert rng.uniform(0.03, 0.10) == pytest.approx(0.03)

    def test_uniform_interpolation_max(self):
        """fraction=1.0 returns high."""
        rng = ControlledRng(np.random.RandomState(42), fraction=1.0)
        assert rng.uniform(0.03, 0.10) == pytest.approx(0.10)

    def test_uniform_interpolation_custom(self):
        """fraction=0.3 returns low + 0.3 * (high - low)."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.3)
        result = rng.uniform(0.03, 0.10)
        expected = 0.03 + 0.3 * (0.10 - 0.03)
        assert result == pytest.approx(expected)

    def test_uniform_with_size(self):
        """uniform() with size returns array of identical values."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.5)
        result = rng.uniform(0.0, 1.0, size=5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        np.testing.assert_allclose(result, 0.5)

    def test_randn_passthrough(self):
        """randn() passes through to underlying RNG (random, not deterministic)."""
        base = np.random.RandomState(42)
        rng = ControlledRng(base, fraction=0.5)
        result = rng.randn(3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # Should not be all zeros or all same value
        assert not np.allclose(result, result[0])

    def test_randint_passthrough(self):
        """randint() passes through to underlying RNG."""
        base = np.random.RandomState(42)
        rng = ControlledRng(base, fraction=0.5)
        result = rng.randint(0, 100, size=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(0 <= v < 100 for v in result)

    def test_choice_passthrough(self):
        """choice() passes through to underlying RNG."""
        base = np.random.RandomState(42)
        rng = ControlledRng(base, fraction=0.5)
        result = rng.choice([10, 20, 30])
        assert result in [10, 20, 30]

    def test_shuffle_passthrough(self):
        """shuffle() passes through to underlying RNG."""
        base = np.random.RandomState(42)
        rng = ControlledRng(base, fraction=0.5)
        arr = [1, 2, 3, 4, 5]
        rng.shuffle(arr)
        # Should be a permutation (same elements)
        assert sorted(arr) == [1, 2, 3, 4, 5]

    def test_random_returns_fraction(self):
        """random() returns fraction as deterministic value."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.7)
        assert rng.random() == pytest.approx(0.7)

    def test_random_with_size(self):
        """random(size=N) returns array of fraction values."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.7)
        result = rng.random(size=3)
        np.testing.assert_allclose(result, 0.7)

    def test_fraction_clipping_above(self):
        """fraction > 1.0 is clipped to 1.0."""
        rng = ControlledRng(np.random.RandomState(42), fraction=1.5)
        assert rng.fraction == pytest.approx(1.0)
        assert rng.uniform(0.0, 1.0) == pytest.approx(1.0)

    def test_fraction_clipping_below(self):
        """fraction < 0.0 is clipped to 0.0."""
        rng = ControlledRng(np.random.RandomState(42), fraction=-0.5)
        assert rng.fraction == pytest.approx(0.0)
        assert rng.uniform(0.0, 1.0) == pytest.approx(0.0)

    def test_fraction_setter(self):
        """fraction property setter updates the value with clipping."""
        rng = ControlledRng(np.random.RandomState(42), fraction=0.5)
        rng.fraction = 0.8
        assert rng.fraction == pytest.approx(0.8)
        assert rng.uniform(0.0, 1.0) == pytest.approx(0.8)

    def test_getattr_delegation(self):
        """Unrecognized attributes delegate to underlying RNG."""
        base = np.random.RandomState(42)
        rng = ControlledRng(base, fraction=0.5)
        # get_state is a method on RandomState
        state = rng.get_state()
        assert state[0] == 'MT19937'

    def test_training_level_sweep(self):
        """Simulate L0-L9 training levels: fractions 0.0, 0.1, ..., 0.9."""
        base = np.random.RandomState(42)
        low, high = 0.03, 0.10
        values = []
        for level in range(10):
            fraction = level / 9.0
            rng = ControlledRng(base, fraction=fraction)
            values.append(rng.uniform(low, high))

        # Values should be monotonically increasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

        # First and last should be close to low and high
        assert values[0] == pytest.approx(low)
        assert values[9] == pytest.approx(high)
