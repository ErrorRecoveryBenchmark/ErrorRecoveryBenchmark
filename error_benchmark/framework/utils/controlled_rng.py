#!/usr/bin/env python
"""
ControlledRng - Magnitude-controlled random number generator wrapper.

Wraps np.random.RandomState to control magnitude distribution in training
scene generation. uniform() calls return linearly interpolated values
(fraction 0.0→1.0), while randn()/randint() remain random for direction
diversity.

Usage:
    base_rng = np.random.RandomState(seed)
    rng = ControlledRng(base_rng, fraction=0.5)
    val = rng.uniform(0.03, 0.10)  # Returns 0.03 + 0.5 * (0.10 - 0.03) = 0.065
    direction = rng.randn(3)       # Random (passthrough)
"""

import numpy as np
from typing import Optional, Union


class ControlledRng:
    """
    RNG wrapper that controls magnitude via linear interpolation.

    For training scene generation, this ensures uniform magnitude
    distribution across L0-L9 levels (fraction 0.0, 0.1, ..., 0.9).

    - uniform(low, high) → low + fraction * (high - low)
    - randn(*args) → passthrough (random direction)
    - randint(*args) → passthrough (random choice)
    - All other methods → passthrough to underlying RNG
    """

    def __init__(self, base_rng: np.random.RandomState, fraction: float = 0.5):
        """
        Args:
            base_rng: Underlying numpy RandomState for non-controlled calls
            fraction: Value in [0.0, 1.0] controlling magnitude.
                0.0 = minimum magnitude, 1.0 = maximum magnitude.
        """
        self._rng = base_rng
        self._fraction = np.clip(fraction, 0.0, 1.0)

    @property
    def fraction(self) -> float:
        return self._fraction

    @fraction.setter
    def fraction(self, value: float):
        self._fraction = np.clip(value, 0.0, 1.0)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Return linearly interpolated value: low + fraction * (high - low)."""
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        value = low + self._fraction * (high - low)
        if size is not None:
            return np.full(size, value)
        return float(value) if value.ndim == 0 else value

    def randn(self, *args):
        """Passthrough — random direction diversity."""
        return self._rng.randn(*args)

    def randint(self, *args, **kwargs):
        """Passthrough — random choice."""
        return self._rng.randint(*args, **kwargs)

    def choice(self, *args, **kwargs):
        """Passthrough — random selection."""
        return self._rng.choice(*args, **kwargs)

    def shuffle(self, *args, **kwargs):
        """Passthrough — random shuffle."""
        return self._rng.shuffle(*args, **kwargs)

    def random(self, size=None):
        """Return fraction as deterministic 'random' value."""
        if size is not None:
            return np.full(size, self._fraction)
        return self._fraction

    def __getattr__(self, name):
        """Delegate all other methods to underlying RNG."""
        return getattr(self._rng, name)
