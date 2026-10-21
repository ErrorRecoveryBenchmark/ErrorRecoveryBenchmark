#!/usr/bin/env python
"""Consecutive-frame success tracking for RecoverBench evaluation.

EVALUATION.md §1 defines task success as the underlying environment's
instantaneous success check returning ``True`` for **N consecutive frames**
(N = 10) during a rollout. The protocol calls this rule *load-bearing* and
forbids relaxing it for any Recovery Behavior Group (including RBG_E).

This module centralizes that rule so every rollout loop — error-scene RSR and
paired clean CSR, across Pi0.5 and BC-RNN, single- and multi-worker — applies
one identical criterion instead of breaking on the first success frame.

It is deliberately dependency-free (standard library only) so it imports
without robosuite / openpi and stays unit-testable in any environment.
"""

from __future__ import annotations

# EVALUATION.md §1 — load-bearing; do not loosen in future revisions.
SUCCESS_STREAK_FRAMES = 10

# Provenance string stamped into result JSONs so a run is self-describing
# (see acceptance standards S-A1 / S-E6).
SUCCESS_CRITERION_NAME = "10_consecutive_frames"


class ConsecutiveSuccessTracker:
    """Declare success only after ``n`` consecutive instantaneous-success frames.

    Usage (one tracker per episode/scene)::

        tracker = ConsecutiveSuccessTracker()
        for step in range(max_steps):
            obs, *_ = env_wrapper.step(action)
            if tracker.update(bool(env_wrapper.check_success())):
                success = True
                break

    A single ``True`` frame is **not** a success; the streak resets to zero on
    any ``False`` frame.
    """

    def __init__(self, n: int = SUCCESS_STREAK_FRAMES):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self._streak = 0

    def reset(self) -> None:
        """Clear the streak; call at the start of every episode/scene."""
        self._streak = 0

    def update(self, success_now: bool) -> bool:
        """Feed one frame's instantaneous success.

        Returns ``True`` once ``n`` consecutive successes have been observed.
        """
        if success_now:
            self._streak += 1
        else:
            self._streak = 0
        return self.succeeded

    @property
    def streak(self) -> int:
        """Current run of consecutive successes."""
        return self._streak

    @property
    def succeeded(self) -> bool:
        """Whether the consecutive-frame threshold has been reached."""
        return self._streak >= self.n
