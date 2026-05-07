#!/usr/bin/env python
"""
Unit tests for v5.0 Clean Trajectory Collector rollout collection.

Tests:
- PolicyResult unwrapping in collect_from_rollouts
- min_successes / max_attempts logic
- start_episode is called for RNN-based policies
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, field

from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
from error_benchmark.framework.core import CleanTrajectory


@dataclass
class MockPolicyResult:
    """Mimics PolicyResult from collector.py."""
    action: np.ndarray
    info: dict = field(default_factory=dict)


@pytest.fixture
def mock_env():
    """Create a mock EnvWrapper that simulates a simple task."""
    env = MagicMock()
    env._env = MagicMock()
    env._env.reset.return_value = {"obs": np.zeros(10)}
    env.step.return_value = ({"obs": np.zeros(10)}, 0.0, False, {})
    env.get_sim_state_flat.return_value = np.zeros(100)
    env.check_success.return_value = False
    env.get_task_phase.return_value = "reach"
    return env


@pytest.fixture
def task_config():
    return {"task": {"name": "pick_place"}}


@pytest.fixture
def collector(mock_env, task_config):
    return CleanTrajectoryCollector(mock_env, task_config, task_name="pick_place")


class TestPolicyResultUnwrapping:
    """Test that collect_from_rollouts correctly unwraps PolicyResult."""

    def test_unwraps_policy_result_with_action_attr(self, collector, mock_env):
        """predict() returning PolicyResult should have .action extracted."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(
            action=np.zeros(7), info={}
        )
        mock_env.check_success.return_value = True  # Succeed immediately

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=1,
            max_attempts=2,
            max_steps=5,
        )

        assert len(trajs) >= 1
        # Verify step() was called with the unwrapped action, not PolicyResult
        step_call_args = mock_env.step.call_args_list
        for c in step_call_args:
            action_arg = c[0][0]
            assert isinstance(action_arg, np.ndarray), \
                f"Expected ndarray, got {type(action_arg)}"

    def test_unwraps_plain_ndarray(self, collector, mock_env):
        """predict() returning plain ndarray should also work."""
        policy = MagicMock()
        # ndarray has no .action attribute, so hasattr check returns False
        # and the fallback path uses the raw return value
        policy.predict.return_value = np.zeros(7)
        mock_env.check_success.return_value = True

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=1,
            max_attempts=2,
            max_steps=5,
        )

        assert len(trajs) >= 1


class TestMinSuccessesMaxAttempts:
    """Test the collect-until-K-successes logic."""

    def test_collects_until_min_successes(self, collector, mock_env):
        """Should keep trying until min_successes are reached."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))

        # First 3 rollouts fail, next 2 succeed
        call_count = [0]

        def success_schedule():
            call_count[0] += 1
            return call_count[0] > 3

        mock_env.check_success.side_effect = lambda: success_schedule()

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=2,
            max_attempts=20,
            max_steps=3,
        )

        assert len(trajs) == 2

    def test_respects_max_attempts(self, collector, mock_env):
        """Should stop after max_attempts even if min_successes not reached."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))
        mock_env.check_success.return_value = False

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=10,
            max_attempts=5,
            max_steps=3,
        )

        # Should have 0 successes but stopped after 5 attempts
        assert len(trajs) == 0
        # reset should have been called 5 times
        assert mock_env._env.reset.call_count == 5

    def test_stops_early_when_min_successes_reached(self, collector, mock_env):
        """Should stop as soon as min_successes is reached."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))
        mock_env.check_success.return_value = True

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=3,
            max_attempts=100,
            max_steps=2,
        )

        assert len(trajs) == 3
        # Should have tried exactly 3 times (all succeeded)
        assert mock_env._env.reset.call_count == 3


class TestStartEpisodeCalled:
    """Test that start_episode() is called for policies that support it."""

    def test_start_episode_called(self, collector, mock_env):
        """Policies with start_episode() should have it called each rollout."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))
        policy.start_episode = MagicMock()
        mock_env.check_success.return_value = True

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=3,
            max_attempts=5,
            max_steps=2,
        )

        assert len(trajs) == 3
        assert policy.start_episode.call_count == 3
        assert policy.reset.call_count == 3

    def test_no_start_episode_if_not_available(self, collector, mock_env):
        """Policies without start_episode() should not crash."""
        policy = MagicMock(spec=['predict', 'reset'])
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))
        mock_env.check_success.return_value = True

        # Should not raise
        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=1,
            max_attempts=2,
            max_steps=2,
        )

        assert len(trajs) >= 1


class TestTrajectoryOutput:
    """Test the output trajectory format."""

    def test_trajectory_ids_are_sequential(self, collector, mock_env):
        """Trajectory IDs should be sequential starting from 0."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.zeros(7))
        mock_env.check_success.return_value = True

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=3,
            max_attempts=5,
            max_steps=2,
            policy_name="bc_rnn",
        )

        assert trajs[0].trajectory_id == "rollout_pick_place_bc_rnn_0"
        assert trajs[1].trajectory_id == "rollout_pick_place_bc_rnn_1"
        assert trajs[2].trajectory_id == "rollout_pick_place_bc_rnn_2"

    def test_trajectory_has_actions_and_states(self, collector, mock_env):
        """Each trajectory should have actions and states arrays."""
        policy = MagicMock()
        policy.predict.return_value = MockPolicyResult(action=np.ones(7))
        mock_env.check_success.return_value = True

        trajs = collector.collect_from_rollouts(
            policy_adapter=policy,
            min_successes=1,
            max_attempts=2,
            max_steps=3,
        )

        traj = trajs[0]
        assert traj.actions is not None
        assert traj.states is not None
        assert traj.source == "vla_pi0"  # default policy_name
        assert traj.task_name == "pick_place"
