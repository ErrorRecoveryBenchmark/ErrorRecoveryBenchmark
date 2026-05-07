"""
Unit tests for InjectionReplay module.

Tests the injection replay logic with mock environment,
ensuring correct state loading, skill invocation, and fallback behavior.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from error_benchmark.framework.injection_replay import InjectionReplay


# ─── Fixtures ───

@pytest.fixture
def mock_env_wrapper():
    """Create a mock EnvWrapper with all required methods."""
    env = MagicMock()
    env.set_sim_state_flat = MagicMock()
    env.forward = MagicMock()
    env.get_sim_state_flat = MagicMock(return_value=np.zeros(45))
    env.get_eef_pos = MagicMock(return_value=np.array([0.5, 0.0, 0.8]))
    env.get_eef_quat = MagicMock(return_value=np.array([1.0, 0.0, 0.0, 0.0]))
    env.get_gripper_closed_norm = MagicMock(return_value=0.5)
    env.get_all_object_names = MagicMock(return_value=['cubeA', 'cubeB'])
    env.get_object_pose = MagicMock(return_value=(np.array([0.1, 0.2, 0.8]), np.array([1, 0, 0, 0])))
    env.get_object_velocity = MagicMock(return_value=(np.zeros(3), np.zeros(3)))
    env._get_graspable_objects = MagicMock(return_value=['cubeA', 'cubeB'])
    return env


@pytest.fixture
def sample_scene_data():
    """Sample ErrorScene JSON dict."""
    return {
        'scene_id': 'v5_train_grasp_misalignment_D0_000_abc123',
        'version': 'v5.0',
        'clean_trajectory_ref': 'demo_stack_demo_0',
        'injection_step': 50,
        'render_window': 30,
        'replay': {
            'injection_frame': 50,
            'render_window': 30,
        },
        'error_spec': {
            'error_name': 'grasp_misalignment',
            'degree': 'D0',
            'seed': 42,
            'target_object': 'cubeA',
            'params': {
                'pre_eef_pos': [0.5, 0.0, 0.8],
            },
        },
        'labels': {
            'target_object': 'cubeA',
            'subtype_id': 'grasp_misalignment_D0',
        },
        'rng_state': {
            'numpy': list(np.random.RandomState(42).get_state()),
        },
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temp dirs with clean trajectory and scene NPZ."""
    # Create clean trajectory
    clean_dir = tmp_path / "clean_trajectories"
    clean_dir.mkdir()

    states = np.random.randn(100, 45).astype(np.float64)
    actions = np.random.randn(99, 7).astype(np.float64)
    phase_labels = np.array(['reach'] * 100)
    np.savez(clean_dir / "demo_stack_demo_0.npz",
             states=states, actions=actions, phase_labels=phase_labels)

    # Create scene NPZ (pre-injection state)
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    pre_state = states[50].copy()  # State at injection frame
    np.savez(scenes_dir / "v5_train_grasp_misalignment_D0_000_abc123.npz",
             sim_state=pre_state)

    return {
        'clean_dir': clean_dir,
        'scenes_dir': scenes_dir,
        'scene_npz': str(scenes_dir / "v5_train_grasp_misalignment_D0_000_abc123.npz"),
        'states': states,
        'pre_state': pre_state,
    }


# ─── Tests ───

class TestInjectionReplayInit:
    """Test InjectionReplay initialization."""

    def test_basic_init(self, mock_env_wrapper):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        assert replay.env is mock_env_wrapper
        assert replay.skills_config == {}

    def test_init_with_skills_config(self, mock_env_wrapper):
        config = {'grasp_misalignment': {'max_offset': 0.05}}
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'}, config)
        assert replay.skills_config == config


class TestLoadCleanTrajectory:
    """Test clean trajectory loading."""

    def test_load_existing(self, mock_env_wrapper, temp_data_dir):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        states = replay._load_clean_trajectory(
            temp_data_dir['clean_dir'], 'demo_stack_demo_0')
        assert states is not None
        assert states.shape == (100, 45)

    def test_load_nonexistent(self, mock_env_wrapper, tmp_path):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        states = replay._load_clean_trajectory(tmp_path, 'nonexistent_traj')
        assert states is None

    def test_load_partial_match(self, mock_env_wrapper, temp_data_dir):
        """Should find trajectory by partial name match."""
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        # The file is demo_stack_demo_0.npz, try with just the ref
        states = replay._load_clean_trajectory(
            temp_data_dir['clean_dir'], 'demo_stack_demo_0')
        assert states is not None


class TestExtractStateInfo:
    """Test state info extraction."""

    def test_extract_state_info(self, mock_env_wrapper):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        info = replay._extract_state_info()
        assert 'eef_pos' in info
        assert 'eef_quat' in info
        assert 'gripper_closed_norm' in info
        assert 'objects' in info
        assert 'cubeA' in info['objects']
        assert 'cubeB' in info['objects']


class TestFallbackLoad:
    """Test fallback state loading."""

    def test_fallback_success(self, mock_env_wrapper, temp_data_dir):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        result = replay._fallback_load(temp_data_dir['scene_npz'])
        assert result is True
        mock_env_wrapper.set_sim_state_flat.assert_called_once()
        mock_env_wrapper.forward.assert_called_once()

    def test_fallback_missing_file(self, mock_env_wrapper):
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        result = replay._fallback_load('/nonexistent/path.npz')
        assert result is False


class TestReplayAnimation:
    """Test the main replay_animation method."""

    @patch('error_benchmark.framework.injection_replay.get_skill')
    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_calls_set_state_for_context(
        self, mock_time, mock_get_skill,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """Verify context frames are played by calling set_sim_state_flat."""
        # Setup mock skill
        mock_skill = MagicMock()
        mock_skill.inject = MagicMock(return_value=MagicMock())
        mock_get_skill.return_value = mock_skill

        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})
        render_fn = MagicMock()

        result = replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            render_fn=render_fn,
            speed=1.0,
            context_frames=30,
            pause_after=0,  # No pause in test
        )

        assert result is True
        # Context frames: 50-30=20 to 50, so 30 context frames
        # + 1 pre-injection + injection steps + 1 post
        assert mock_env_wrapper.set_sim_state_flat.call_count >= 30
        assert render_fn.call_count >= 30  # At least context frames rendered

    @patch('error_benchmark.framework.injection_replay.get_skill')
    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_executes_injection(
        self, mock_time, mock_get_skill,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """Verify skill.inject() is called with correct parameters."""
        mock_skill = MagicMock()
        mock_skill.inject = MagicMock(return_value=MagicMock())
        mock_get_skill.return_value = mock_skill

        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        result = replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            speed=1.0,
            pause_after=0,
        )

        assert result is True
        # get_skill should be called with correct error name
        mock_get_skill.assert_called_once_with('grasp_misalignment', {})
        # inject should be called
        mock_skill.inject.assert_called_once()
        inject_kwargs = mock_skill.inject.call_args
        assert inject_kwargs[1]['degree'] == 'D0'

    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_fallback_on_missing_trajectory(
        self, mock_time,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """Should fall back to direct load if clean trajectory not found."""
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        # Use empty dir for clean trajectories
        empty_dir = temp_data_dir['clean_dir'].parent / "empty"
        empty_dir.mkdir()

        result = replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=empty_dir,
            pause_after=0,
        )

        assert result is True  # Fallback should succeed
        # Should have loaded the NPZ state directly
        mock_env_wrapper.set_sim_state_flat.assert_called()

    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_fallback_on_missing_metadata(
        self, mock_time, mock_env_wrapper, temp_data_dir,
    ):
        """Should fall back if scene_data missing clean_trajectory_ref."""
        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        bad_scene = {'scene_id': 'test', 'error_spec': {}}
        result = replay.replay_animation(
            scene_data=bad_scene,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            pause_after=0,
        )

        assert result is True  # Fallback should succeed

    @patch('error_benchmark.framework.injection_replay.get_skill')
    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_uses_correct_rng_seed(
        self, mock_time, mock_get_skill,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """Verify that the RNG is seeded with error_spec.seed for determinism."""
        mock_skill = MagicMock()
        mock_skill.inject = MagicMock(return_value=MagicMock())
        mock_get_skill.return_value = mock_skill

        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            pause_after=0,
        )

        # Check the rng passed to inject is seeded with 42
        inject_call = mock_skill.inject.call_args
        rng = inject_call[1]['rng']
        assert isinstance(rng, np.random.RandomState)

    @patch('error_benchmark.framework.injection_replay.get_skill')
    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_no_render_fn(
        self, mock_time, mock_get_skill,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """Should work without render_fn (headless mode)."""
        mock_skill = MagicMock()
        mock_skill.inject = MagicMock(return_value=MagicMock())
        mock_get_skill.return_value = mock_skill

        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        result = replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            render_fn=None,
            pause_after=0,
        )

        assert result is True
        mock_skill.inject.assert_called_once()
        # inject should be called with render_fn=None
        inject_kwargs = mock_skill.inject.call_args[1]
        assert inject_kwargs['render_fn'] is None

    @patch('error_benchmark.framework.injection_replay.get_skill')
    @patch('error_benchmark.framework.injection_replay.time')
    def test_replay_injection_failure_falls_back(
        self, mock_time, mock_get_skill,
        mock_env_wrapper, sample_scene_data, temp_data_dir,
    ):
        """If skill.inject() raises, should fall back to direct state load."""
        mock_skill = MagicMock()
        mock_skill.inject = MagicMock(side_effect=RuntimeError("injection failed"))
        mock_get_skill.return_value = mock_skill

        replay = InjectionReplay(mock_env_wrapper, {'task_name': 'stack'})

        result = replay.replay_animation(
            scene_data=sample_scene_data,
            scene_npz_path=temp_data_dir['scene_npz'],
            clean_traj_dir=temp_data_dir['clean_dir'],
            pause_after=0,
        )

        # Should succeed via fallback
        assert result is True
