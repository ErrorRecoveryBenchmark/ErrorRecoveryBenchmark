#!/usr/bin/env python
"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from error_benchmark.framework.core import ErrorSpec, ErrorScene, DetectionResult
from error_benchmark.framework.env_wrapper import EnvWrapper


@pytest.fixture
def sample_task_config():
    """Sample task configuration for testing"""
    return {
        'task_name': 'pick_place',
        'objects': [
            {
                'name': 'cube',
                'body_name': 'cube_main',
                'grasp_geoms': ['cube_g0', 'cube_g1']
            }
        ],
        'phases': {
            'reach': 'eef_to_obj_dist < 0.06',
            'grasp': 'eef_to_obj_dist < 0.06 and gripper_closed_norm > 0.7',
            'lift': 'obj_z > 0.85',
            'transport': 'obj_to_target_dist < 0.05',
            'place': 'success == True'
        },
        'thresholds': {
            'reach': 0.06,
            'grasp_closed': 0.7,
            'lift_height': 0.85,
            'transport': 0.05
        },
        'completion_weights': {
            'reach': 0.1,
            'grasp': 0.2,
            'lift': 0.2,
            'transport': 0.3,
            'place': 0.2
        },
        'gripper': {
            'type': 'two_finger',
            'open_qpos': [0.0, 0.0],
            'close_qpos': [0.04, -0.04]
        },
        'controller': 'OSC'
    }


@pytest.fixture
def sample_error_spec():
    """Sample ErrorSpec for testing"""
    return ErrorSpec(
        type="impulse",
        family="physics",
        target={"body": "cube_main"},
        params={
            "force_norm": 5.0,
            "force_direction": [1.0, 0.0, 0.0],
            "duration_steps": 1
        },
        apply={"mode": "at_step", "t": 50},
        seed=42,
        direction_strategy="random_unit"
    )


@pytest.fixture
def sample_state_info():
    """Sample state_info dict for testing"""
    return {
        'step': 50,
        'eef_pos': np.array([0.5, 0.0, 0.8]),
        'eef_quat': np.array([1.0, 0.0, 0.0, 0.0]),
        'robot_qpos': np.zeros(7),
        'robot_qvel': np.zeros(7),
        'gripper_qpos_raw': np.array([0.02, -0.02]),
        'gripper_closed_norm': 0.5,
        'objects': {
            'cube': {
                'pos': np.array([0.5, 0.0, 0.75]),
                'quat': np.array([1.0, 0.0, 0.0, 0.0]),
                'linvel': np.zeros(3),
                'angvel': np.zeros(3),
            }
        },
        'contact_summary': [],
        'task_phase': 'grasp',
        'task_completion': {
            'reach': True,
            'grasp': False,
            'lift': False,
            'transport': False,
            'place': False
        }
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env():
    """Mock robosuite environment for testing"""
    # This would be implemented with a real mock in practice
    # For now, return None - tests that need real env should be integration tests
    return None
