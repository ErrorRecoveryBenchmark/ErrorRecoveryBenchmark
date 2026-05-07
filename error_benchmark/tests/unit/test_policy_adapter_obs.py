#!/usr/bin/env python
"""
Unit tests for robomimic observation normalization in the policy adapter.
"""

import numpy as np

from error_benchmark.framework.policy_adapter import RobomimicPolicyAdapter


def test_to_robosuite_obs_maps_object_state_and_casts_low_dim():
    obs = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "robot0_gripper_qpos": np.array([0.1, -0.1], dtype=np.float64),
        "object-state": np.arange(23, dtype=np.float64),
        "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((84, 84, 3), dtype=np.uint8),
        "hinge_angle": np.float64(0.25),
        "robot0_joint_vel": np.ones(7, dtype=np.float64),
    }

    converted = RobomimicPolicyAdapter._to_robosuite_obs(obs)

    assert set(converted.keys()) == {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
        "agentview_image",
        "robot0_eye_in_hand_image",
    }
    assert converted["robot0_eef_pos"].dtype == np.float32
    assert converted["robot0_eef_quat"].dtype == np.float32
    assert converted["robot0_gripper_qpos"].dtype == np.float32
    assert converted["object"].dtype == np.float32
    assert converted["agentview_image"].dtype == np.uint8
    assert converted["robot0_eye_in_hand_image"].dtype == np.uint8


def test_to_robosuite_obs_drops_object_dtype_entries():
    obs = {
        "_raw_obs": {
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "object": np.array([{"bad": "value"}], dtype=object),
            "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
        }
    }

    converted = RobomimicPolicyAdapter._to_robosuite_obs(obs)

    assert set(converted.keys()) == {"robot0_eef_pos", "agentview_image"}
    assert converted["robot0_eef_pos"].dtype == np.float32
