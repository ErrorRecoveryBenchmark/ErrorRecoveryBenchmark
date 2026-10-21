import os

import numpy as np
import pytest

from error_benchmark.framework.policy_adapter import PolicyServerAdapter
from error_benchmark.scripts.training.eval_clean_multi import configure_runtime_env
from error_benchmark.scripts.training.openvla_oft_vla_server import (
    OpenVLAOFTConfig,
    OpenVLAOFTModelWrapper,
    postprocess_openvla_actions,
)


def test_clean_worker_configures_headless_egl(monkeypatch):
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    configure_runtime_env(6)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_openvla_oft_config_matches_training_recipe():
    cfg = OpenVLAOFTConfig(
        pretrained_checkpoint="/checkpoint",
        unnorm_key="recover_bench_open_vla/coffee_nominal",
    )
    assert cfg.use_l1_regression is True
    assert cfg.use_diffusion is False
    assert cfg.use_film is False
    assert cfg.num_images_in_input == 2
    assert cfg.use_proprio is True
    assert cfg.center_crop is True
    assert cfg.num_open_loop_steps == 8
    assert cfg.lora_rank == 32


def test_openvla_observation_maps_two_images_and_8d_state():
    obs = {
        "state": np.arange(8, dtype=np.float32),
        "images": {
            "agentview": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.ones((224, 224, 3), dtype=np.uint8),
        },
    }
    converted = OpenVLAOFTModelWrapper._observation(obs)
    assert converted["state"].shape == (8,)
    assert converted["full_image"].shape == (224, 224, 3)
    assert converted["wrist_image"].shape == (224, 224, 3)

    with pytest.raises(KeyError):
        OpenVLAOFTModelWrapper._observation(
            {"state": obs["state"], "images": {"agentview": obs["images"]["agentview"]}}
        )


def test_gripper_postprocessing_matches_official_libero_eval():
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, -1] = [0.0, 0.5, 1.0]
    processed = postprocess_openvla_actions(actions)
    np.testing.assert_array_equal(processed[:, -1], [1.0, 0.0, -1.0])
    np.testing.assert_array_equal(actions[:, -1], [0.0, 0.5, 1.0])


def test_policy_adapter_consumes_full_eight_action_chunk():
    policy = PolicyServerAdapter(replan_interval=8)
    policy._sock = object()
    policy._ensure_connected = lambda: None
    policy._preprocess_obs = lambda obs: obs
    chunk = np.arange(56, dtype=np.float32).reshape(8, 7)
    calls = []

    def fake_command(message):
        calls.append(message)
        return {"action": chunk}

    policy._send_command = fake_command
    outputs = [policy.predict_from_obs({}).action for _ in range(8)]
    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(outputs), chunk)
