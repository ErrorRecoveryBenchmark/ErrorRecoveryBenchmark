import numpy as np
import pytest

from error_benchmark.scripts.training.gr00t_vla_server import (
    actions_to_env_chunks,
    build_gr00t_observation,
)


def _server_obs(offset: int = 0) -> dict:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + offset
    return {
        "state": np.arange(8, dtype=np.float32) + offset,
        "images": {
            "agentview": image,
            "robot0_eye_in_hand": image + 20,
        },
    }


def test_build_gr00t_observation_matches_libero_panda_modalities():
    first = _server_obs()
    second = _server_obs(1)

    observation = build_gr00t_observation(
        [first, second],
        ["make coffee", "stack the blocks"],
    )

    assert observation["video.image"].shape == (2, 1, 2, 3, 3)
    assert observation["video.image"].dtype == np.uint8
    np.testing.assert_array_equal(
        observation["video.image"][0, 0],
        first["images"]["agentview"],
    )
    assert observation["state.x"].shape == (2, 1, 1)
    assert observation["state.gripper"].shape == (2, 1, 2)
    assert observation["state.x"].dtype == np.float32
    assert observation["annotation.human.action.task_description"] == [
        "make coffee",
        "stack the blocks",
    ]


def test_build_gr00t_observation_rejects_missing_camera():
    obs = _server_obs()
    del obs["images"]["robot0_eye_in_hand"]

    with pytest.raises(KeyError, match="robot0_eye_in_hand"):
        build_gr00t_observation([obs], ["task"])


def test_actions_to_env_chunks_matches_raw_robosuite_gripper_contract():
    batch = 2
    horizon = 5
    actions = {
        f"action.{key}": np.full((batch, horizon, 1), value, dtype=np.float32)
        for key, value in zip(
            ("x", "y", "z", "roll", "pitch", "yaw"),
            (1, 2, 3, 4, 5, 6),
            strict=True,
        )
    }
    actions["action.gripper"] = np.array(
        [[[-1.0], [-0.25], [0.0], [0.25], [1.0]]] * batch,
        dtype=np.float32,
    )

    chunks = actions_to_env_chunks(actions)

    assert len(chunks) == batch
    assert chunks[0].shape == (horizon, 7)
    np.testing.assert_array_equal(
        chunks[0][:, -1],
        [-1.0, -1.0, -1.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(chunks[0][0, :6], [1, 2, 3, 4, 5, 6])
