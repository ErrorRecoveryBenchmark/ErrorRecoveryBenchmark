from copy import deepcopy

import pytest

from error_benchmark.scripts.conversion.convert_lerobot_v21_to_gr00t import (
    LIBERO_MODALITY,
    build_output_info,
    missing_source_columns,
    validate_source_info,
)


def source_info():
    return {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": [84, 84, 3]},
            "wrist_image": {"dtype": "image", "shape": [84, 84, 3]},
            "state": {"dtype": "float32", "shape": [8]},
            "actions": {"dtype": "float32", "shape": [7]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }


def test_output_info_uses_gr00t_video_and_low_dimensional_keys():
    info = build_output_info(source_info(), episode_count=2)

    assert info["total_videos"] == 4
    assert info["features"]["observation.state"]["shape"] == [8]
    assert info["features"]["action"]["shape"] == [7]
    assert info["features"]["observation.images.image"]["dtype"] == "video"
    assert info["features"]["observation.images.wrist_image"]["info"]["video.fps"] == 20
    assert "image" not in info["features"]
    assert "actions" not in info["features"]


def test_modality_matches_official_libero_panda_dimensions():
    assert LIBERO_MODALITY["state"]["gripper"] == {"start": 6, "end": 8}
    assert LIBERO_MODALITY["action"]["gripper"] == {"start": 6, "end": 7}
    assert LIBERO_MODALITY["video"]["image"]["original_key"] == "observation.images.image"
    assert LIBERO_MODALITY["annotation"]["human.action.task_description"] == {
        "original_key": "task_index"
    }


def test_source_validation_rejects_incompatible_state_shape():
    info = deepcopy(source_info())
    info["features"]["state"]["shape"] = [7]

    with pytest.raises(ValueError, match=r"state must have shape \[8\]"):
        validate_source_info(info)


def test_required_column_check_accepts_the_real_source_schema():
    assert not missing_source_columns(
        [
            "image",
            "wrist_image",
            "state",
            "actions",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    )
