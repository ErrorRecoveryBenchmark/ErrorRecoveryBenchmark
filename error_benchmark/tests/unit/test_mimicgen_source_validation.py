#!/usr/bin/env python
"""Unit tests for stack MimicGen source validation and conversion helpers."""

import json
from pathlib import Path

import h5py
import numpy as np

from error_benchmark.framework.mimicgen_source_validation import (
    ensure_env_interface_attrs,
    find_first_binary_transition,
    validate_stack_prepared_demo,
)
from error_benchmark.scripts.mimicgen import augment_success_demos
from error_benchmark.scripts.mimicgen import convert_teleop_to_hdf5


STACK_TASK_CONFIG = {
    "task_name": "stack",
    "objects": [
        {"name": "cubeA"},
        {"name": "cubeB"},
    ],
    "mimicgen_source_validation": {
        "settle_linvel_threshold": 0.01,
        "settle_angvel_threshold": 0.05,
        "settle_hold_frames": 10,
        "max_settle_steps": 40,
        "min_pre_grasp_frames": 5,
        "min_post_grasp_frames": 21,
        "require_final_replay_success": True,
        "reject_if_success_at_start": True,
    },
}


def _write_demo_group(
    data_group: h5py.Group,
    demo_key: str,
    num_steps: int = 40,
    grasp_signal=None,
    attrs=None,
) -> None:
    group = data_group.create_group(demo_key)
    group.attrs["num_samples"] = num_steps
    if attrs:
        for key, value in attrs.items():
            group.attrs[key] = value

    group.create_dataset("actions", data=np.zeros((num_steps, 7), dtype=np.float64))
    group.create_dataset("states", data=np.zeros((num_steps, 8), dtype=np.float64))

    if grasp_signal is not None:
        datagen_info = group.create_group("datagen_info")
        term_group = datagen_info.create_group("subtask_term_signals")
        term_group.create_dataset("grasp", data=np.asarray(grasp_signal, dtype=np.float64))


def test_find_first_binary_transition_detects_first_rise():
    signal = np.array([0, 0, 0, 1, 1, 0, 1], dtype=np.float64)
    assert find_first_binary_transition(signal) == 3


def test_validate_stack_prepared_demo_accepts_valid_demo(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(
            data_group,
            "demo_0",
            grasp_signal=[0] * 8 + [1] * 32,
            attrs={"scene_id": "scene_valid", "validation_status": "accepted"},
        )

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.details["first_grasp_frame"] == 8


def test_validate_stack_prepared_demo_rejects_missing_datagen_info(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=None)

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is False
    assert result.reason == "missing_datagen_info"


def test_validate_stack_prepared_demo_rejects_all_zero_grasp_signal(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 40)

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is False
    assert result.reason == "no_grasp_transition"


def test_validate_stack_prepared_demo_rejects_too_early_transition(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=[0, 0, 1] + [1] * 37)

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is False
    assert result.reason == "grasp_transition_too_early"


def test_validate_stack_prepared_demo_rejects_too_late_transition(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    signal = [0] * 25 + [1] * 15
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=signal)

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is False
    assert result.reason == "grasp_transition_too_late"


def test_validate_stack_prepared_demo_rejects_unstable_start_metadata(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(
            data_group,
            "demo_0",
            grasp_signal=[0] * 8 + [1] * 32,
            attrs={"validation_status": "rejected", "validation_reason": "unstable_start"},
        )

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: True,
        )

    assert result.accepted is False
    assert result.reason == "unstable_start"


def test_validate_stack_prepared_demo_rejects_failed_final_replay(tmp_path: Path):
    hdf5_path = tmp_path / "prepared.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 8 + [1] * 32)

    with h5py.File(hdf5_path, "r") as handle:
        result = validate_stack_prepared_demo(
            demo_key="demo_0",
            demo_group=handle["data/demo_0"],
            task_config=STACK_TASK_CONFIG,
            replay_success_checker=lambda _: False,
        )

    assert result.accepted is False
    assert result.reason == "final_replay_not_success"


def test_stack_merge_only_copies_accepted_demo_and_sets_attrs(tmp_path: Path):
    prepared_path = tmp_path / "prepared.hdf5"
    existing_path = tmp_path / "existing.hdf5"

    with h5py.File(prepared_path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["env_args"] = json.dumps({"env_name": "Stack_D0"})
        data_group.attrs["total"] = 80
        data_group.attrs["scene_id_map"] = json.dumps(
            {"demo_0": "scene_accept", "demo_1": "scene_reject"}
        )
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 8 + [1] * 32)
        _write_demo_group(data_group, "demo_1", grasp_signal=[0] * 40)

    with h5py.File(existing_path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["env_args"] = json.dumps({"env_name": "Stack_D0"})
        data_group.attrs["total"] = 10
        data_group.attrs["scene_id_map"] = json.dumps({"demo_0": "seed_demo"})
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 8 + [1] * 32)

    report = convert_teleop_to_hdf5.validate_stack_prepared_hdf5(
        prepared_path,
        STACK_TASK_CONFIG,
        replay_success_checker=lambda demo_key: demo_key == "demo_0",
    )
    merged = convert_teleop_to_hdf5.merge_prepared_demos(
        existing_path,
        prepared_path,
        report["accepted"],
    )

    assert merged == 1

    with h5py.File(existing_path, "r") as handle:
        data_group = handle["data"]
        demo_keys = sorted(data_group.keys())
        assert demo_keys == ["demo_0", "demo_1"]
        merged_group = data_group["demo_1"]
        assert merged_group.attrs["demo_origin"] == "human_teleop"
        assert merged_group.attrs["scene_id"] == "scene_accept"
        assert merged_group.attrs["validation_status"] == "accepted"
        scene_id_map = json.loads(data_group.attrs["scene_id_map"])
        assert scene_id_map["demo_1"] == "scene_accept"


def test_filter_valid_source_reports_missing_metadata_without_crashing(tmp_path: Path):
    source_path = tmp_path / "success_demos.hdf5"
    filtered_path = tmp_path / "filtered_source.hdf5"

    with h5py.File(source_path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["env_args"] = json.dumps({"env_name": "Stack_D0"})
        data_group.attrs["total"] = 120
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 8 + [1] * 32)
        _write_demo_group(data_group, "demo_1", grasp_signal=None)
        _write_demo_group(data_group, "demo_2", grasp_signal=[0] * 40)

    report = augment_success_demos.filter_valid_source(
        str(source_path),
        "stack",
        str(filtered_path),
    )

    assert report["valid_count"] == 1
    assert report["rejection_counts"]["missing_datagen_info"] == 1
    assert report["rejection_counts"]["no_grasp_transition"] == 1
    assert filtered_path.exists()
    assert (tmp_path / "filtered_source_report.json").exists()

    with h5py.File(filtered_path, "r") as handle:
        assert sorted(handle["data"].keys()) == ["demo_0"]
        datagen_info = handle["data/demo_0/datagen_info"]
        assert datagen_info.attrs["env_interface_name"] == "MG_Stack"
        assert datagen_info.attrs["env_interface_type"] == "robosuite"


def test_ensure_env_interface_attrs_patches_existing_dataset(tmp_path: Path):
    dataset_path = tmp_path / "success_demos.hdf5"

    with h5py.File(dataset_path, "w") as handle:
        data_group = handle.create_group("data")
        _write_demo_group(data_group, "demo_0", grasp_signal=[0] * 8 + [1] * 32)
        _write_demo_group(data_group, "demo_1", grasp_signal=[0] * 8 + [1] * 32)
        data_group["demo_1"]["datagen_info"].attrs["env_interface_name"] = "MG_Stack"
        data_group["demo_1"]["datagen_info"].attrs["env_interface_type"] = "robosuite"

    report = ensure_env_interface_attrs(dataset_path, task_name="stack")

    assert report["demo_count"] == 2
    assert report["missing_datagen_info_count"] == 0
    assert report["updated_count"] == 1

    with h5py.File(dataset_path, "r") as handle:
        for demo_key in ("demo_0", "demo_1"):
            datagen_info = handle[f"data/{demo_key}/datagen_info"]
            assert datagen_info.attrs["env_interface_name"] == "MG_Stack"
            assert datagen_info.attrs["env_interface_type"] == "robosuite"
