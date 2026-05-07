#!/usr/bin/env python
"""Unit tests for combining human recovery demos with MimicGen success sources."""

import json
from pathlib import Path

import h5py
import numpy as np

from error_benchmark.scripts.mimicgen import augment_success_demos
from error_benchmark.scripts.mimicgen import build_combined_success_source as builder


def _write_source_demo(data_group: h5py.Group, demo_key: str, steps: int, origin: str = ""):
    group = data_group.create_group(demo_key)
    group.attrs["num_samples"] = steps
    if origin:
        group.attrs["demo_origin"] = origin
    group.create_dataset("actions", data=np.zeros((steps, 7), dtype=np.float64))
    group.create_dataset("states", data=np.zeros((steps, 8), dtype=np.float64))
    group.create_group("obs")
    datagen_info = group.create_group("datagen_info")
    datagen_info.create_dataset("eef_pose", data=np.zeros((steps, 4, 4), dtype=np.float64))


def test_load_success_recovery_demo_accepts_success_and_trims_states(tmp_path: Path):
    npz_path = tmp_path / "recovery_demo.npz"
    np.savez_compressed(
        npz_path,
        success=np.bool_(True),
        validation_passed=np.bool_(True),
        replay_passed=np.bool_(True),
        scene_id=np.array("scene_npz"),
        actions=np.ones((3, 7), dtype=np.float64),
        states=np.ones((4, 8), dtype=np.float64),
    )
    entry = builder.RecoveryEntry(
        npz_path=npz_path,
        task_name="coffee",
        demo_id="recovery_demo",
        scene_id="scene_manifest",
    )

    demo, rejection = builder.load_success_recovery_demo(entry)

    assert rejection is None
    assert demo["states"].shape[0] == 3
    assert demo["scene_id"] == "scene_manifest"
    assert demo["metadata"]["demo_origin"] == "human_recovery"


def test_load_success_recovery_demo_rejects_failed_npz(tmp_path: Path):
    npz_path = tmp_path / "failed_demo.npz"
    np.savez_compressed(
        npz_path,
        success=np.bool_(False),
        actions=np.ones((3, 7), dtype=np.float64),
        states=np.ones((3, 8), dtype=np.float64),
    )
    entry = builder.RecoveryEntry(
        npz_path=npz_path,
        task_name="coffee",
        demo_id="failed_demo",
    )

    demo, rejection = builder.load_success_recovery_demo(entry)

    assert demo is None
    assert rejection["reason"] == "not_success"


def test_merge_sources_appends_human_demos_without_modifying_existing(tmp_path: Path):
    existing_path = tmp_path / "success_demos.hdf5"
    human_path = tmp_path / "human_filtered.hdf5"
    output_path = tmp_path / "success_demos_combined.hdf5"

    with h5py.File(existing_path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["total"] = 5
        data_group.attrs["scene_id_map"] = json.dumps({"demo_0": "existing_scene"})
        _write_source_demo(data_group, "demo_0", 5)

    with h5py.File(human_path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs["total"] = 4
        data_group.attrs["scene_id_map"] = json.dumps({"demo_0": "human_scene"})
        _write_source_demo(data_group, "demo_0", 4, origin="human_recovery")
        data_group["demo_0"].attrs["scene_id"] = "human_scene"

    report = builder.merge_sources(existing_path, human_path, output_path)

    assert report["existing_count"] == 1
    assert report["human_merged"] == 1
    assert report["total_count"] == 2

    with h5py.File(existing_path, "r") as handle:
        assert "demo_origin" not in handle["data/demo_0"].attrs

    with h5py.File(output_path, "r") as handle:
        data_group = handle["data"]
        assert sorted(data_group.keys()) == ["demo_0", "demo_1"]
        assert data_group.attrs["total"] == 9
        assert data_group["demo_0"].attrs["demo_origin"] == "mimicgen_success"
        assert data_group["demo_1"].attrs["demo_origin"] == "human_recovery"
        scene_id_map = json.loads(data_group.attrs["scene_id_map"])
        assert scene_id_map["demo_1"] == "human_scene"


def test_resolve_source_dataset_prefers_combined_when_present(tmp_path: Path):
    task_dir = tmp_path / "coffee"
    task_dir.mkdir()
    (task_dir / "success_demos.hdf5").touch()
    combined = task_dir / "success_demos_combined.hdf5"
    combined.touch()

    resolved = augment_success_demos.resolve_source_dataset(str(tmp_path), "coffee")

    assert resolved == str(combined)
    assert augment_success_demos.resolve_source_dataset(
        str(tmp_path),
        "coffee",
        "success_demos.hdf5",
    ) == str(task_dir / "success_demos.hdf5")
