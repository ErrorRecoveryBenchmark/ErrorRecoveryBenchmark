import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from error_benchmark.scripts.verification import audit_mimicgen_hdf5 as audit


ENV_ARGS = {
    "env_name": "Coffee_D0",
    "env_kwargs": {"robots": ["Panda"]},
}


def test_simulator_import_roots_are_bootstrapped():
    workspace = audit.PROJECT_ROOT / "shared" / "mimicgen_workspace"
    assert str(workspace / "robosuite") in audit.sys.path
    assert str(workspace / "mimicgen") in audit.sys.path


def test_audit_workers_must_be_positive():
    with pytest.raises(SystemExit, match="--workers must be at least 1"):
        audit.main(["--audit", ".", "--report-dir", ".", "--workers", "0"])


def _add_demo(data, key="demo_0", *, horizon=2, family="core", value=0.0):
    group = data.create_group(key)
    group.attrs["num_samples"] = horizon
    actions = np.full((horizon, 7), value, dtype=np.float32)
    group.create_dataset("actions", data=actions, compression="gzip")
    obs = group.create_group("obs")
    obs.create_dataset(
        "agentview_image", data=np.zeros((horizon, 84, 84, 3), dtype=np.uint8), compression="gzip"
    )
    obs.create_dataset(
        "robot0_eye_in_hand_image",
        data=np.zeros((horizon, 84, 84, 3), dtype=np.uint8),
        compression="gzip",
    )
    if family == "bc_rnn_humandemo":
        obs.create_dataset("state", data=np.zeros((horizon, 8), dtype=np.float32))
        # Converter contract (build_bc_rnn_humandemo_hdf5.py): rewards are ALL
        # ZEROS (robomimic placeholder, BC-RNN consumes no reward); terminal
        # evidence is dones[-1] == 1. A rewards>0 requirement mislabelled 100%
        # of legitimate converter output as unusable (2026-07-17 prune).
        group.create_dataset("rewards", data=np.zeros(2, dtype=np.float32))
        group.create_dataset("dones", data=np.array([0, 1], dtype=np.int64))
        group.attrs["source_dataset"] = "lerobot/coffee_humandemo"
    else:
        group.create_dataset("states", data=np.full((horizon, 5), value, dtype=np.float64))
        group.attrs["model_file"] = "<mujoco/>"
    return group


def _make_file(tmp_path: Path, family="core", demos=1) -> Path:
    directory = tmp_path / family
    directory.mkdir(parents=True)
    suffix = {
        "core": "coffee_d0.hdf5",
        "mixed_2000": "coffee_mixed_2000.hdf5",
        "bc_rnn_pi05": "coffee_pi05_v1.hdf5",
        "bc_rnn_humandemo": "coffee_humandemo.hdf5",
    }[family]
    path = directory / suffix
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        if family == "bc_rnn_pi05":
            data.attrs["bc_rnn_pi05_task"] = "coffee"
        if family == "bc_rnn_humandemo":
            data.attrs["humandemo_task"] = "coffee"
        for index in range(demos):
            _add_demo(data, f"demo_{index}", family=family, value=float(index))
        data.attrs["total"] = demos * 2
    return path


class _Replay:
    def __init__(self, result=True):
        self.result = result

    def check(self, _group):
        return self.result

    def close(self):
        pass


def test_core_demo_passes_structural_and_replay_checks(tmp_path):
    path = _make_file(tmp_path)
    rows, file_row = audit.audit_file(
        path, tmp_path, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    assert rows[0]["status"] == "pass"
    assert rows[0]["qualification_mode"] == "simulator_action_replay"
    assert file_row["file_errors"] == []


def test_nan_wrong_robot_and_length_are_concrete_failures(tmp_path):
    path = _make_file(tmp_path)
    with h5py.File(path, "r+") as handle:
        handle["data"].attrs["env_args"] = json.dumps(
            {"env_name": "Coffee_D0", "env_kwargs": {"robots": ["Sawyer"]}}
        )
        actions = handle["data/demo_0/actions"]
        actions[0, 0] = np.nan
        del handle["data/demo_0/obs/agentview_image"]
        handle["data/demo_0/obs"].create_dataset(
            "agentview_image", data=np.zeros((1, 84, 84, 3), dtype=np.uint8)
        )
    rows, _ = audit.audit_file(
        path, tmp_path, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    assert rows[0]["status"] == "fail"
    assert {"non_finite_actions", "robot_mismatch", "invalid_agentview_image_schema"} <= set(
        rows[0]["reasons"]
    )


def test_humandemo_pass_is_training_only(tmp_path):
    path = _make_file(tmp_path, "bc_rnn_humandemo")
    rows, _ = audit.audit_file(path, tmp_path, "bc_rnn_humandemo")
    assert rows[0]["status"] == "pass"
    assert rows[0]["qualification_mode"] == "bc_training_schema_and_terminal_evidence"


def test_humandemo_accepts_real_file_level_source_attribute(tmp_path):
    path = _make_file(tmp_path, "bc_rnn_humandemo")
    with h5py.File(path, "r+") as handle:
        del handle["data/demo_0"].attrs["source_dataset"]
        handle["data"].attrs["humandemo_source"] = "lerobot/coffee_humandemo"
    rows, _ = audit.audit_file(path, tmp_path, "bc_rnn_humandemo")
    assert rows[0]["status"] == "pass"


def test_humandemo_without_terminal_success_fails(tmp_path):
    path = _make_file(tmp_path, "bc_rnn_humandemo")
    with h5py.File(path, "r+") as handle:
        handle["data/demo_0/dones"][...] = 0
    rows, _ = audit.audit_file(path, tmp_path, "bc_rnn_humandemo")
    assert rows[0]["status"] == "fail"
    assert "missing_terminal_success_evidence" in rows[0]["reasons"]


def test_humandemo_all_zero_rewards_is_normal_converter_output(tmp_path):
    """Regression for the 2026-07-17 false-positive prune: converter-shaped
    episodes (zero rewards, dones[-1]=1) must PASS structural qualification."""
    path = _make_file(tmp_path, "bc_rnn_humandemo")
    with h5py.File(path, "r") as handle:
        assert not np.any(np.asarray(handle["data/demo_0/rewards"][()]) > 0)
    rows, _ = audit.audit_file(path, tmp_path, "bc_rnn_humandemo")
    assert rows[0]["status"] == "pass"
    assert "missing_terminal_success_evidence" not in rows[0]["reasons"]


def test_duplicate_content_within_file_fails_both(tmp_path):
    path = _make_file(tmp_path, demos=2)
    with h5py.File(path, "r+") as handle:
        handle["data/demo_1/actions"][...] = handle["data/demo_0/actions"][...]
        handle["data/demo_1/states"][...] = handle["data/demo_0/states"][...]
    rows, _ = audit.audit_file(
        path, tmp_path, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    assert [row["status"] for row in rows] == ["fail", "fail"]
    assert all("duplicate_within_file" in row["reasons"] for row in rows)


def test_derived_demo_resolves_to_qualified_source_fingerprint(tmp_path):
    core_path = _make_file(tmp_path)
    core_rows, _ = audit.audit_file(
        core_path, tmp_path, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    mixed_path = _make_file(tmp_path, "mixed_2000")
    source_index = {core_rows[0]["fingerprint"]: core_rows}
    rows, _ = audit.audit_file(
        mixed_path,
        tmp_path,
        "mixed_2000",
        source_fingerprints=source_index,
        replay_factory=None,
    )
    assert rows[0]["status"] == "pass"
    assert rows[0]["qualification_mode"] == "qualified_source_fingerprint"


def test_unresolved_derived_demo_without_replay_is_unverifiable(tmp_path):
    path = _make_file(tmp_path, "bc_rnn_pi05")
    rows, _ = audit.audit_file(path, tmp_path, "bc_rnn_pi05", replay_factory=None)
    assert rows[0]["status"] == "unverifiable"
    assert rows[0]["reasons"] == ["semantic_replay_not_run"]


def test_filtered_copy_preserves_keys_attrs_and_compression(tmp_path):
    path = _make_file(tmp_path, demos=2)
    rows, _ = audit.audit_file(
        path, tmp_path, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    temp = path.with_name("filtered.hdf5")
    audit._build_filtered_file(path, temp, [rows[1]])
    audit._verify_filtered_file(temp, [rows[1]])
    with h5py.File(temp, "r") as handle:
        assert audit._demo_keys(handle["data"]) == ["demo_1"]
        assert handle["data"].attrs["total"] == 2
        assert handle["data/demo_1/actions"].compression == "gzip"
        assert handle["data/demo_1"].attrs["model_file"] == "<mujoco/>"


def test_activation_failure_rolls_original_back(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    path = _make_file(root)
    rows, file_row = audit.audit_file(
        path, root, "core", replay_factory=lambda _path, _task: _Replay(True)
    )
    rows[0]["status"] = "fail"
    rows[0]["reasons"] = ["synthetic_failure"]
    report = tmp_path / "report"
    report.mkdir()
    audit._write_jsonl(report / "demos.jsonl", rows)
    audit._write_jsonl(report / "files.jsonl", [file_row])

    real_replace = audit.os.replace
    calls = 0

    def flaky_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("activation failed")
        return real_replace(source, destination)

    monkeypatch.setattr(audit.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="activation failed"):
        audit.clean_active(root, report)
    assert path.exists()
    assert not (root / ".quarantine_20260714" / "core" / path.name).exists()
