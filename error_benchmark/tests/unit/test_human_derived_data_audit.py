import json
from pathlib import Path

import h5py
import numpy as np

from error_benchmark.scripts.verification import audit_human_derived_data as audit


def test_origin_classification_distinguishes_normal_and_recovery():
    recovery = {"r": [{"item_id": "human:1", "qualified": True}]}
    assert audit.classify_origin("n", {"n": ["core:demo_0"]}, recovery)[0] == "normal_mimicgen"
    assert audit.classify_origin("r", {"n": ["core:demo_0"]}, recovery)[0] == "human_recovery"
    assert audit.classify_origin("x", {}, recovery)[0] == "unmatched"


def test_recovery_training_requires_strictly_qualified_source():
    rows = [{"task": "coffee", "action_hash": "r", "structural_qualified": True}]
    refs = {task: {} for task in audit.TASKS}
    recovery = {task: {} for task in audit.TASKS}
    recovery["coffee"] = {"r": [{"item_id": "human:1", "qualified": False}]}
    audit.apply_origin_and_usability(rows, refs, recovery)
    assert rows[0]["human_origin_verified"] is True
    assert rows[0]["usability"]["recovery_policy_training"] is False
    recovery["coffee"]["r"][0]["qualified"] = True
    audit.apply_origin_and_usability(rows, refs, recovery)
    assert rows[0]["usability"]["recovery_policy_training"] is True


def test_logical_hdf5_duplicates_have_one_primary():
    rows = [
        {"item_id": "b", "task": "threading", "trajectory_hash": "same"},
        {"item_id": "a", "task": "threading", "trajectory_hash": "same"},
    ]
    audit.mark_logical_duplicates(rows)
    primary = [row for row in rows if not row["logical_duplicate"]]
    assert [row["item_id"] for row in primary] == ["a"]
    assert all(row["logical_primary_id"] == "a" for row in rows)


def test_training_lineage_needs_all_four_gates():
    scene = {"qualified": True}
    conformance = {"generation_conformance": "pass"}
    clean = {
        "human_origin_verified": True,
        "usability": {"error_scene_generation_source": True},
    }
    assert audit.training_lineage_decision(scene, conformance, clean) == (True, [])
    conformance["generation_conformance"] = "unverifiable"
    usable, reasons = audit.training_lineage_decision(scene, conformance, clean)
    assert usable is False
    assert "generation_conformance_unverifiable" in reasons


def test_check_report_reconciles_claimed_count(tmp_path: Path):
    rows = {
        "lerobot": [
            {
                "item_id": "lerobot:coffee:0",
                "dataset_family": "claimed_humandemo",
            }
        ],
        "hdf5": [],
        "clean": [],
        "training_lineage": [],
    }
    for name, values in rows.items():
        audit._write_jsonl(tmp_path / f"{name}.jsonl", values)
    summary = {
        "claimed_humandemo": 1,
        "stages": {
            "lerobot": {"items": 1},
            "hdf5": {"items": 0},
            "clean_trajectory": {"items": 0},
            "training_scene": {"items": 0},
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit.check_report(tmp_path, 1) == 0


def test_dataset_names_are_only_claims(tmp_path: Path):
    root = tmp_path / "coffee_humandemo"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    datasets = audit.discover_lerobot_datasets(tmp_path)
    assert datasets[0]["family"] == "claimed_humandemo"
    assert "origin" not in datasets[0]


def test_hdf5_audit_uses_existing_env_parser(tmp_path: Path):
    path = tmp_path / "coffee_humandemo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Coffee", "env_kwargs": {"robots": ["Panda"]}}
        )
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 1
        demo.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        demo.create_dataset("rewards", data=np.ones(1, dtype=np.float32))
        demo.create_dataset("dones", data=np.ones(1, dtype=bool))
        obs = demo.create_group("obs")
        obs.create_dataset("state", data=np.zeros((1, 8), dtype=np.float32))
        for image_key in ("agentview_image", "robot0_eye_in_hand_image"):
            obs.create_dataset(image_key, data=np.zeros((1, 84, 84, 3), dtype=np.uint8))
    rows = audit.audit_hdf5_files([tmp_path], [])
    assert len(rows) == 1
    assert "missing_env_args" not in rows[0]["failure_reasons"]
