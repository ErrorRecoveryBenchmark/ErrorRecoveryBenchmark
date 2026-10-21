import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from error_benchmark.scripts.pipeline import generate_training_scenes as generator
from error_benchmark.scripts.verification import audit_dataset_qualification as audit


def _valid_case(tmp_path: Path, *, seq=3, point_index=4):
    task = "stack"
    subtype = "collision_empty_D0"
    clean_ref = "demo_stack_001"
    injection_step = 42
    hash_input = f"train_{task}_{subtype}_{seq:03d}_{clean_ref}_{injection_step}"
    suffix = hashlib.sha1(hash_input.encode()).hexdigest()[:8]
    scene_id = f"v5_train_{subtype}_{seq:03d}_{suffix}"
    source_path = tmp_path / task / "scenes" / f"{scene_id}.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audit_version": audit.AUDIT_VERSION,
        "dataset_type": "training",
        "item_id": f"training:{task}/scenes/{scene_id}",
        "scene_id": scene_id,
        "source_path": str(source_path),
        "task": task,
        "subtype": subtype,
        "canonical_eligible": True,
    }
    meta = {
        "scene_id": scene_id,
        "dataset": {"trajectory_id": clean_ref},
        "replay": {"injection_frame": injection_step},
        "error_spec": {
            "source_trajectory": clean_ref,
            "source_frame": injection_step,
            "apply": {"t": injection_step},
        },
        "labels": {
            "subtype_id": subtype,
            "training_seq": seq,
            "generation_mode": "controlled_magnitude",
            "training_point_index": point_index,
            "magnitude_fraction": point_index / 9,
            "scenes_per_subtype": 10,
            "injection_seed": 123,
            "injection_attempt": 2,
            "generator_version": "v5.0_training_v3",
            "generator_sha256": "sha256:" + "a" * 64,
            "config_sha256": "sha256:" + "b" * 64,
        },
        "clean_trajectory_ref": clean_ref,
        "injection_step": injection_step,
    }
    return row, meta


def test_explicit_consistent_controlled_magnitude_evidence_passes(tmp_path):
    row, meta = _valid_case(tmp_path)
    result = audit.evaluate_training_generation(row, meta, {row["scene_id"]})
    assert result["generation_conformance"] == "pass"
    assert result["failure_reasons"] == []
    assert result["unverifiable_reasons"] == []


def test_legacy_scene_without_magnitude_evidence_is_unverifiable(tmp_path):
    row, meta = _valid_case(tmp_path)
    for field in (
        "generation_mode",
        "training_point_index",
        "magnitude_fraction",
        "scenes_per_subtype",
        "injection_seed",
        "injection_attempt",
        "generator_version",
        "generator_sha256",
        "config_sha256",
    ):
        meta["labels"].pop(field)
    result = audit.evaluate_training_generation(row, meta, {row["scene_id"]})
    assert result["generation_conformance"] == "unverifiable"
    assert "missing_magnitude_fraction" in result["unverifiable_reasons"]


def test_inconsistent_fraction_is_a_hard_failure(tmp_path):
    row, meta = _valid_case(tmp_path, point_index=4)
    meta["labels"]["magnitude_fraction"] = 0.9
    result = audit.evaluate_training_generation(row, meta, {row["scene_id"]})
    assert result["generation_conformance"] == "fail"
    assert "magnitude_fraction_mismatch" in result["failure_reasons"]


def test_coverage_repair_is_not_controlled_magnitude(tmp_path):
    row, meta = _valid_case(tmp_path)
    meta["labels"]["coverage_repair"] = True
    result = audit.evaluate_training_generation(row, meta, {row["scene_id"]})
    assert result["generation_conformance"] == "fail"
    assert "coverage_repair_uncontrolled" in result["failure_reasons"]


def test_scene_id_hash_must_match_generator_contract(tmp_path):
    row, meta = _valid_case(tmp_path)
    bad_id = row["scene_id"][:-8] + "deadbeef"
    row["scene_id"] = bad_id
    row["source_path"] = str(Path(row["source_path"]).with_name(f"{bad_id}.json"))
    meta["scene_id"] = bad_id
    result = audit.evaluate_training_generation(row, meta, {bad_id})
    assert result["generation_conformance"] == "fail"
    assert "scene_id_hash_mismatch" in result["failure_reasons"]


def test_filter_is_logical_and_reconciles_all_quality_eligible_rows(tmp_path):
    report_dir = tmp_path / "report"
    training_root = tmp_path / "training"
    output_dir = report_dir / "training_filter"
    report_dir.mkdir()
    row, meta = _valid_case(training_root)
    Path(row["source_path"]).write_text(json.dumps(meta))
    task_meta = training_root / "stack" / "meta.json"
    task_meta.write_text(json.dumps({"scenes": [{"scene_id": row["scene_id"]}]}))
    excluded = dict(row)
    excluded.update({"item_id": "training:excluded", "canonical_eligible": False})
    audit._write_jsonl(report_dir / "training.jsonl", [row, excluded])

    summary = audit.build_training_filter(report_dir, training_root, output_dir)
    assert summary["source"] == 2
    assert summary["logically_excluded"] == 1
    assert summary["quality_eligible"] == 1
    assert summary["generation_pass"] == 1
    assert audit.check_training_filter(output_dir, 2, 1) == 0
    assert Path(row["source_path"]).exists()


def test_future_generator_records_fraction_separately_from_success_sequence():
    scene = SimpleNamespace(labels={})
    generator._record_generation_provenance(
        scene,
        mode="controlled_magnitude",
        point_index=7,
        fraction=7 / 9,
        scenes_per_subtype=10,
        base_seed=42,
        config={"error_skills": {}},
    )
    assert scene.labels["training_point_index"] == 7
    assert scene.labels["magnitude_fraction"] == 7 / 9
    assert scene.labels["generation_mode"] == "controlled_magnitude"
    assert scene.labels["generator_sha256"].startswith("sha256:")


def test_future_generator_records_retry_seed_and_attempt():
    class Engine:
        def execute(self, trajectory, opportunity, skill, rng):
            return SimpleNamespace(labels={})

    opportunity = SimpleNamespace(subtype_id=lambda: "collision_empty_D0")
    scene = generator._try_inject(
        Engine(), object(), opportunity, object(), base_seed=42, point_idx=3,
        fraction=3 / 9,
    )
    assert isinstance(scene.labels["injection_seed"], int)
    assert scene.labels["injection_attempt"] == 0
