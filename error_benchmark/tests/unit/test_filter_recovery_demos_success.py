import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "filter_recovery_demos_success.py"
SPEC = importlib.util.spec_from_file_location("filter_recovery_demos_success", SCRIPT_PATH)
filter_demos = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(filter_demos)


def _row(task, subtype, demo_id, source_path, selected=True):
    checks = {gate: selected for gate in filter_demos.GATES}
    return {
        "item_id": f"human:{demo_id}",
        "task": task,
        "subtype": subtype,
        "demo_id": demo_id,
        "source_path": source_path,
        "training_success_qualified": True,
        "checks": checks,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture_tree(tmp_path):
    source = tmp_path / "source"
    manifests = tmp_path / "manifests"
    task = "stack"
    subtype = "position_error_D0"
    demo_id = "recovery_stack_position_error_D0_0000"
    rejected_id = "recovery_stack_position_error_D0_0001"
    source_dir = source / task / subtype
    source_dir.mkdir(parents=True)
    (source_dir / f"{demo_id}.npz").write_bytes(b"selected")
    (source_dir / f"{rejected_id}.npz").write_bytes(b"rejected")
    manifest_dir = manifests / task
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "v1.0",
                "task": task,
                "status": {subtype: {"collected": 2, "augmentable": 2}},
                "demos": [
                    {"demo_id": demo_id, "subtype_id": subtype, "success": True},
                    {"demo_id": rejected_id, "subtype_id": subtype, "success": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _row(task, subtype, demo_id, f"/obsolete/server/{demo_id}.npz"),
        _row(task, subtype, rejected_id, f"/obsolete/server/{rejected_id}.npz", False),
        _row("coffee", "object_fell_D1", "no_source", "/obsolete/server/no_source.npz", False),
    ]
    audit = tmp_path / "human.jsonl"
    _write_jsonl(audit, rows)
    return audit, source, manifests, rows, demo_id, rejected_id


def test_copy_filters_exact_gates_and_manifest_with_safe_local_path(tmp_path, capsys):
    audit, source, manifests, _, demo_id, rejected_id = _fixture_tree(tmp_path)
    output = tmp_path / "recovery_demos_success"

    assert filter_demos.main(
        [
            "--audit", str(audit),
            "--source-root", str(source),
            "--manifest-root", str(manifests),
            "--output", str(output),
            "--mode", "copy",
            "--check",
        ]
    ) == 0

    staged = output / "stack" / "position_error_D0"
    assert (staged / f"{demo_id}.npz").read_bytes() == b"selected"
    assert not (staged / f"{rejected_id}.npz").exists()
    manifest = json.loads((output / "stack" / "manifest.json").read_text())
    assert [demo["demo_id"] for demo in manifest["demos"]] == [demo_id]
    assert manifest["demos"][0]["npz_path"] == f"position_error_D0/{demo_id}.npz"
    assert manifest["status"]["position_error_D0"]["collected"] == 1
    selection = json.loads((output / "selection_manifest.json").read_text())
    assert selection["selected"] == 1
    assert selection["complete"] is True
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["total_cells"] == 2
    assert coverage["no_source_cells"] == 1
    assert "CHECK=PASS" in capsys.readouterr().out

    # An identical rerun is a supported, safe no-op.
    assert filter_demos.main(
        [
            "--audit", str(audit),
            "--source-root", str(source),
            "--manifest-root", str(manifests),
            "--output", str(output),
            "--mode", "copy",
        ]
    ) == 0


def test_selection_requires_literal_json_true_for_every_gate(tmp_path):
    row = _row("stack", "position_error_D0", "demo", "/old/demo.npz")
    row["checks"][filter_demos.GATES[-1]] = 1
    assert filter_demos.build_selection([row], tmp_path) == []


def test_manifest_duplicate_keeps_the_unique_successful_record(tmp_path):
    audit, source, manifests, _, demo_id, _ = _fixture_tree(tmp_path)
    manifest_path = manifests / "stack" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["demos"].insert(
        0,
        {
            "demo_id": demo_id,
            "subtype_id": "position_error_D0",
            "success": False,
            "num_steps": 500,
            "subtasks": [],
        },
    )
    manifest["demos"][1]["num_steps"] = 172
    manifest["demos"][1]["subtasks"] = [{"name": "pre_grasp"}, {"name": "post_grasp"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "recovery_demos_success"
    assert filter_demos.main(
        [
            "--audit", str(audit),
            "--source-root", str(source),
            "--manifest-root", str(manifests),
            "--output", str(output),
            "--mode", "copy",
        ]
    ) == 0

    filtered = json.loads((output / "stack" / "manifest.json").read_text())
    assert len(filtered["demos"]) == 1
    assert filtered["demos"][0]["success"] is True
    assert filtered["demos"][0]["num_steps"] == 172
    assert len(filtered["demos"][0]["subtasks"]) == 2


def test_manifest_duplicate_rejects_multiple_successful_records(tmp_path):
    _, source, manifests, rows, demo_id, _ = _fixture_tree(tmp_path)
    manifest_path = manifests / "stack" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["demos"].insert(0, dict(manifest["demos"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selection = filter_demos.build_selection(rows, source)

    with pytest.raises(ValueError, match="ambiguously duplicated"):
        filter_demos._filtered_manifests(selection, manifests)


def test_refuses_unrelated_nonempty_output_and_unsafe_cell(tmp_path):
    audit, source, manifests, rows, *_ = _fixture_tree(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="unrelated non-empty"):
        filter_demos.main(
            [
                "--audit", str(audit),
                "--source-root", str(source),
                "--manifest-root", str(manifests),
                "--output", str(output),
            ]
        )
    unsafe = _row("../escape", "position_error_D0", "demo", "/old/demo.npz")
    with pytest.raises(ValueError, match="unsafe task"):
        filter_demos.build_selection([unsafe], source)


def test_coverage_counts_only_successful_manifest_entries_with_npz(tmp_path):
    audit, source, manifests, rows, demo_id, _ = _fixture_tree(tmp_path)
    augmented = tmp_path / "augmented"
    aug_dir = augmented / "stack" / "position_error_D0"
    aug_dir.mkdir(parents=True)
    (aug_dir / "aug_ok.npz").write_bytes(b"ok")
    (aug_dir / "aug_failed.npz").write_bytes(b"failed")
    (augmented / "stack" / "augmentation_manifest_worker0.json").write_text(
        json.dumps(
            {
                "task": "stack",
                "subtypes": {
                    "position_error_D0": {
                        "entries": [
                            {
                                "augmented_id": "aug_ok",
                                "task_name": "stack",
                                "subtype_id": "position_error_D0",
                                "success": True,
                                "npz_path": "/old/aug_ok.npz",
                            },
                            {
                                "augmented_id": "aug_failed",
                                "task_name": "stack",
                                "subtype_id": "position_error_D0",
                                "success": False,
                                "npz_path": "/old/aug_failed.npz",
                            },
                            {
                                "augmented_id": "aug_missing",
                                "task_name": "stack",
                                "subtype_id": "position_error_D0",
                                "success": True,
                                "npz_path": "/old/aug_missing.npz",
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "unused_staging"

    assert filter_demos.main(
        [
            "--audit", str(audit),
            "--source-root", str(source),
            "--manifest-root", str(manifests),
            "--output", str(output),
            "--augmented-root", str(augmented),
            "--target", "3",
            "--coverage-only",
        ]
    ) == 0

    coverage = json.loads((augmented / "coverage.json").read_text())
    stack = next(item for item in coverage["cells"] if item["task"] == "stack")
    assert stack == {
        "task": "stack",
        "subtype": "position_error_D0",
        "human": 1,
        "augmented": 1,
        "combined": 2,
        "target": 3,
        "deficit": 1,
        "reachable": True,
        "status": "needs_augmentation",
    }
    assert coverage["augmented"] == 1
    assert coverage["total_deficit"] == 4
    assert (augmented / "coverage.csv").is_file()
