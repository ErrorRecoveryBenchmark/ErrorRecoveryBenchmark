import json

import numpy as np

from error_benchmark.scripts.verification import audit_dataset_qualification as audit


def _scene_row(item_id, fingerprint, dataset_type="training", provenance_valid=True):
    row = audit._base_row(dataset_type, item_id, __file__, "stack")
    row.update({
        "scene_id": item_id,
        "subtype": "position_error_D0",
        "fingerprint": fingerprint,
        "provenance": {"valid": provenance_valid},
    })
    row["checks"] = {name: True for name in audit.SCENE_QUALITY_CHECKS}
    audit._recompute_scene_decision(row)
    return row


def _human_row(item_id, source_scene_id, qualified=True):
    row = audit._base_row("human", item_id, __file__, "stack")
    row.update({
        "source_scene_id": source_scene_id,
        "subtype": "position_error_D0",
    })
    row["checks"] = {name: qualified for name in audit.HUMAN_QUALITY_CHECKS}
    row["checks"].update({
        name: qualified for name in audit.HUMAN_TRAINING_SUCCESS_GATES
    })
    audit._recompute_human_decision(row)
    return row


def test_provenance_keeps_controlled_training_and_rejects_backfill():
    controlled = audit.classify_scene_provenance(
        "training",
        "v5_train_position_error_D0_009_deadbeef",
        {},
    )
    backfill = audit.classify_scene_provenance(
        "training",
        "v5_position_error_D0_deadbeef",
        {"backfill_source": "v5/stack/scenes/source.json"},
    )
    assert controlled == {
        "kind": "controlled_training",
        "valid": True,
        "sequence_index": 9,
    }
    assert backfill["kind"] == "validation_backfill"
    assert backfill["valid"] is False


def test_eval_provenance_is_distinct_from_training_provenance():
    evaluation = audit.classify_scene_provenance(
        "eval",
        "v5_position_error_D0_deadbeef",
        {},
    )
    controlled = audit.classify_scene_provenance(
        "eval",
        "v5_train_position_error_D0_000_deadbeef",
        {},
    )
    assert evaluation["kind"] == "validation_generation"
    assert evaluation["valid"] is True
    assert controlled["valid"] is False


def test_duplicate_fingerprint_fails_quality_for_every_duplicate():
    rows = [
        _scene_row("training:a", "sha256:same"),
        _scene_row("training:b", "sha256:same"),
    ]
    audit._mark_duplicate_fingerprints(rows)
    assert all(not row["qualified"] for row in rows)
    assert all("duplicate_fingerprint" in row["failure_reasons"] for row in rows)


def test_validation_pool_overlap_preserves_quality_but_blocks_eligibility():
    training = [_scene_row("training:a", "sha256:same")]
    evaluation = [_scene_row("eval:a", "sha256:same", "eval")]
    audit.finalize_scene_records(training, evaluation)
    assert training[0]["qualified"] is True
    assert training[0]["canonical_eligible"] is False
    assert "overlap_validation_pool" in training[0]["failure_reasons"]
    assert evaluation[0]["canonical_eligible"] is True


def test_provenance_is_selection_eligibility_not_scene_quality():
    row = _scene_row("training:a", "sha256:a", provenance_valid=False)
    assert row["qualified"] is True
    assert row["canonical_eligible"] is False


def test_human_qualification_is_conjunction_of_all_checks():
    row = _human_row("human:a", "scene-a")
    assert row["ground_truth_qualified"] is True
    row["checks"]["success_streak"] = False
    audit._recompute_human_decision(row)
    assert row["ground_truth_qualified"] is False


def test_training_success_is_separate_from_canonical_source_replay():
    row = _human_row("human:a", "scene-a")
    row["checks"]["success_streak"] = False
    audit._recompute_human_decision(row)
    assert row["ground_truth_qualified"] is False
    assert row["training_success_qualified"] is True


def test_training_success_funnel_counts_after_every_and():
    passed = _human_row("human:pass", "scene-a")
    fails_final_state = _human_row("human:final", "scene-b")
    fails_final_state["checks"]["saved_final_state_success"] = False
    audit._recompute_human_decision(fails_final_state)
    fails_release = _human_row("human:release", "scene-c")
    fails_release["checks"]["saved_start_release_lift"] = False
    audit._recompute_human_decision(fails_release)

    funnel = audit.build_human_training_success_funnel([
        passed,
        fails_final_state,
        fails_release,
    ])["overall"]

    assert funnel["total"] == 3
    assert funnel["state_eef_error"] == 3
    assert funnel["saved_final_state_success"] == 2
    assert funnel["saved_start_success_streak"] == 2
    assert funnel["saved_start_release_lift"] == 1


def test_human_lengths_match_collectors_post_action_state_layout():
    actions = np.zeros((4, 7))
    states = np.zeros((5, 32))
    eef_positions = np.zeros((5, 3))
    assert audit._trajectory_lengths_ok(actions, states, eef_positions, 4)
    assert not audit._trajectory_lengths_ok(actions, np.zeros((4, 32)), eef_positions, 4)
    assert not audit._trajectory_lengths_ok(actions, states, np.zeros((4, 3)), 4)


def test_human_source_prefers_scene_origin_with_legacy_fallback():
    assert audit._human_source_scene_id({
        "scene_origin": np.array("origin"),
        "scene_id": np.array("legacy"),
    }) == "origin"
    assert audit._human_source_scene_id({"scene_id": np.array("legacy")}) == "legacy"


def test_duplicate_human_demos_only_credit_one_source_scene():
    rows = [
        _human_row("human:b", "scene-a"),
        _human_row("human:a", "scene-a"),
        _human_row("human:c", "scene-b", qualified=False),
    ]
    audit.assign_coverage_credit(rows)
    credited = [row["item_id"] for row in rows if row["coverage_credit"]]
    assert credited == ["human:a"]
    duplicate = next(row for row in rows if row["item_id"] == "human:b")
    assert "duplicate_source_no_coverage" in duplicate["warnings"]


def test_summary_uses_unique_human_coverage_for_gap():
    training = [_scene_row("training:a", "sha256:t")]
    evaluation = [_scene_row("eval:a", "sha256:e", "eval")]
    human = [_human_row("human:a", "training:a")]
    audit.assign_coverage_credit(human)
    summary = audit.build_summary(training, evaluation, human)
    cell = "stack/position_error_D0"
    assert summary["gaps_to_ten_per_cell"]["training"][cell] == 9
    assert summary["gaps_to_ten_per_cell"]["eval"][cell] == 9
    assert summary["gaps_to_ten_per_cell"]["human"][cell] == 9


def test_human_summary_reports_unique_coverage_and_task_target():
    training = [_scene_row("training:a", "sha256:t")]
    human = [
        _human_row("human:a", "training:a"),
        _human_row("human:b", "training:a"),
    ]
    audit.assign_coverage_credit(human)
    summary = audit.build_human_summary(training, human)
    assert summary["candidates"] == 2
    assert summary["qualified"] == 2
    assert summary["coverage_credit"] == 1
    assert summary["target"] == 1360
    assert summary["by_task"]["stack"]["target"] == 240
    assert (
        summary["human_training_success_funnel"]["overall"][
            "saved_start_release_lift"
        ]
        == 2
    )


def test_discovery_keeps_unpaired_scene_as_failed_candidate(tmp_path):
    root = tmp_path / "stack" / "scenes"
    root.mkdir(parents=True)
    (root / "v5_position_error_D0_deadbeef.json").write_text("{}")
    candidates = audit.discover_scene_candidates(tmp_path, "eval")
    assert len(candidates) == 1
    assert candidates[0]["npz_path"].endswith("v5_position_error_D0_deadbeef.npz")


def test_historical_scene_id_loader_reads_nested_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "error_scenes": {
            "episodes": [
                {"scene_id": "scene-a"},
                {"nested": {"scene_id": "scene-b"}},
            ]
        }
    }))
    assert audit.load_historical_scene_ids(path) == {"scene-a", "scene-b"}


def test_dependency_jsonl_can_load_a_prior_audit_version(tmp_path):
    path = tmp_path / "training.jsonl"
    row = {"audit_version": "1.0", "item_id": "training:a"}
    path.write_text(json.dumps(row) + "\n")
    assert audit._load_jsonl(path) == []
    assert audit._load_jsonl(path, required_version=None) == [row]


def test_report_checker_reconciles_all_rows(tmp_path, capsys):
    rows = {
        "training": [_scene_row("training:a", "sha256:t")],
        "eval": [_scene_row("eval:a", "sha256:e", "eval")],
        "human": [_human_row("human:a", "training:a")],
    }
    for name, dataset_rows in rows.items():
        audit._write_jsonl(tmp_path / f"{name}.jsonl", dataset_rows)
    summary = audit.build_summary(rows["training"], rows["eval"], rows["human"])
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit.check_report(tmp_path, {name: 1 for name in rows}) == 0
    assert (
        "AUDIT REPORT COMPLETE training=1 eval=1 human=1 missing=0 duplicate_records=0"
        in capsys.readouterr().out
    )


def test_human_report_checker_reconciles_candidates_and_unique_credit(tmp_path, capsys):
    rows = [_human_row("human:a", "training:a")]
    audit.assign_coverage_credit(rows)
    audit._write_jsonl(tmp_path / "human.jsonl", rows)
    summary = audit.build_human_summary([_scene_row("training:a", "sha256:t")], rows)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit.check_human_report(tmp_path, 1) == 0
    assert (
        "HUMAN RECOVERY REPORT COMPLETE candidates=1 missing=0 duplicate_records=0"
        in capsys.readouterr().out
    )


# ── Harness robustness: a broken/missing SOURCE SCENE must not masquerade as
#    six independent demo failures (root cause of the 2026-07-23 0/1360 run). ──

_REPLAY_GATE_REASONS = {
    "action_replay_error",
    "final_eef_error",
    "success_streak_too_short",
    "release_lift_failed",
}


class _FakeEnv:
    """Duck-typed EnvWrapper stand-in; optionally raises when restoring the
    source scene state (identified by a sentinel fill value)."""

    def __init__(self, raise_on_source_restore=False):
        self.raise_on_source_restore = raise_on_source_restore

    def get_action_dim(self):
        return 7

    def set_sim_state_flat(self, state):
        state = np.asarray(state)
        if self.raise_on_source_restore and state.size and float(state.flat[0]) == 7.0:
            raise RuntimeError("simulated restore failure")

    def forward(self):
        pass

    def check_success(self):
        return False

    def get_eef_pos(self):
        return np.zeros(3)

    def step(self, action):
        pass

    def set_gripper_state(self, value, steps=1):
        pass

    def apply_eef_offset(self, offset, max_steps=1):
        pass


def _write_demo_npz(path, scene_id, steps=3):
    np.savez(
        path,
        actions=np.zeros((steps, 7), dtype=np.float64),
        states=np.zeros((steps + 1, 5), dtype=np.float64),
        eef_positions=np.zeros((steps + 1, 3), dtype=np.float64),
        task_name="stack",
        error_name="position_error",
        degree="D0",
        scene_origin=scene_id,
        demo_id=path.stem,
        num_steps=steps,
        success=True,
    )


def _run_human_candidate(tmp_path, monkeypatch, *, source_npz_exists, env):
    scene_id = "v5_train_position_error_D0_000_deadbeef"
    scene_npz = tmp_path / f"{scene_id}.npz"
    if source_npz_exists:
        np.savez(scene_npz, post_sim_state=np.full(4, 7.0))
    demo_npz = tmp_path / "recovery_stack_position_error_D0_0000.npz"
    _write_demo_npz(demo_npz, scene_id)
    monkeypatch.setattr(audit, "_WORKER_ENV", env)
    monkeypatch.setattr(audit, "_WORKER_ENV_ERROR", None)
    monkeypatch.setattr(audit, "_WORKER_TRAINING", {
        scene_id: {
            "scene_id": scene_id,
            "npz_path": str(scene_npz),
            "qualified": True,
            "canonical_eligible": True,
            "task": "stack",
            "subtype": "position_error_D0",
        }
    })
    return audit._audit_human_candidate({
        "item_id": f"human:stack/position_error_D0/{demo_npz.stem}",
        "npz_path": str(demo_npz),
        "task": "stack",
    })


def test_missing_source_npz_reports_single_root_cause(tmp_path, monkeypatch):
    row = _run_human_candidate(
        tmp_path, monkeypatch, source_npz_exists=False, env=_FakeEnv())
    reasons = set(row["failure_reasons"])
    assert "source_scene_npz_missing" in reasons
    assert "scene_already_successful" not in reasons
    assert "state_restore_error" not in reasons
    # the un-runnable replay gates must not fan out into four fake failures
    assert not (reasons & _REPLAY_GATE_REASONS), reasons
    assert "action_replay_not_attempted" in reasons
    # ...but the demo must still NOT qualify (unverifiable != pass)
    assert row["checks"]["action_replay"] is False
    assert row["checks"]["source_not_successful"] is False
    assert row["qualified"] is False


def test_restore_exception_is_not_labelled_scene_already_successful(tmp_path, monkeypatch):
    row = _run_human_candidate(
        tmp_path, monkeypatch, source_npz_exists=True,
        env=_FakeEnv(raise_on_source_restore=True))
    reasons = set(row["failure_reasons"])
    assert "state_restore_error" in reasons
    assert "scene_already_successful" not in reasons
    assert "action_replay_not_attempted" in reasons
    assert not (reasons & _REPLAY_GATE_REASONS), reasons
