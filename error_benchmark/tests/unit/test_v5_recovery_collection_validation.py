#!/usr/bin/env python
"""Unit tests for recovery collection-time validation helpers."""

from pathlib import Path

import numpy as np

from error_benchmark.framework.recovery_collection_validation import (
    RecoveryDemoValidator,
    dedupe_demo_entries,
    determine_counts_toward_target,
    get_next_demo_attempt_index,
    load_validation_records,
    normalize_quota_rule,
    recompute_collected_counts,
    recompute_collected_scene_ids,
    save_validation_report,
    upsert_demo_entry,
    upsert_validation_record,
)
from error_benchmark.framework.recovery_types import RecoveryDemo


class FakeEnvWrapper:
    def __init__(self, success_after_steps=0):
        self.success_after_steps = success_after_steps
        self.current_state = None
        self.step_count = 0

    def set_sim_state_flat(self, state):
        self.current_state = np.asarray(state).copy()
        self.step_count = 0

    def forward(self):
        return None

    def step(self, action):
        self.step_count += 1
        return {}, 0.0, False, {}

    def check_success(self):
        return self.step_count >= self.success_after_steps


class FakeAugmentedDemo:
    def __init__(self, augmented_id="aug_ok", num_steps=12, metadata=None):
        self.augmented_id = augmented_id
        self.num_steps = num_steps
        self.success = True
        self.actions = np.zeros((num_steps, 7))
        self.metadata = metadata or {}


class FakeAugmenter:
    def __init__(self, env_wrapper, task_config, aug_config, rng):
        self.env_wrapper = env_wrapper

    def _replay_with_warping(self, demo, scene):
        scene_id = scene.get("scene_id")
        if scene_id == "scene_success":
            return FakeAugmentedDemo(
                augmented_id="aug_scene_success",
                metadata={
                    "base_scene_id": scene_id,
                    "variant_id": "scene_success__variant_0000",
                    "variant_index": 0,
                    "randomized_objects": ["cubeA"],
                    "anchored_objects": ["cubeB"],
                },
            )
        if scene_id == "scene_error":
            raise ValueError("bad source sequence")
        return None


def _make_demo(success=True):
    return RecoveryDemo(
        demo_id="recovery_stack_test_0000",
        task_name="stack",
        error_name="collision_empty",
        degree="D0",
        scene_id="scene_origin",
        success=success,
        num_steps=3,
        actions=np.zeros((3, 7)),
        states=np.zeros((4, 5)),
        subtasks=[],
    )


def test_dedupe_demo_entries_keeps_latest_record():
    demos = [
        {"demo_id": "demo_0", "success": False, "subtype_id": "a_D0"},
        {"demo_id": "demo_1", "success": True, "subtype_id": "a_D0"},
        {"demo_id": "demo_0", "success": True, "subtype_id": "a_D0"},
    ]
    deduped, removed = dedupe_demo_entries(demos)
    assert removed == 1
    assert len(deduped) == 2
    assert [d for d in deduped if d["demo_id"] == "demo_0"][0]["success"] is True


def test_recompute_collected_counts_uses_quota_annotations():
    demos = [
        {"demo_id": "demo_0", "success": False, "subtype_id": "a_D0"},
        {
            "demo_id": "demo_1",
            "success": True,
            "subtype_id": "a_D0",
            "counts_toward_target": True,
        },
        {
            "demo_id": "demo_2",
            "success": True,
            "subtype_id": "b_D0",
            "counts_toward_target": False,
        },
    ]
    counts = recompute_collected_counts(
        demos,
        {"a_D0": 3, "b_D0": 2, "c_D0": 1},
        quota_rule="augmentable_only",
    )
    assert counts == {"a_D0": 1, "b_D0": 0, "c_D0": 0}


def test_recompute_collected_counts_legacy_entries_require_validation_under_new_quota():
    demos = [
        {"demo_id": "demo_0", "success": True, "subtype_id": "a_D0"},
        {
            "demo_id": "demo_1",
            "success": True,
            "subtype_id": "a_D0",
            "collection_validation": {
                "summary": {"replay_success": True, "augmentable": True},
            },
        },
        {
            "demo_id": "demo_2",
            "success": True,
            "subtype_id": "b_D0",
            "collection_validation": {
                "summary": {"replay_success": True, "augmentable": False},
            },
        },
    ]
    counts = recompute_collected_counts(
        demos,
        {"a_D0": 3, "b_D0": 2},
        quota_rule="augmentable_only",
    )
    assert counts == {"a_D0": 1, "b_D0": 0}


def test_recompute_collected_scene_ids_dedupes_by_scene():
    demos = [
        {
            "demo_id": "demo_0",
            "success": True,
            "subtype_id": "a_D0",
            "scene_id": "scene_0",
            "counts_toward_target": True,
        },
        {
            "demo_id": "demo_1",
            "success": True,
            "subtype_id": "a_D0",
            "scene_id": "scene_0",
            "counts_toward_target": True,
        },
        {
            "demo_id": "demo_2",
            "success": True,
            "subtype_id": "a_D0",
            "scene_id": "scene_1",
            "counts_toward_target": True,
        },
    ]
    covered = recompute_collected_scene_ids(
        demos,
        {"a_D0": ["scene_0", "scene_1", "scene_2"]},
        quota_rule="success_only",
    )
    assert covered == {"a_D0": ["scene_0", "scene_1"]}


def test_determine_counts_toward_target_supports_multiple_quota_rules():
    entry = {
        "demo_id": "demo_0",
        "success": True,
        "collection_validation": {
            "summary": {"replay_success": True, "augmentable": False},
        },
    }

    assert normalize_quota_rule("Replay_Only") == "replay_only"
    assert determine_counts_toward_target(entry, "success_only") == (True, "accepted")
    assert determine_counts_toward_target(entry, "replay_only") == (True, "accepted")
    assert determine_counts_toward_target(entry, "augmentable_only") == (
        False,
        "non_augmentable",
    )


def test_determine_counts_toward_target_replay_and_augmentable_requires_both():
    entry = {
        "demo_id": "demo_0",
        "success": True,
        "collection_validation": {
            "summary": {"replay_success": True, "augmentable": True},
        },
    }
    assert determine_counts_toward_target(entry, "replay_and_augmentable") == (
        True,
        "accepted",
    )

    entry["collection_validation"]["summary"]["replay_success"] = False
    assert determine_counts_toward_target(entry, "replay_and_augmentable") == (
        False,
        "replay_failed",
    )


def test_get_next_demo_attempt_index_uses_highest_existing_attempt():
    demos = [
        {"demo_id": "recovery_stack_collision_empty_D0_0000", "subtype_id": "collision_empty_D0"},
        {"demo_id": "recovery_stack_collision_empty_D0_0002", "subtype_id": "collision_empty_D0"},
        {"demo_id": "recovery_stack_collision_empty_D1_0007", "subtype_id": "collision_empty_D1"},
        {"demo_id": "legacy_demo_name", "subtype_id": "collision_empty_D0"},
    ]

    assert get_next_demo_attempt_index(demos, "stack", "collision_empty_D0") == 3
    assert get_next_demo_attempt_index(demos, "stack", "collision_empty_D1") == 8
    assert get_next_demo_attempt_index(demos, "stack", "missing_D0") == 0


def test_upsert_demo_entry_replaces_existing_demo_id():
    demos = [{"demo_id": "demo_0", "success": False}]
    upsert_demo_entry(demos, {"demo_id": "demo_0", "success": True})
    assert demos == [{"demo_id": "demo_0", "success": True}]


def test_validator_action_replay_marks_success_match():
    validator = RecoveryDemoValidator(
        env_wrapper=FakeEnvWrapper(success_after_steps=3),
        task_config={},
        aug_config={},
        scenes_dir=Path("."),
        validation_config={
            "check_action_replay": True,
            "check_scene_augmentation": False,
        },
    )
    record = validator.validate_demo(_make_demo(success=True))
    assert record["action_replay"]["status"] == "ok"
    assert record["action_replay"]["replay_success"] is True
    assert record["action_replay"]["success_match"] is True
    assert record["summary"]["replay_success"] is True


def test_validator_scene_augmentation_reports_augmentability():
    scenes = [
        {"scene_id": "scene_success", "_npz_path": "dummy_success.npz"},
        {"scene_id": "scene_fail", "_npz_path": "dummy_fail.npz"},
        {"scene_id": "scene_error", "_npz_path": "dummy_error.npz"},
    ]

    validator = RecoveryDemoValidator(
        env_wrapper=FakeEnvWrapper(success_after_steps=0),
        task_config={},
        aug_config={},
        scenes_dir=Path("."),
        validation_config={
            "check_action_replay": False,
            "check_scene_augmentation": True,
            "max_target_scenes": 3,
        },
        augmenter_factory=FakeAugmenter,
        scenes_loader=lambda scenes_dir, subtype_id: scenes,
    )
    demo = _make_demo(success=True)
    demo.subtasks = [object()]
    record = validator.validate_demo(demo)

    scene_aug = record["scene_augmentation"]
    assert scene_aug["status"] == "ok"
    assert scene_aug["tested_scene_count"] == 3
    assert scene_aug["tested_variant_count"] == 3
    assert scene_aug["success_count"] == 1
    assert scene_aug["is_augmentable"] is True
    assert scene_aug["successful_scene_ids"] == ["scene_success"]
    assert scene_aug["successful_variant_ids"] == ["scene_success__variant_0000"]
    assert scene_aug["failure_reason_counts"]["scene_replay_not_success_or_empty"] == 1
    assert scene_aug["failure_reason_counts"]["exception_ValueError"] == 1
    success_result = next(
        result for result in scene_aug["scene_results"]
        if result["scene_id"] == "scene_success"
    )
    assert success_result["variant_id"] == "scene_success__variant_0000"
    assert success_result["randomized_objects"] == ["cubeA"]
    assert success_result["anchored_objects"] == ["cubeB"]
    assert record["summary"]["augmentable"] is True
    assert record["summary"]["augment_tested_variant_count"] == 3


def test_validation_report_roundtrip(tmp_path):
    report_path = tmp_path / "human_demo_test_report.json"
    records = []
    record = {
        "demo_id": "demo_0",
        "summary": {
            "replay_success": True,
            "augmentable": False,
            "augment_success_count": 0,
            "augment_tested_scene_count": 2,
        },
        "scene_augmentation": {
            "tested_scene_count": 2,
            "success_count": 0,
            "failure_reason_counts": {
                "scene_replay_not_success_or_empty": 2,
            },
        },
    }
    upsert_validation_record(records, record)
    save_validation_report(report_path, "stack", records, {"enabled": True})

    loaded = load_validation_records(report_path)
    assert loaded == [record]
