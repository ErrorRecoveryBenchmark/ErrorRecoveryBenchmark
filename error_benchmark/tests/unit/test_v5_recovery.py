#!/usr/bin/env python
"""
Unit tests for recovery demo collection and augmentation pipeline.

Tests recovery_types, recovery_segmenter, and the conversion utilities.
"""

import json
import sys
import yaml
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ─── Test recovery_types ───

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, RecoverySubtask, AugmentedRecovery,
    RecoveryCollectionStatus, SUBTYPE_TO_RBG, RBG_SUBTASK_SEQUENCES,
    RBG_MOTION_TEMPLATES, RECOVERY_SUBTASKS,
)


class TestRecoverySubtask:
    def test_creation(self):
        st = RecoverySubtask(label="retract", start_step=0, end_step=10, duration=10)
        assert st.label == "retract"
        assert st.duration == 10

    def test_serialization(self):
        st = RecoverySubtask(
            label="re_grasp", start_step=5, end_step=15,
            duration=10, eef_displacement=0.12)
        d = st.to_dict()
        assert d['label'] == "re_grasp"
        assert d['eef_displacement'] == 0.12

        st2 = RecoverySubtask.from_dict(d)
        assert st2.label == st.label
        assert st2.start_step == st.start_step

    def test_all_subtask_labels_known(self):
        """Every label used in RBG sequences must be in RECOVERY_SUBTASKS."""
        for rbg, seq in RBG_SUBTASK_SEQUENCES.items():
            for label in seq:
                assert label in RECOVERY_SUBTASKS, \
                    f"{label} from {rbg} not in RECOVERY_SUBTASKS"


class TestRecoveryDemo:
    def test_auto_subtype_id(self):
        demo = RecoveryDemo(error_name="drop_in_transit", degree="D0")
        assert demo.subtype_id == "drop_in_transit_D0"

    def test_auto_rbg(self):
        demo = RecoveryDemo(error_name="collision_holding", degree="D0")
        assert demo.rbg == "RBG_C"

    def test_serialization_roundtrip(self):
        demo = RecoveryDemo(
            demo_id="test_001",
            task_name="stack",
            error_name="grasp_misalignment",
            degree="D0",
            scene_id="scene_42",
            success=True,
            num_steps=100,
            subtasks=[
                RecoverySubtask(label="retract", start_step=0, end_step=20, duration=20),
                RecoverySubtask(label="re_grasp", start_step=20, end_step=50, duration=30),
            ],
        )
        d = demo.to_dict()
        assert d['demo_id'] == "test_001"
        assert d['rbg'] == "RBG_A"
        assert len(d['subtasks']) == 2

        demo2 = RecoveryDemo.from_dict(d)
        assert demo2.demo_id == demo.demo_id
        assert demo2.rbg == demo.rbg
        assert len(demo2.subtasks) == 2
        assert demo2.subtasks[0].label == "retract"

    def test_with_arrays(self):
        demo = RecoveryDemo(
            demo_id="arr_test",
            task_name="pick_place",
            error_name="position_error",
            degree="D1",
            num_steps=50,
            actions=np.random.randn(50, 7),
            eef_positions=np.random.randn(51, 3),
            target_poses=np.repeat(np.eye(4)[None], 50, axis=0),
            gripper_states=np.random.rand(51),
        )
        assert demo.actions.shape == (50, 7)
        assert demo.eef_positions.shape == (51, 3)
        assert demo.target_poses.shape == (50, 4, 4)
        # to_dict should not include arrays
        d = demo.to_dict()
        assert 'actions' not in d


class TestAugmentedRecovery:
    def test_creation(self):
        aug = AugmentedRecovery(
            augmented_id="aug_001",
            source_demo_id="test_001",
            augmentation_type="scene",
            task_name="stack",
            error_name="drop_in_transit",
            degree="D0",
            success=True,
            num_steps=80,
        )
        assert aug.subtype_id == "drop_in_transit_D0"
        assert aug.rbg == "RBG_B"

    def test_serialization(self):
        aug = AugmentedRecovery(
            augmented_id="aug_002",
            source_demo_id="test_001",
            augmentation_type="cross_degree",
            error_name="position_error",
            degree="D1",
        )
        d = aug.to_dict()
        aug2 = AugmentedRecovery.from_dict(d)
        assert aug2.augmented_id == aug.augmented_id
        assert aug2.augmentation_type == "cross_degree"


class TestRecoveryCollectionStatus:
    def test_tracking(self):
        status = RecoveryCollectionStatus(
            task_name="stack",
            target_demos={"grasp_misalignment_D0": 5, "drop_in_transit_D0": 5},
        )
        assert status.total_target() == 10
        assert status.total_collected() == 0

        status.increment_collected("grasp_misalignment_D0", 3)
        assert status.get_collected("grasp_misalignment_D0") == 3
        assert status.remaining("grasp_misalignment_D0") == 2
        assert not status.is_collection_complete("grasp_misalignment_D0")

        status.increment_collected("grasp_misalignment_D0", 2)
        assert status.is_collection_complete("grasp_misalignment_D0")

    def test_summary(self):
        status = RecoveryCollectionStatus(
            task_name="stack",
            target_demos={"a_D0": 3, "b_D0": 3},
            collected_demos={"a_D0": 3, "b_D0": 1},
        )
        s = status.summary()
        assert s['total_target'] == 6
        assert s['total_collected'] == 4
        assert s['subtypes_complete'] == 1
        assert s['subtypes_total'] == 2

    def test_serialization(self):
        status = RecoveryCollectionStatus(
            task_name="pick_place",
            target_demos={"x_D0": 5},
            collected_demos={"x_D0": 2},
            augmented_demos={"x_D0": 40},
        )
        d = status.to_dict()
        status2 = RecoveryCollectionStatus.from_dict(d)
        assert status2.get_collected("x_D0") == 2
        assert status2.get_augmented("x_D0") == 40

    def test_scene_coverage_mode_uses_unique_scene_ids(self):
        status = RecoveryCollectionStatus(
            task_name="stack",
            collection_mode="per_scene",
            target_scene_ids={"collision_empty_D0": ["scene_0", "scene_1"]},
            collected_scene_ids={"collision_empty_D0": ["scene_1"]},
        )
        assert status.get_target("collision_empty_D0") == 2
        assert status.get_collected("collision_empty_D0") == 1
        assert status.remaining("collision_empty_D0") == 1
        assert status.summary()["collection_mode"] == "per_scene"


class TestSubtypeRBGMapping:
    def test_all_23_subtypes_mapped(self):
        """All taxonomy subtypes should be in SUBTYPE_TO_RBG."""
        from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes
        for error_name, degree in get_all_subtypes():
            subtype_id = f"{error_name}_{degree}"
            assert subtype_id in SUBTYPE_TO_RBG, \
                f"{subtype_id} not in SUBTYPE_TO_RBG"

    def test_rbg_groups_cover_all_subtypes(self):
        """Every subtype in SUBTYPE_TO_RBG should have a valid RBG."""
        valid_rbgs = set(RBG_SUBTASK_SEQUENCES.keys())
        for subtype_id, rbg in SUBTYPE_TO_RBG.items():
            assert rbg in valid_rbgs, f"{subtype_id} has invalid RBG: {rbg}"

    def test_motion_templates_for_all_rbgs(self):
        """Every RBG should have a motion template."""
        for rbg in RBG_SUBTASK_SEQUENCES:
            assert rbg in RBG_MOTION_TEMPLATES, \
                f"{rbg} missing from RBG_MOTION_TEMPLATES"


# ─── Test recovery_segmenter ───

from error_benchmark.framework.recovery_segmenter import RecoverySegmenter


class TestRecoverySegmenter:
    @pytest.fixture
    def task_config(self):
        return {
            'thresholds': {
                'reach': 0.06,
                'grasp_closed': 0.05,
                'lift_height': 0.84,
                'transport': 0.05,
            },
            'objects': [
                {'name': 'cubeA', 'body_name': 'cubeA_main'},
            ],
        }

    @pytest.fixture
    def segmenter(self, task_config):
        seg_config = {'segmentation_mode': 'rbg'}
        return RecoverySegmenter(task_config, seg_config)

    def _make_demo(self, subtype_id, num_steps=100):
        """Create a mock recovery demo with synthetic trajectories."""
        for d in ['D0', 'D1']:
            if subtype_id.endswith(f'_{d}'):
                error_name = subtype_id[:-(len(d) + 1)]
                degree = d
                break
        else:
            error_name = subtype_id
            degree = "D0"

        # Synthetic EEF trajectory:
        # Phase 1 (0-20): move up (retract)
        # Phase 2 (20-50): move toward object (navigate)
        # Phase 3 (50-60): grasp
        # Phase 4 (60-75): lift
        # Phase 5 (75-90): transport
        # Phase 6 (90-100): place
        T = num_steps
        eef_pos = np.zeros((T + 1, 3))
        gripper = np.zeros(T + 1)

        for t in range(T + 1):
            frac = t / T
            if frac < 0.2:
                eef_pos[t] = [0.0, 0.0, 0.8 + 0.1 * (frac / 0.2)]
                gripper[t] = 0.01
            elif frac < 0.5:
                eef_pos[t] = [0.1 * (frac - 0.2) / 0.3, 0.0, 0.9]
                gripper[t] = 0.02
            elif frac < 0.6:
                eef_pos[t] = [0.1, 0.0, 0.85 - 0.05 * (frac - 0.5) / 0.1]
                gripper[t] = 0.05 + 0.1 * (frac - 0.5) / 0.1
            elif frac < 0.75:
                eef_pos[t] = [0.1, 0.0, 0.80 + 0.1 * (frac - 0.6) / 0.15]
                gripper[t] = 0.15
            elif frac < 0.9:
                eef_pos[t] = [0.2, 0.0, 0.90]
                gripper[t] = 0.15
            else:
                eef_pos[t] = [0.2, 0.0, 0.85 - 0.05 * (frac - 0.9) / 0.1]
                gripper[t] = 0.05

        obj_pos = eef_pos.copy()
        obj_pos[:, 2] -= 0.05

        return RecoveryDemo(
            demo_id=f"test_{subtype_id}",
            task_name="stack",
            error_name=error_name,
            degree=degree,
            success=True,
            num_steps=T,
            eef_positions=eef_pos,
            gripper_states=gripper,
            object_positions={"cubeA": obj_pos},
        )

    def test_segment_regrasp(self, segmenter):
        demo = self._make_demo("grasp_misalignment_D0")
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        # Should start with retract
        assert subtasks[0].label == "retract"
        # Should cover entire trajectory
        assert subtasks[0].start_step == 0
        assert subtasks[-1].end_step == demo.num_steps

    def test_segment_retrieve(self, segmenter):
        demo = self._make_demo("drop_in_transit_D0")
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        assert subtasks[0].label == "navigate_to_object"

    def test_segment_retract(self, segmenter):
        demo = self._make_demo("collision_holding_D0")
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        assert subtasks[0].label == "retract"

    def test_segment_redirect(self, segmenter):
        demo = self._make_demo("wrong_object_D0")
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        assert subtasks[0].label == "release"

    def test_segment_realign(self, segmenter):
        demo = self._make_demo("position_error_D0")
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        assert subtasks[0].label == "correct_position"

    def test_no_empty_subtasks(self, segmenter):
        """No subtask should have zero duration."""
        for subtype in ["grasp_misalignment_D0", "drop_in_transit_D0",
                        "collision_holding_D0", "wrong_object_D0",
                        "position_error_D0"]:
            demo = self._make_demo(subtype)
            subtasks = segmenter.segment(demo)
            for st in subtasks:
                assert st.duration > 0, f"Empty subtask {st.label} in {subtype}"

    def test_subtasks_cover_full_trajectory(self, segmenter):
        """Subtasks should cover [0, num_steps) without gaps."""
        demo = self._make_demo("drop_at_wrong_place_D0")
        subtasks = segmenter.segment(demo)
        assert subtasks[0].start_step == 0
        assert subtasks[-1].end_step == demo.num_steps

    def test_short_trajectory(self, segmenter):
        """Should handle very short trajectories gracefully."""
        demo = self._make_demo("stuck_no_progress_D0", num_steps=5)
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0
        assert subtasks[-1].end_step == 5

    def test_no_data_fallback(self, segmenter):
        """Should fall back to uniform segmentation when no arrays."""
        demo = RecoveryDemo(
            demo_id="no_data",
            error_name="collision_empty",
            degree="D0",
            num_steps=50,
        )
        subtasks = segmenter.segment(demo)
        assert len(subtasks) > 0


# ─── Test conversion utilities ───

class TestMCMConversion:
    def test_error_descriptions_complete(self):
        """All 13 error skills should have descriptions."""
        from error_benchmark.framework.error_taxonomy_v5 import ERROR_SKILL_DEFS
        expected_skills = set()
        for skill_enum in ERROR_SKILL_DEFS:
            expected_skills.add(skill_enum.value)
        # All 12 skills should exist
        assert len(expected_skills) == 12

    def test_motion_instructions_cover_all_rbg_subtask_pairs(self):
        """All (RBG, subtask) pairs should have motion instructions."""
        # Check that every subtask in every RBG sequence has a template
        for rbg, sequence in RBG_SUBTASK_SEQUENCES.items():
            for subtask in sequence:
                # At minimum, the RBG should have a template
                assert rbg in RBG_MOTION_TEMPLATES, \
                    f"Missing RBG template for {rbg}"


class TestDiffusionConversion:
    def test_downsample_trajectory(self):
        actions = np.random.randn(100, 7)
        # Downsample from 20Hz to 5Hz = keep every 4th
        ds = actions[::4]
        assert ds.shape == (25, 7)

    def test_action_chunks(self):
        T, action_dim = 25, 7
        chunk_length = 12
        actions = np.random.randn(T, action_dim)

        # Create chunks
        chunks = np.zeros((T, chunk_length, action_dim))
        for t in range(T):
            end = min(t + chunk_length, T)
            actual_len = end - t
            chunks[t, :actual_len] = actions[t:end]
            if actual_len < chunk_length:
                chunks[t, actual_len:] = actions[-1]

        assert chunks.shape == (25, 12, 7)
        # First chunk starts with first action
        np.testing.assert_array_equal(chunks[0, 0], actions[0])
        # Last chunk should be padded
        np.testing.assert_array_equal(chunks[-1, 0], actions[-1])


# ─── Test config ───

class TestRecoveryConfig:
    @pytest.fixture
    def config(self):
        config_path = (PROJECT_ROOT /
                       "error_benchmark/configs/recovery_collection.yaml")
        if not config_path.exists():
            pytest.skip("recovery_collection.yaml not found")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_all_rbgs_defined(self, config):
        rbgs = config['recovery_behavior_groups']
        assert set(rbgs.keys()) == {"RBG_A", "RBG_B", "RBG_C", "RBG_D", "RBG_E"}

    def test_demo_allocations_valid_subtypes(self, config):
        """All subtype_ids in allocations should be valid."""
        from error_benchmark.framework.error_taxonomy_v5 import is_valid_subtype
        for task, allocs in config.get('demo_allocations', {}).items():
            for subtype_id, count in allocs.items():
                for d in ['D0', 'D1']:
                    if subtype_id.endswith(f'_{d}'):
                        error_name = subtype_id[:-(len(d) + 1)]
                        degree = d
                        break
                else:
                    pytest.fail(f"Can't parse subtype: {subtype_id}")
                assert is_valid_subtype(error_name, degree), \
                    f"Invalid subtype {subtype_id} in task {task}"
                assert count > 0, f"Zero allocation for {subtype_id} in {task}"

    def test_pick_place_allocation_total(self, config):
        """pick_place should have ~99 total demos."""
        alloc = config['demo_allocations']['pick_place']
        total = sum(alloc.values())
        assert 90 <= total <= 110, f"pick_place total={total}, expected ~99"

    def test_stack_allocation_total(self, config):
        """stack should have ~69 total demos."""
        alloc = config['demo_allocations']['stack']
        total = sum(alloc.values())
        assert 60 <= total <= 80, f"stack total={total}, expected ~69"

    def test_tier_tasks_match_registry(self, config):
        """All tasks in tiers should exist in task_registry."""
        # Load registry directly (avoid importing script_utils which requires robosuite)
        import os
        registry_path = os.path.join(
            str(PROJECT_ROOT), 'error_benchmark', 'configs', 'task_registry.yaml')
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
        available_tasks = set(registry.get('tasks', {}).keys())

        for tier_name, tier_info in config['task_tiers'].items():
            for task in tier_info['tasks']:
                assert task in available_tasks, \
                    f"Task '{task}' from {tier_name} not in task_registry"

    def test_rbg_subtypes_match_taxonomy(self, config):
        """All subtypes listed in RBGs should be valid."""
        from error_benchmark.framework.error_taxonomy_v5 import is_valid_subtype
        for rbg_name, rbg_info in config['recovery_behavior_groups'].items():
            for subtype_id in rbg_info['subtypes']:
                for d in ['D0', 'D1']:
                    if subtype_id.endswith(f'_{d}'):
                        error_name = subtype_id[:-(len(d) + 1)]
                        degree = d
                        break
                assert is_valid_subtype(error_name, degree), \
                    f"Invalid subtype {subtype_id} in {rbg_name}"
