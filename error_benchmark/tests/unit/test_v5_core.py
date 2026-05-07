#!/usr/bin/env python
"""
Unit tests for v5.0 Core Data Structures
"""

import pytest
import numpy as np
from error_benchmark.framework.core import (
    ErrorSpec,
    ErrorScene,
    CleanTrajectory,
    InjectionOpportunity,
    QuotaStatus,
    ValidationResult_v5,
    PostRolloutStats,
    map_phase_to_group,
)


class TestErrorSpecV5:
    """Tests for v5 ErrorSpec extensions."""

    def test_v5_fields_default(self):
        """v5 fields default to empty."""
        spec = ErrorSpec(
            type="impulse",
            family="physics",
            target={"body": "cube"},
            params={"force": 5.0},
            apply={"mode": "at_step", "t": 50},
            seed=42,
        )
        assert spec.error_name == ""
        assert spec.degree == ""
        assert spec.source_frame == -1
        assert spec.source_trajectory == ""

    def test_v5_fields_set(self):
        """v5 fields can be set."""
        spec = ErrorSpec(
            type="error_skill",
            family="action_based",
            target={"eef": True},
            params={"offset": [0.1, 0, 0]},
            apply={"mode": "at_step", "t": 50},
            seed=42,
            error_name="position_error",
            degree="D0",
            source_frame=50,
            source_trajectory="demo_pick_place_demo_0",
        )
        assert spec.error_name == "position_error"
        assert spec.degree == "D0"
        assert spec.source_frame == 50

    def test_v5_serialization(self):
        """v5 fields survive to_dict/from_dict round trip."""
        spec = ErrorSpec(
            type="error_skill",
            family="action_based",
            target={"eef": True},
            params={"offset": [0.1, 0, 0]},
            apply={"mode": "at_step", "t": 50},
            seed=42,
            error_name="drop_in_transit",
            degree="D1",
            source_frame=75,
            source_trajectory="demo_traj_1",
        )
        d = spec.to_dict()
        restored = ErrorSpec.from_dict(d)

        assert restored.error_name == "drop_in_transit"
        assert restored.degree == "D1"
        assert restored.source_frame == 75
        assert restored.source_trajectory == "demo_traj_1"

    def test_target_object_default(self):
        """target_object defaults to empty string."""
        spec = ErrorSpec(
            type="impulse", family="physics", target={"body": "cube"},
            params={"force": 5.0}, apply={"mode": "at_step", "t": 50}, seed=42,
        )
        assert spec.target_object == ""

    def test_target_object_roundtrip(self):
        """target_object survives to_dict/from_dict."""
        spec = ErrorSpec(
            type="error_skill", family="action_based", target={"eef": True},
            params={}, apply={"mode": "at_step", "t": 50}, seed=42,
            error_name="drop_in_transit", degree="D0",
            target_object="cubeA",
        )
        d = spec.to_dict()
        assert d['target_object'] == "cubeA"
        restored = ErrorSpec.from_dict(d)
        assert restored.target_object == "cubeA"


class TestErrorSceneV5:
    """Tests for v5 ErrorScene extensions."""

    def test_v5_fields_default(self):
        """v5 fields default to empty."""
        scene = ErrorScene()
        assert scene.clean_trajectory_ref == ""
        assert scene.injection_step == -1
        assert scene.render_window == 0

    def test_v5_fields_set(self):
        """v5 fields can be set."""
        scene = ErrorScene(
            version="v5.0",
            scene_id="v5_drop_D0_abc123",
            clean_trajectory_ref="demo_pick_place_demo_0",
            injection_step=50,
            render_window=40,
        )
        assert scene.clean_trajectory_ref == "demo_pick_place_demo_0"
        assert scene.injection_step == 50
        assert scene.render_window == 40

    def test_v5_serialization(self):
        """v5 fields survive to_dict/from_dict round trip."""
        scene = ErrorScene(
            version="v5.0",
            scene_id="test_scene",
            clean_trajectory_ref="traj_1",
            injection_step=30,
            render_window=25,
        )
        d = scene.to_dict()
        restored = ErrorScene.from_dict(d)

        assert restored.clean_trajectory_ref == "traj_1"
        assert restored.injection_step == 30
        assert restored.render_window == 25


    def test_post_injection_roundtrip(self):
        """post_injection field survives to_dict/from_dict round trip."""
        post_inj = {
            'obj_poses': {
                'cubeA': {'pos': [0.05, -0.03, 0.82], 'quat': [0.7, 0.7, 0.0, 0.0]},
                'cubeB': {'pos': [-0.08, 0.01, 0.82], 'quat': [1.0, 0.0, 0.0, 0.0]},
            },
            'gripper_closed_norm': 0.02,
            'eef_pos': [0.02, 0.11, 0.95],
            'id_eligible': True,
            'target_object': 'cubeA',
        }
        scene = ErrorScene(
            version="v5.0",
            scene_id="test_post_inj",
            post_injection=post_inj,
        )
        d = scene.to_dict()
        assert 'post_injection' in d
        assert d['post_injection']['id_eligible'] is True

        restored = ErrorScene.from_dict(d)
        assert restored.post_injection['id_eligible'] is True
        assert restored.post_injection['target_object'] == 'cubeA'
        assert len(restored.post_injection['obj_poses']) == 2
        assert restored.post_injection['obj_poses']['cubeA']['pos'] == [0.05, -0.03, 0.82]

    def test_post_injection_default_empty(self):
        """post_injection defaults to empty dict."""
        scene = ErrorScene()
        assert scene.post_injection == {}

    def test_post_injection_missing_in_old_json(self):
        """from_dict handles missing post_injection (backward compat)."""
        d = {'version': 'v5.0', 'scene_id': 'old_scene'}
        scene = ErrorScene.from_dict(d)
        assert scene.post_injection == {}


class TestCleanTrajectory:
    """Tests for CleanTrajectory data structure."""

    def test_creation(self):
        """Basic creation with fields."""
        traj = CleanTrajectory(
            trajectory_id="demo_pick_place_demo_0",
            source="demo",
            task_name="pick_place",
            dataset_path="/path/to/dataset.hdf5",
            demo_key="demo_0",
            num_steps=100,
        )
        assert traj.trajectory_id == "demo_pick_place_demo_0"
        assert traj.source == "demo"
        assert traj.num_steps == 100

    def test_with_arrays(self):
        """Can store numpy arrays."""
        traj = CleanTrajectory(
            trajectory_id="test",
            num_steps=10,
            actions=np.zeros((10, 7)),
            states=np.zeros((11, 50)),
        )
        assert traj.actions.shape == (10, 7)
        assert traj.states.shape == (11, 50)

    def test_serialization(self):
        """to_dict excludes large arrays."""
        traj = CleanTrajectory(
            trajectory_id="test",
            source="demo",
            task_name="pick_place",
            num_steps=10,
            actions=np.zeros((10, 7)),
        )
        d = traj.to_dict()
        assert 'trajectory_id' in d
        assert 'actions' not in d  # Arrays excluded

    def test_from_dict(self):
        """from_dict restores metadata."""
        d = {
            'trajectory_id': 'test',
            'source': 'vla_pi0',
            'task_name': 'pick_place',
            'num_steps': 50,
        }
        traj = CleanTrajectory.from_dict(d)
        assert traj.trajectory_id == 'test'
        assert traj.source == 'vla_pi0'
        assert traj.num_steps == 50


class TestInjectionOpportunity:
    """Tests for InjectionOpportunity data structure."""

    def test_creation(self):
        opp = InjectionOpportunity(
            trajectory_id="traj_0",
            frame_index=50,
            error_name="drop_in_transit",
            degree="D0",
            task_phase="lift",
        )
        assert opp.trajectory_id == "traj_0"
        assert opp.frame_index == 50
        assert opp.subtype_id() == "drop_in_transit_D0"

    def test_serialization(self):
        opp = InjectionOpportunity(
            trajectory_id="traj_0",
            frame_index=50,
            error_name="position_error",
            degree="D1",
            task_phase="transport",
            confidence=0.9,
        )
        d = opp.to_dict()
        restored = InjectionOpportunity.from_dict(d)
        assert restored.trajectory_id == "traj_0"
        assert restored.frame_index == 50
        assert restored.error_name == "position_error"
        assert restored.degree == "D1"
        assert restored.confidence == 0.9


class TestQuotaStatus:
    """Tests for QuotaStatus data structure."""

    def test_initial_state(self):
        quota = QuotaStatus(task_name="pick_place", quota_per_subtype=100)
        assert quota.get_count("drop_in_transit", "D0") == 0
        assert quota.remaining("drop_in_transit", "D0") == 100
        assert quota.is_fulfilled("drop_in_transit", "D0") is False

    def test_increment(self):
        quota = QuotaStatus(task_name="pick_place", quota_per_subtype=10)
        quota.increment("drop_in_transit", "D0", 5)
        assert quota.get_count("drop_in_transit", "D0") == 5
        assert quota.remaining("drop_in_transit", "D0") == 5

        quota.increment("drop_in_transit", "D0", 5)
        assert quota.is_fulfilled("drop_in_transit", "D0") is True
        assert quota.remaining("drop_in_transit", "D0") == 0

    def test_summary(self):
        quota = QuotaStatus(task_name="pick_place", quota_per_subtype=10)
        summary = quota.summary()
        assert summary['task_name'] == "pick_place"
        assert summary['total_subtypes'] == 24
        assert summary['fulfilled'] == 0

    def test_serialization(self):
        quota = QuotaStatus(task_name="test", quota_per_subtype=50)
        quota.increment("drop_in_transit", "D0", 10)
        d = quota.to_dict()
        restored = QuotaStatus.from_dict(d)
        assert restored.get_count("drop_in_transit", "D0") == 10
        assert restored.quota_per_subtype == 50


class TestValidationResultV5:
    """Tests for ValidationResult_v5."""

    def test_creation(self):
        result = ValidationResult_v5(
            ok=True,
            error_name="drop_in_transit",
            degree="D0",
            metrics={"delta_z": 0.15},
            reason="Drop confirmed",
        )
        assert result.ok is True
        assert result.error_name == "drop_in_transit"

    def test_serialization(self):
        result = ValidationResult_v5(
            ok=False,
            error_name="stuck_no_progress",
            degree="D0",
            metrics={"displacement": 0.001},
            reason="Not stuck enough",
        )
        d = result.to_dict()
        assert d['ok'] is False
        assert d['error_name'] == "stuck_no_progress"


class TestPhaseGroupMapping:
    def test_approach_phases(self):
        assert map_phase_to_group("reach") == "approach"
        assert map_phase_to_group("pre_reach") == "approach"
        assert map_phase_to_group("pre_grasp") == "approach"

    def test_grasp_phase(self):
        assert map_phase_to_group("grasp") == "grasp"

    def test_transfer_phases(self):
        assert map_phase_to_group("lift") == "transfer"
        assert map_phase_to_group("transport") == "transfer"

    def test_place_phases(self):
        assert map_phase_to_group("place") == "place"
        assert map_phase_to_group("post_place") == "place"

    def test_unknown_defaults_to_approach(self):
        assert map_phase_to_group("") == "approach"
        assert map_phase_to_group("unknown") == "approach"
