#!/usr/bin/env python
"""
Unit tests for v5.0 Error Skills Framework
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from error_benchmark.framework.error_skills import (
    BaseErrorSkill,
    SkillConfig,
    SKILL_REGISTRY,
    get_skill,
    get_all_skills,
)
from error_benchmark.framework.error_skills.base_skill import BaseErrorSkill as BaseSkillDirect
from error_benchmark.framework.error_skills.e10_stuck_no_progress import StuckNoProgressSkill
from error_benchmark.framework.error_skills.e12_position_error import PositionErrorSkill
from error_benchmark.framework.error_skills.e02_drop_in_transit import DropInTransitSkill
from error_benchmark.framework.error_skills.e03_drop_at_wrong_place import DropAtWrongPlaceSkill


class TestSkillRegistry:
    """Tests for the skill registry."""

    def test_registry_has_all_skills(self):
        """Registry contains all 12 skills."""
        assert len(SKILL_REGISTRY) == 12

    def test_registry_has_core_skills(self):
        """Registry contains core skills."""
        assert "stuck_no_progress" in SKILL_REGISTRY
        assert "position_error" in SKILL_REGISTRY
        assert "drop_in_transit" in SKILL_REGISTRY
        assert "drop_at_wrong_place" in SKILL_REGISTRY

    def test_old_drop_removed(self):
        """Old 'drop' key no longer in registry."""
        assert "drop" not in SKILL_REGISTRY

    def test_get_skill(self):
        """get_skill returns correct skill type."""
        skill = get_skill("stuck_no_progress")
        assert isinstance(skill, StuckNoProgressSkill)

    def test_get_skill_with_config(self):
        """get_skill passes config to skill."""
        config = {"freeze_steps": 50}
        skill = get_skill("stuck_no_progress", config)
        assert skill.config.get("freeze_steps") == 50

    def test_get_skill_unknown(self):
        """get_skill raises KeyError for unknown skill."""
        with pytest.raises(KeyError):
            get_skill("nonexistent_skill")

    def test_get_all_skills(self):
        """get_all_skills returns all registered skills."""
        skills = get_all_skills()
        assert len(skills) == 12
        names = {s.name for s in skills}
        assert "stuck_no_progress" in names
        assert "position_error" in names
        assert "drop_in_transit" in names
        assert "drop_at_wrong_place" in names


class TestSkillConfig:
    """Tests for SkillConfig."""

    def test_empty_config(self):
        config = SkillConfig()
        assert config.get("key") is None
        assert config.get("key", 42) == 42

    def test_with_params(self):
        config = SkillConfig(params={"freeze_steps": 30, "D0": {"offset": 0.1}})
        assert config.get("freeze_steps") == 30

    def test_degree_params(self):
        config = SkillConfig(params={"D0": {"offset_range": [0.03, 0.10]}})
        d0_params = config.get_degree_params("D0")
        assert d0_params["offset_range"] == [0.03, 0.10]

        d1_params = config.get_degree_params("D1")
        assert d1_params == {}


class TestStuckNoProgressSkill:
    """Tests for E11 stuck_no_progress skill."""

    def setup_method(self):
        self.skill = StuckNoProgressSkill()

    def test_name(self):
        assert self.skill.name == "stuck_no_progress"

    def test_valid_degrees(self):
        assert self.skill.valid_degrees == ["D0", "D1"]

    def test_valid_phases(self):
        phases = self.skill.valid_phases
        assert "reach" in phases
        assert "lift" in phases
        assert "transport" in phases

    def test_supports_degree(self):
        assert self.skill.supports_degree("D0") is True
        assert self.skill.supports_degree("D1") is True

    def test_supports_phase(self):
        assert self.skill.supports_phase("reach") is True
        assert self.skill.supports_phase("done") is False

    def test_can_inject_valid(self):
        """Can inject during valid phase with enough remaining frames."""
        frame_state = {'task_phase': 'reach'}
        context = {
            'task_phase': 'reach',
            'frame_index': 20,
            'total_frames': 100,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == ["D0", "D1"]

    def test_can_inject_done_phase(self):
        """Cannot inject during 'done' phase."""
        frame_state = {'task_phase': 'done'}
        context = {
            'task_phase': 'done',
            'frame_index': 90,
            'total_frames': 100,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == []

    def test_can_inject_too_few_frames(self):
        """Cannot inject if too few frames remaining."""
        frame_state = {'task_phase': 'reach'}
        context = {
            'task_phase': 'reach',
            'frame_index': 95,
            'total_frames': 100,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == []


class TestPositionErrorSkill:
    """Tests for E13 position_error skill."""

    def setup_method(self):
        self.skill = PositionErrorSkill(
            config=SkillConfig(params={
                "D0": {"offset_range": [0.03, 0.10]},
                "D1": {"rotation_range": [0.2, 0.8]},
            })
        )

    def test_name(self):
        assert self.skill.name == "position_error"

    def test_valid_degrees(self):
        assert set(self.skill.valid_degrees) == {"D0", "D1"}

    def test_can_inject_valid(self):
        frame_state = {'task_phase': 'reach'}
        context = {
            'task_phase': 'reach',
            'frame_index': 20,
            'total_frames': 100,
        }
        result = self.skill.can_inject(frame_state, context)
        assert "D0" in result
        assert "D1" in result

    def test_can_inject_invalid_phase(self):
        frame_state = {'task_phase': 'done'}
        context = {
            'task_phase': 'done',
            'frame_index': 20,
            'total_frames': 100,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == []


class TestDropInTransitSkill:
    """Tests for E2 drop_in_transit skill."""

    def setup_method(self):
        self.skill = DropInTransitSkill(
            config=SkillConfig(params={
                "target_distance_threshold": 0.10,
                "orientation_threshold": 0.3,
                "D0": {"min_hold_height": 0.85, "settle_steps": 40},
                "D1": {"min_hold_height": 0.85, "settle_steps": 40},
            })
        )

    def test_name(self):
        assert self.skill.name == "drop_in_transit"

    def test_valid_degrees(self):
        assert set(self.skill.valid_degrees) == {"D0", "D1"}

    def test_can_inject_holding_high(self):
        """Can inject when holding object above table."""
        frame_state = {
            'task_phase': 'lift',
            'gripper_closed_norm': 0.25,
            'eef_pos': np.array([0.5, 0.0, 0.92]),
            'objects': {'cube': {'pos': np.array([0.5, 0.0, 0.9])}},
        }
        context = {
            'task_phase': 'lift',
            'frame_index': 50,
            'total_frames': 200,
        }
        result = self.skill.can_inject(frame_state, context)
        assert "D0" in result

    def test_can_inject_not_holding(self):
        """Cannot inject when object is far from EEF."""
        frame_state = {
            'task_phase': 'lift',
            'gripper_closed_norm': 0.05,
            'eef_pos': np.array([0.0, 0.0, 1.0]),
            'objects': {'cube': {'pos': np.array([0.5, 0.0, 0.9])}},
        }
        context = {
            'task_phase': 'lift',
            'frame_index': 50,
            'total_frames': 200,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == []

    def test_can_inject_object_too_low(self):
        """Cannot inject when object is too low."""
        frame_state = {
            'task_phase': 'lift',
            'gripper_closed_norm': 0.25,
            'eef_pos': np.array([0.5, 0.0, 0.72]),
            'objects': {'cube': {'pos': np.array([0.5, 0.0, 0.7])}},
        }
        context = {
            'task_phase': 'lift',
            'frame_index': 50,
            'total_frames': 200,
        }
        result = self.skill.can_inject(frame_state, context)
        assert result == []


class TestDropAtWrongPlaceSkill:
    """Tests for E3 drop_at_wrong_place skill."""

    def setup_method(self):
        self.skill = DropAtWrongPlaceSkill(
            config=SkillConfig(params={
                "target_distance_threshold": 0.10,
                "orientation_threshold": 0.3,
                "D0": {"min_hold_height": 0.85, "eef_offset_range": [0.05, 0.10]},
                "D1": {"min_hold_height": 0.85, "eef_offset_range": [0.05, 0.10]},
            })
        )

    def test_name(self):
        assert self.skill.name == "drop_at_wrong_place"

    def test_valid_degrees(self):
        assert set(self.skill.valid_degrees) == {"D0", "D1"}

    def test_valid_phases(self):
        assert "transport" in self.skill.valid_phases
        assert "place" in self.skill.valid_phases


class TestBaseErrorSkillHelpers:
    """Tests for BaseErrorSkill helper methods (check_remaining_frames, normalize_direction, random_direction)."""

    def _make_skill(self):
        """Return a concrete skill instance for testing base class helpers."""
        return StuckNoProgressSkill()

    # ── check_remaining_frames ──

    def test_check_remaining_frames_default_thresholds(self):
        skill = self._make_skill()
        # remaining=25 → D0 only
        ctx = {'frame_index': 5, 'total_frames': 30}
        assert skill.check_remaining_frames(ctx) == ['D0']
        # remaining=35 → D0 + D1
        ctx = {'frame_index': 5, 'total_frames': 40}
        assert skill.check_remaining_frames(ctx) == ['D0', 'D1']
        # remaining=15 → empty
        ctx = {'frame_index': 5, 'total_frames': 20}
        assert skill.check_remaining_frames(ctx) == []

    def test_check_remaining_frames_custom_thresholds(self):
        skill = self._make_skill()
        # e08 thresholds (30/40)
        ctx = {'frame_index': 0, 'total_frames': 35}
        assert skill.check_remaining_frames(ctx, min_frames=30, d1_min_frames=40) == ['D0']
        ctx = {'frame_index': 0, 'total_frames': 45}
        assert skill.check_remaining_frames(ctx, min_frames=30, d1_min_frames=40) == ['D0', 'D1']
        # e11 thresholds (25/30)
        ctx = {'frame_index': 0, 'total_frames': 27}
        assert skill.check_remaining_frames(ctx, min_frames=25, d1_min_frames=30) == ['D0']

    def test_check_remaining_frames_edge_cases(self):
        skill = self._make_skill()
        # Exactly at threshold → D0
        ctx = {'frame_index': 0, 'total_frames': 20}
        assert skill.check_remaining_frames(ctx) == ['D0']
        # One below D1 threshold
        ctx = {'frame_index': 0, 'total_frames': 29}
        assert skill.check_remaining_frames(ctx) == ['D0']
        # Exactly at D1 threshold
        ctx = {'frame_index': 0, 'total_frames': 30}
        assert skill.check_remaining_frames(ctx) == ['D0', 'D1']
        # total_frames=0
        ctx = {'frame_index': 0, 'total_frames': 0}
        assert skill.check_remaining_frames(ctx) == []
        # Missing keys default to 0
        assert skill.check_remaining_frames({}) == []

    # ── normalize_direction ──

    def test_normalize_direction_unit_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = BaseErrorSkill.normalize_direction(v)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-10)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_normalize_direction_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = BaseErrorSkill.normalize_direction(v)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])
        # Must be a copy, not the same object
        assert result is not v

    def test_normalize_direction_does_not_mutate_input(self):
        v = np.array([3.0, 4.0])
        _ = BaseErrorSkill.normalize_direction(v)
        assert v[0] == 3.0 and v[1] == 4.0

    def test_normalize_direction_near_zero(self):
        v = np.array([1e-12, 0.0])
        result = BaseErrorSkill.normalize_direction(v)
        # Near-zero returns copy unchanged
        np.testing.assert_array_equal(result, v)

    # ── random_direction ──

    def test_random_direction_unit_length(self):
        rng = np.random.RandomState(42)
        for dims in [2, 3]:
            d = BaseErrorSkill.random_direction(rng, dims=dims)
            assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-7)

    def test_random_direction_dimensionality(self):
        rng = np.random.RandomState(42)
        assert BaseErrorSkill.random_direction(rng, dims=2).shape == (2,)
        assert BaseErrorSkill.random_direction(rng, dims=3).shape == (3,)

    def test_random_direction_reproducible(self):
        d1 = BaseErrorSkill.random_direction(np.random.RandomState(99), dims=2)
        d2 = BaseErrorSkill.random_direction(np.random.RandomState(99), dims=2)
        np.testing.assert_array_equal(d1, d2)


class TestBaseSkillCreateErrorSpec:
    """Test the create_error_spec helper method."""

    def test_creates_v5_spec(self):
        skill = StuckNoProgressSkill()
        spec = skill.create_error_spec(
            degree="D0",
            target={"eef": True},
            params={"freeze_steps": 30},
            frame_index=50,
            trajectory_id="demo_0",
            seed=42,
        )
        assert spec.type == "error_skill"
        assert spec.family == "action_based"
        assert spec.error_name == "stuck_no_progress"
        assert spec.degree == "D0"
        assert spec.source_frame == 50
        assert spec.source_trajectory == "demo_0"
