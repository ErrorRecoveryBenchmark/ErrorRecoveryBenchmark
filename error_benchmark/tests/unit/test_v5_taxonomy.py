#!/usr/bin/env python
"""
Unit tests for v5.0 Error Taxonomy
"""

import pytest
from error_benchmark.framework.error_taxonomy_v5 import (
    ErrorDegree,
    ErrorSkillName,
    ERROR_SKILL_DEFS,
    get_all_subtypes,
    get_subtype_id,
    get_skill_def,
    is_valid_subtype,
    get_valid_degrees,
    get_valid_phases,
    PHASE_ORDER,
    TOTAL_SKILLS,
    TOTAL_SUBTYPES,
)


class TestErrorTaxonomyV5:
    """Tests for the v5 error taxonomy."""

    def test_total_skills(self):
        """12 error skills defined."""
        assert TOTAL_SKILLS == 12

    def test_total_subtypes(self):
        """24 subtypes total."""
        assert TOTAL_SUBTYPES == 24

    def test_all_subtypes_count(self):
        """get_all_subtypes returns exactly 24 items."""
        subtypes = get_all_subtypes()
        assert len(subtypes) == 24

    def test_subtype_format(self):
        """Each subtype is (error_name, degree) tuple."""
        for error_name, degree in get_all_subtypes():
            assert isinstance(error_name, str)
            assert degree in ("D0", "D1")

    def test_degree_enum(self):
        """ErrorDegree enum has 2 values."""
        assert len(ErrorDegree) == 2
        assert ErrorDegree.D0 == "D0"
        assert ErrorDegree.D1 == "D1"

    def test_skill_name_enum(self):
        """ErrorSkillName enum has 12 values."""
        assert len(ErrorSkillName) == 12

    def test_skill_defs_complete(self):
        """Each skill def has required fields."""
        for skill_name, skill_def in ERROR_SKILL_DEFS.items():
            assert "id" in skill_def
            assert "name" in skill_def
            assert "description" in skill_def
            assert "valid_degrees" in skill_def
            assert "valid_phases" in skill_def
            assert len(skill_def["valid_degrees"]) > 0
            assert len(skill_def["valid_phases"]) > 0

    def test_get_subtype_id(self):
        """Subtype ID format is error_name_degree."""
        assert get_subtype_id("drop_in_transit", "D0") == "drop_in_transit_D0"
        assert get_subtype_id("grasp_misalignment", "D1") == "grasp_misalignment_D1"

    def test_get_skill_def(self):
        """get_skill_def returns correct definition."""
        skill_def = get_skill_def("drop_in_transit")
        assert skill_def["id"] == "E2"
        assert ErrorDegree.D0 in skill_def["valid_degrees"]

    def test_get_skill_def_drop_variants(self):
        """Drop variants have correct IDs."""
        assert get_skill_def("drop_in_transit")["id"] == "E2"
        assert get_skill_def("drop_at_wrong_place")["id"] == "E3"

    def test_get_skill_def_invalid(self):
        """get_skill_def raises ValueError for unknown skill."""
        with pytest.raises(ValueError):
            get_skill_def("nonexistent_skill")

    def test_is_valid_subtype(self):
        """is_valid_subtype checks degree validity."""
        assert is_valid_subtype("drop_in_transit", "D0") is True
        assert is_valid_subtype("drop_in_transit", "D1") is True
        assert is_valid_subtype("drop_at_wrong_place", "D0") is True
        assert is_valid_subtype("stuck_no_progress", "D0") is True
        assert is_valid_subtype("stuck_no_progress", "D1") is True

    def test_is_valid_subtype_invalid(self):
        """is_valid_subtype returns False for unknown skill/degree."""
        assert is_valid_subtype("nonexistent", "D0") is False
        assert is_valid_subtype("drop_in_transit", "D3") is False

    def test_get_valid_degrees(self):
        """get_valid_degrees returns correct list."""
        assert set(get_valid_degrees("stuck_no_progress")) == {"D0", "D1"}
        assert set(get_valid_degrees("grasp_wrong_pose")) == {"D0", "D1"}
        assert set(get_valid_degrees("position_error")) == {"D0", "D1"}

    def test_get_valid_phases(self):
        """get_valid_phases returns non-empty list."""
        phases = get_valid_phases("drop_in_transit")
        assert "lift" in phases
        assert "transport" in phases

    def test_get_valid_phases_drop_variants(self):
        """Drop variants have correct phase lists."""
        assert "lift" in get_valid_phases("drop_in_transit")
        assert "transport" in get_valid_phases("drop_at_wrong_place")
        assert "place" in get_valid_phases("drop_at_wrong_place")

    def test_phase_order(self):
        """Phase ordering is consistent."""
        assert PHASE_ORDER["pre_reach"] < PHASE_ORDER["reach"]
        assert PHASE_ORDER["reach"] < PHASE_ORDER["grasp"]
        assert PHASE_ORDER["grasp"] < PHASE_ORDER["lift"]
        assert PHASE_ORDER["lift"] < PHASE_ORDER["transport"]

    def test_specific_subtypes(self):
        """Verify specific subtype counts per skill."""
        subtypes = get_all_subtypes()

        # Count per skill
        by_skill = {}
        for name, degree in subtypes:
            by_skill[name] = by_skill.get(name, 0) + 1

        assert by_skill["grasp_misalignment"] == 2  # D0/D1
        assert by_skill["drop_in_transit"] == 2      # D0/D1
        assert by_skill["drop_at_wrong_place"] == 2  # D0/D1
        assert by_skill["stuck_no_progress"] == 2    # D0/D1
        assert by_skill["grasp_wrong_pose"] == 2     # D0/D1
        assert by_skill["position_error"] == 2       # D0/D1

    def test_old_drop_removed(self):
        """Old 'drop' skill name no longer exists."""
        assert is_valid_subtype("drop", "D0") is False
        with pytest.raises(ValueError):
            get_skill_def("drop")
