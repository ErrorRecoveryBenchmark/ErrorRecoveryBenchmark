#!/usr/bin/env python
"""
Unit tests for v5.0 Pipeline modules (QuotaScheduler, OpportunityScanner)
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from error_benchmark.framework.core import (
    CleanTrajectory,
    InjectionOpportunity,
    QuotaStatus,
)
from error_benchmark.framework.quota_scheduler import QuotaScheduler
from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes


class TestQuotaScheduler:
    """Tests for the QuotaScheduler."""

    def setup_method(self):
        self.scheduler = QuotaScheduler(target_per_subtype=10, seed=42)

    def test_empty_opportunities(self):
        """Empty input produces empty schedule."""
        schedule = self.scheduler.create_schedule([])
        assert len(schedule) == 0

    def test_basic_scheduling(self):
        """Schedules opportunities to fill quota."""
        opportunities = []
        for i in range(20):
            opportunities.append(InjectionOpportunity(
                trajectory_id="traj_0",
                frame_index=i * 5,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="lift",
            ))

        schedule = self.scheduler.create_schedule(opportunities)
        # Should schedule up to 10 (target_per_subtype)
        assert len(schedule) <= 10
        assert len(schedule) > 0

    def test_respects_existing_quota(self):
        """Respects existing quota when scheduling."""
        quota = QuotaStatus(task_name="pick_place", quota_per_subtype=10)
        quota.increment("drop_in_transit", "D0", 8)  # Already have 8

        opportunities = [
            InjectionOpportunity(
                trajectory_id="traj_0",
                frame_index=i * 5,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="lift",
            )
            for i in range(20)
        ]

        schedule = self.scheduler.create_schedule(opportunities, quota)
        # Should only schedule 2 more (10 - 8 = 2)
        drop_d0_count = sum(1 for o in schedule if o.error_name == "drop_in_transit" and o.degree == "D0")
        assert drop_d0_count <= 2

    def test_per_trajectory_limit(self):
        """Respects per-trajectory limit."""
        scheduler = QuotaScheduler(target_per_subtype=100, max_per_trajectory=3, seed=42)

        opportunities = [
            InjectionOpportunity(
                trajectory_id="traj_0",
                frame_index=i * 5,
                error_name="stuck_no_progress",
                degree="D0",
                task_phase="reach",
            )
            for i in range(50)
        ]

        schedule = scheduler.create_schedule(opportunities)
        # All from same trajectory, so limited to max_per_trajectory
        assert len(schedule) <= 3

    def test_multi_subtype_scheduling(self):
        """Schedules across multiple subtypes."""
        opportunities = []
        for i in range(15):
            opportunities.append(InjectionOpportunity(
                trajectory_id=f"traj_{i % 3}",
                frame_index=i * 5,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="lift",
            ))
        for i in range(15):
            opportunities.append(InjectionOpportunity(
                trajectory_id=f"traj_{i % 3}",
                frame_index=i * 5,
                error_name="position_error",
                degree="D0",
                task_phase="reach",
            ))

        schedule = self.scheduler.create_schedule(opportunities)
        subtypes = {o.subtype_id() for o in schedule}
        assert len(subtypes) >= 2

    def test_estimate_coverage(self):
        """Coverage estimation works correctly."""
        opportunities = [
            InjectionOpportunity(
                trajectory_id="traj_0",
                frame_index=i,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="lift",
            )
            for i in range(5)
        ]

        coverage = self.scheduler.estimate_coverage(opportunities)
        assert coverage["drop_in_transit_D0"]["available"] == 5
        assert coverage["drop_in_transit_D0"]["target"] == 10
        assert coverage["drop_in_transit_D0"]["gap"] == 5


class TestScheduleSerialization:
    """Tests for schedule save/load."""

    def test_save_load_roundtrip(self):
        """Schedule survives save/load cycle."""
        schedule = [
            InjectionOpportunity(
                trajectory_id="traj_0",
                frame_index=10,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="lift",
                confidence=0.95,
            ),
            InjectionOpportunity(
                trajectory_id="traj_1",
                frame_index=20,
                error_name="position_error",
                degree="D1",
                task_phase="reach",
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name

        try:
            QuotaScheduler.save_schedule(schedule, path)
            loaded = QuotaScheduler.load_schedule(path)

            assert len(loaded) == 2
            assert loaded[0].trajectory_id == "traj_0"
            assert loaded[0].error_name == "drop_in_transit"
            assert loaded[1].error_name == "position_error"
            assert loaded[1].degree == "D1"
        finally:
            Path(path).unlink(missing_ok=True)


class TestCleanTrajectoryIO:
    """Tests for CleanTrajectory save/load."""

    def test_save_load_roundtrip(self):
        """Trajectories survive save/load cycle."""
        from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector

        trajectories = [
            CleanTrajectory(
                trajectory_id="test_traj_0",
                source="demo",
                task_name="pick_place",
                num_steps=10,
                actions=np.random.randn(10, 7),
                states=np.random.randn(11, 50),
                phase_labels=["reach"] * 11,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            CleanTrajectoryCollector.save(trajectories, tmpdir)
            loaded = CleanTrajectoryCollector.load(tmpdir)

            assert len(loaded) == 1
            assert loaded[0].trajectory_id == "test_traj_0"
            assert loaded[0].num_steps == 10
            assert loaded[0].actions.shape == (10, 7)
            assert loaded[0].states.shape == (11, 50)
            assert loaded[0].phase_labels == ["reach"] * 11
