#!/usr/bin/env python
"""Unit tests for QuotaScheduler 3D bucketing."""

import numpy as np
import pytest
from error_benchmark.framework.core import InjectionOpportunity, QuotaStatus
from error_benchmark.framework.quota_scheduler import QuotaScheduler


def _make_opp(error_name, degree, traj_id, frame, target_object, phase_group):
    """Helper to create an InjectionOpportunity with metadata."""
    return InjectionOpportunity(
        trajectory_id=traj_id,
        frame_index=frame,
        error_name=error_name,
        degree=degree,
        task_phase=phase_group,
        metadata={
            'target_object': target_object,
            'phase_group': phase_group,
        },
    )


class TestQuotaScheduler3D:
    def test_single_object_backward_compatible(self):
        """Single-object task: phase bucketing only, no object dimension."""
        opps = []
        for i in range(50):
            opps.append(_make_opp("drop_in_transit", "D0", f"traj_{i%10}",
                                  i*10, "cube", "transfer"))
        for i in range(50):
            opps.append(_make_opp("drop_in_transit", "D0", f"traj_{i%10}",
                                  i*10+5, "cube", "grasp"))

        scheduler = QuotaScheduler(target_per_subtype=20, seed=42)
        schedule = scheduler.create_schedule(opps)

        transfer_count = sum(
            1 for o in schedule
            if o.metadata.get('phase_group') == 'transfer')
        grasp_count = sum(
            1 for o in schedule
            if o.metadata.get('phase_group') == 'grasp')
        assert abs(transfer_count - grasp_count) <= 2
        assert len(schedule) == 20

    def test_multi_object_uniform(self):
        """Multi-object task: each object gets equal share."""
        opps = []
        for obj in ["cubeA", "cubeB", "cubeC"]:
            for i in range(40):
                opps.append(_make_opp("drop_in_transit", "D0",
                                      f"traj_{i%10}", i*10, obj, "transfer"))

        scheduler = QuotaScheduler(target_per_subtype=30, seed=42)
        schedule = scheduler.create_schedule(opps)

        per_obj = {}
        for o in schedule:
            obj = o.metadata.get('target_object', '')
            per_obj[obj] = per_obj.get(obj, 0) + 1

        assert len(per_obj) == 3
        for count in per_obj.values():
            assert count == 10

    def test_overflow_redistribution(self):
        """If one bucket has too few, overflow goes to others."""
        opps = []
        for i in range(5):
            opps.append(_make_opp("drop_in_transit", "D0",
                                  f"traj_{i}", i*10, "cubeA", "transfer"))
        for i in range(40):
            opps.append(_make_opp("drop_in_transit", "D0",
                                  f"traj_{i%10}", i*10, "cubeB", "transfer"))

        scheduler = QuotaScheduler(target_per_subtype=20, seed=42)
        schedule = scheduler.create_schedule(opps)

        per_obj = {}
        for o in schedule:
            obj = o.metadata.get('target_object', '')
            per_obj[obj] = per_obj.get(obj, 0) + 1

        assert per_obj.get("cubeA", 0) == 5
        assert per_obj.get("cubeB", 0) == 15
        assert len(schedule) == 20

    def test_per_trajectory_limit_respected(self):
        """max_per_trajectory still enforced."""
        opps = []
        for i in range(50):
            opps.append(_make_opp("drop_in_transit", "D0",
                                  "traj_0", i, "cube", "transfer"))

        scheduler = QuotaScheduler(
            target_per_subtype=20, max_per_trajectory=5, seed=42)
        schedule = scheduler.create_schedule(opps)
        assert len(schedule) == 5

    def test_empty_metadata_fallback(self):
        """Opportunities without metadata fall into a single default bucket."""
        opps = []
        for i in range(20):
            opps.append(InjectionOpportunity(
                trajectory_id=f"traj_{i%5}",
                frame_index=i*10,
                error_name="drop_in_transit",
                degree="D0",
                task_phase="transport",
            ))

        scheduler = QuotaScheduler(target_per_subtype=10, seed=42)
        schedule = scheduler.create_schedule(opps)
        assert len(schedule) == 10
