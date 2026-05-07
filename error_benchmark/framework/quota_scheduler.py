#!/usr/bin/env python
"""
Quota Scheduler - v5.0

Unified scheduling for injection opportunities with two modes:

    mode='uniform' (Training Set):
        Timeline-based uniform sampling. Merges adjacent injectable frames
        into segments, then uses linspace for even spacing across time.
        Target: 10 scenes / error / task.

    mode='random' (Validation Set):
        Random sampling with replacement from the available opportunity pool.
        Target: 1000 scenes / error / task. Supports iterative batch scheduling.

This is Step 0c of the v5 pipeline:
    Step 0: Collect clean trajectories
    Step 0b: Scan opportunities
    Step 0c: Schedule injections (this module)  <--
    Step 1: Execute injections
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import json

from error_benchmark.framework.core import (
    InjectionOpportunity, QuotaStatus,
)
from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes

logger = logging.getLogger(__name__)


class QuotaScheduler:
    """
    Unified injection opportunity scheduler with dual-mode sampling.

    Modes:
        'uniform'  — 3D uniform bucketing (subtype × object × phase_group),
                     or timeline-based uniform sampling for training data.
        'random'   — Random sampling with replacement for mass generation.

    Usage:
        # Training mode
        scheduler = QuotaScheduler(mode='uniform', target_per_subtype=10)
        schedule = scheduler.create_schedule(opportunities)

        # Validation mode (iterative)
        scheduler = QuotaScheduler(mode='random', target_per_subtype=1000, batch_size=200)
        batch = scheduler.create_batch(opp_by_subtype, existing_counts)
    """

    def __init__(
        self,
        mode: str = 'uniform',
        target_per_subtype: int = 100,
        max_per_trajectory: int = 10,
        seed: int = 42,
        batch_size: int = 200,
        timeline_merge_gap: int = 10,
    ):
        """
        Args:
            mode: 'uniform' for training, 'random' for validation/mass-gen
            target_per_subtype: Target number of scenes per subtype
            max_per_trajectory: Maximum injections from a single trajectory (uniform mode)
            seed: Random seed
            batch_size: Batch size per subtype per iteration (random mode)
            timeline_merge_gap: Frame gap threshold for timeline merging (uniform mode)
        """
        self.mode = mode
        self.target_per_subtype = target_per_subtype
        self.max_per_trajectory = max_per_trajectory
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.timeline_merge_gap = timeline_merge_gap

    # ─── Unified entry point ─────────────────────────────

    def create_schedule(
        self,
        opportunities: List[InjectionOpportunity],
        existing_quota: Optional[QuotaStatus] = None,
    ) -> List[InjectionOpportunity]:
        """
        Create a schedule based on the current mode.

        Args:
            opportunities: All available injection opportunities
            existing_quota: Optional existing quota status (for incremental scheduling)

        Returns:
            List of scheduled InjectionOpportunity objects
        """
        if self.mode == 'uniform':
            return self._schedule_uniform(opportunities, existing_quota)
        elif self.mode == 'random':
            by_subtype = self.group_by_subtype(opportunities)
            existing_counts = {}
            if existing_quota:
                for error_name, degree in get_all_subtypes():
                    sid = f"{error_name}_{degree}"
                    existing_counts[sid] = existing_quota.get_count(error_name, degree)
            return self.create_batch(by_subtype, existing_counts)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'uniform' or 'random'.")

    # ─── Uniform mode (Training Set) ─────────────────────

    def _schedule_uniform(
        self,
        opportunities: List[InjectionOpportunity],
        existing_quota: Optional[QuotaStatus] = None,
    ) -> List[InjectionOpportunity]:
        """
        3D uniform bucketing schedule (subtype × object × phase_group).

        Distributes quota evenly across available (object, phase_group) buckets
        within each subtype. If a bucket has fewer opportunities than its share,
        overflow is redistributed to other buckets in the same subtype.
        """
        quota = existing_quota or QuotaStatus(
            quota_per_subtype=self.target_per_subtype
        )

        schedule = []
        per_trajectory_count: Dict[str, int] = {}

        by_subtype: Dict[str, List[InjectionOpportunity]] = defaultdict(list)
        for opp in opportunities:
            by_subtype[opp.subtype_id()].append(opp)

        all_subtypes = get_all_subtypes()
        needs = []
        for error_name, degree in all_subtypes:
            subtype_id = f"{error_name}_{degree}"
            remaining = quota.remaining(error_name, degree)
            if remaining > 0 and subtype_id in by_subtype:
                needs.append((subtype_id, error_name, degree, remaining))

        needs.sort(key=lambda x: -x[3])

        for subtype_id, error_name, degree, remaining in needs:
            candidates = by_subtype[subtype_id]

            # Sub-bucket by (target_object, phase_group)
            buckets: Dict[Tuple[str, str], List[InjectionOpportunity]] = defaultdict(list)
            for opp in candidates:
                target_obj = opp.metadata.get('target_object', '') if opp.metadata else ''
                phase_grp = opp.metadata.get('phase_group', '') if opp.metadata else ''
                buckets[(target_obj, phase_grp)].append(opp)

            for key in buckets:
                self.rng.shuffle(buckets[key])

            valid_keys = [k for k in buckets if len(buckets[k]) > 0]
            if not valid_keys:
                continue

            per_bucket = remaining // len(valid_keys)
            extra = remaining % len(valid_keys)

            allocated: Dict[Tuple[str, str], int] = {}
            overflow = 0
            for i, key in enumerate(valid_keys):
                target = per_bucket + (1 if i < extra else 0)
                available = len(buckets[key])
                actual = min(target, available)
                allocated[key] = actual
                overflow += (target - actual)

            if overflow > 0:
                for key in valid_keys:
                    if overflow <= 0:
                        break
                    room = len(buckets[key]) - allocated[key]
                    give = min(room, overflow)
                    allocated[key] += give
                    overflow -= give

            for key in valid_keys:
                count = 0
                for opp in buckets[key]:
                    if count >= allocated[key]:
                        break
                    traj_count = per_trajectory_count.get(opp.trajectory_id, 0)
                    if traj_count >= self.max_per_trajectory:
                        continue
                    schedule.append(opp)
                    per_trajectory_count[opp.trajectory_id] = traj_count + 1
                    count += 1

        logger.info(
            f"[uniform] Scheduled {len(schedule)} injections "
            f"({len(set(o.subtype_id() for o in schedule))} subtypes)"
        )
        return schedule

    # ─── Timeline-based uniform sampling (Training) ──────

    def schedule_timeline(
        self,
        opportunities: List[InjectionOpportunity],
        scenes_per_subtype: Optional[int] = None,
    ) -> Dict[str, List[InjectionOpportunity]]:
        """
        Timeline-based uniform sampling for training data.

        Groups opportunities by subtype, builds virtual timeline per subtype,
        then uniformly samples n points from each timeline.

        Args:
            opportunities: All available opportunities
            scenes_per_subtype: Override for self.target_per_subtype

        Returns:
            Dict mapping subtype_id → list of sampled opportunities
        """
        n = scenes_per_subtype or self.target_per_subtype
        by_subtype = self.group_by_subtype(opportunities)
        result = {}

        for subtype_id, opps in by_subtype.items():
            timeline = self.build_virtual_timeline(opps)
            sampled = self.sample_from_timeline(timeline, n)
            if sampled:
                result[subtype_id] = sampled

        total = sum(len(v) for v in result.values())
        logger.info(
            f"[timeline] Sampled {total} points across "
            f"{len(result)} subtypes ({n}/subtype)"
        )
        return result

    @staticmethod
    def build_virtual_timeline(
        opportunities: List[InjectionOpportunity],
        merge_gap: int = 10,
    ) -> List[InjectionOpportunity]:
        """
        Merge injectable frames into continuous segments, then concatenate.

        1. Group by trajectory_id
        2. Sort by frame_index within each trajectory
        3. Merge adjacent frames (gap <= merge_gap) into segments
        4. Concatenate all segments

        Args:
            opportunities: List of InjectionOpportunity for one subtype.
            merge_gap: Maximum frame gap to merge into one segment.

        Returns:
            List of InjectionOpportunity in virtual-timeline order.
        """
        if not opportunities:
            return []

        by_traj = defaultdict(list)
        for opp in opportunities:
            by_traj[opp.trajectory_id].append(opp)

        timeline = []
        for tid in sorted(by_traj.keys()):
            opps = sorted(by_traj[tid], key=lambda o: o.frame_index)
            segment = [opps[0]]
            for i in range(1, len(opps)):
                if opps[i].frame_index - opps[i - 1].frame_index <= merge_gap:
                    segment.append(opps[i])
                else:
                    timeline.extend(segment)
                    segment = [opps[i]]
            timeline.extend(segment)

        return timeline

    @staticmethod
    def sample_from_timeline(
        timeline: List[InjectionOpportunity],
        n: int = 10,
    ) -> List[InjectionOpportunity]:
        """Uniformly sample n points from the virtual timeline using linspace."""
        if not timeline:
            return []
        indices = np.round(np.linspace(0, len(timeline) - 1, n)).astype(int)
        return [timeline[i] for i in indices]

    # ─── Random mode (Validation Set / Mass Generation) ──

    def create_batch(
        self,
        opp_by_subtype: Dict[str, List[InjectionOpportunity]],
        existing_counts: Dict[str, int],
    ) -> List[InjectionOpportunity]:
        """
        Build a batch of opportunities for one iteration (random mode).

        For each unfulfilled subtype, samples `batch_size` opportunities
        with replacement from the available pool.

        Args:
            opp_by_subtype: Dict mapping subtype_id → list of opportunities
            existing_counts: Dict mapping subtype_id → current count

        Returns:
            Shuffled flat list of sampled opportunities
        """
        batch = []
        all_subtypes = get_all_subtypes()

        for error_name, degree in all_subtypes:
            subtype_id = f"{error_name}_{degree}"
            current = existing_counts.get(subtype_id, 0)
            if current >= self.target_per_subtype:
                continue

            pool = opp_by_subtype.get(subtype_id, [])
            if not pool:
                continue

            need = min(self.batch_size, self.target_per_subtype - current)
            indices = self.rng.randint(0, len(pool), size=need)
            for idx in indices:
                batch.append(pool[idx])

        self.rng.shuffle(batch)
        logger.info(f"[random] Created batch of {len(batch)} opportunities")
        return batch

    # ─── Shared utilities ────────────────────────────────

    @staticmethod
    def group_by_subtype(
        opportunities: List[InjectionOpportunity],
    ) -> Dict[str, List[InjectionOpportunity]]:
        """Group opportunities by subtype_id."""
        by_subtype: Dict[str, List[InjectionOpportunity]] = defaultdict(list)
        for opp in opportunities:
            by_subtype[opp.subtype_id()].append(opp)
        return by_subtype

    def estimate_coverage(
        self,
        opportunities: List[InjectionOpportunity],
    ) -> Dict[str, Dict]:
        """
        Estimate which subtypes can be filled and which will fall short.

        Returns:
            Dict mapping subtype_id -> {available, target, fillable, gap}
        """
        by_subtype: Dict[str, int] = {}
        for opp in opportunities:
            key = opp.subtype_id()
            by_subtype[key] = by_subtype.get(key, 0) + 1

        coverage = {}
        for error_name, degree in get_all_subtypes():
            subtype_id = f"{error_name}_{degree}"
            available = by_subtype.get(subtype_id, 0)
            coverage[subtype_id] = {
                'available': available,
                'target': self.target_per_subtype,
                'fillable': min(available, self.target_per_subtype),
                'gap': max(0, self.target_per_subtype - available),
            }

        return coverage

    @staticmethod
    def save_schedule(
        schedule: List[InjectionOpportunity],
        output_path: str,
    ):
        """Save schedule to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for opp in schedule:
                f.write(json.dumps(opp.to_dict()) + '\n')

        logger.info(f"Saved schedule ({len(schedule)} items) to {output_path}")

    @staticmethod
    def load_schedule(input_path: str) -> List[InjectionOpportunity]:
        """Load schedule from JSONL file."""
        schedule = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    schedule.append(InjectionOpportunity.from_dict(d))

        logger.info(f"Loaded schedule ({len(schedule)} items) from {input_path}")
        return schedule
