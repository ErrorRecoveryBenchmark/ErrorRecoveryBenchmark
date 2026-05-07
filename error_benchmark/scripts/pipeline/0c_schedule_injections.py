#!/usr/bin/env python
"""
v5.0 Pipeline Step 0c: Schedule Injections

Selects injection opportunities to fill quota targets.

Usage:
    python error_benchmark/scripts/pipeline/0c_schedule_injections.py \
        --config error_benchmark/configs/benchmark_v5.yaml \
        --task pick_place \
        --opportunities error_benchmark/outputs/v5/opportunity_maps/pick_place/opportunities.jsonl \
        --target 100

Output:
    {output_dir}/schedules/{task}/schedule.jsonl
    {output_dir}/schedules/{task}/coverage_report.json
"""

import argparse
import logging
import json
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from error_benchmark.framework.opportunity_scanner import OpportunityScanner
from error_benchmark.framework.quota_scheduler import QuotaScheduler
from error_benchmark.framework.core import QuotaStatus
from error_benchmark.framework.logger_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="v5 Step 0c: Schedule injections")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/benchmark_v5.yaml")
    parser.add_argument("--task", type=str, default="pick_place")
    parser.add_argument("--opportunities", type=str, default=None)
    parser.add_argument("--target", type=int, default=None,
                        help="Target episodes per subtype")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Paths
    opp_path = args.opportunities or (
        f"error_benchmark/outputs/v5/opportunity_maps/{args.task}/opportunities.jsonl"
    )
    base_sched_dir = config['paths'].get(
        'schedules_dir',
        "error_benchmark/outputs/v5/schedules"
    )
    output_dir = args.output_dir or f"{base_sched_dir}/{args.task}"

    target = args.target or config.get('quota', {}).get('target_per_subtype', 100)

    logger.info(f"=== v5 Step 0c: Schedule Injections ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Target per subtype: {target}")

    # Load opportunities
    full_opp_path = PROJECT_ROOT / opp_path
    opportunities = OpportunityScanner.load(str(full_opp_path))
    logger.info(f"Loaded {len(opportunities)} opportunities")

    # Create scheduler
    scheduler = QuotaScheduler(
        target_per_subtype=target,
        max_per_trajectory=50,
        seed=args.seed,
    )

    # Estimate coverage
    coverage = scheduler.estimate_coverage(opportunities)
    full_output_dir = PROJECT_ROOT / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = full_output_dir / "coverage_report.json"
    with open(coverage_path, 'w') as f:
        json.dump(coverage, f, indent=2)
    logger.info(f"Coverage report saved to {coverage_path}")

    # Print coverage summary
    fillable = sum(1 for v in coverage.values() if v['fillable'] >= target)
    partial = sum(1 for v in coverage.values() if 0 < v['fillable'] < target)
    empty = sum(1 for v in coverage.values() if v['available'] == 0)
    logger.info(f"\nCoverage: {fillable} full, {partial} partial, {empty} empty")

    # Create schedule
    schedule = scheduler.create_schedule(opportunities)

    # Save schedule
    schedule_path = full_output_dir / "schedule.jsonl"
    QuotaScheduler.save_schedule(schedule, str(schedule_path))

    # Summary
    by_subtype = {}
    for opp in schedule:
        key = opp.subtype_id()
        by_subtype[key] = by_subtype.get(key, 0) + 1

    logger.info(f"\n=== Schedule Summary ===")
    logger.info(f"Total scheduled: {len(schedule)}")
    logger.info(f"Subtypes: {len(by_subtype)}")
    for key, count in sorted(by_subtype.items(), key=lambda x: -x[1]):
        logger.info(f"  {key}: {count}")


if __name__ == "__main__":
    main()
