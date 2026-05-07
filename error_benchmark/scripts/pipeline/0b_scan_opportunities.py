#!/usr/bin/env python
"""
v5.0 Pipeline Step 0b: Scan Injection Opportunities

Offline scanner that replays each clean trajectory and calls every
skill's can_inject() at every frame.

Usage:
    python error_benchmark/scripts/pipeline/0b_scan_opportunities.py \
        --config error_benchmark/configs/benchmark_v5.yaml \
        --task pick_place \
        --trajectories_dir error_benchmark/outputs/v5/clean_trajectories/pick_place

Output:
    {output_dir}/opportunity_maps/{task}/opportunities.jsonl
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
from error_benchmark.framework.opportunity_scanner import OpportunityScanner
from error_benchmark.framework.error_skills import get_all_skills
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry, validate_trajectories_for_task


def main():
    parser = argparse.ArgumentParser(description="v5 Step 0b: Scan injection opportunities")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/benchmark_v5.yaml")
    parser.add_argument("--task", type=str, default="pick_place")
    parser.add_argument("--trajectories_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--min_frame_gap", type=int, default=None)
    parser.add_argument("--degrees", nargs="+", default=None,
                        help="Filter to specific degrees (e.g., --degrees D0)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Paths
    task_info = load_task_registry(args.task)
    dataset_path = task_info['dataset_path']

    traj_dir = args.trajectories_dir or (
        f"error_benchmark/outputs/v5/clean_trajectories/{args.task}"
    )
    base_opp_dir = config['paths'].get(
        'opportunity_maps_dir',
        "error_benchmark/outputs/v5/opportunity_maps"
    )
    output_dir = args.output_dir or f"{base_opp_dir}/{args.task}"

    traj_path = PROJECT_ROOT / traj_dir
    output_path = PROJECT_ROOT / output_dir

    min_frame_gap = args.min_frame_gap or config.get('scanning', {}).get('min_frame_gap', 10)

    logger.info(f"=== v5 Step 0b: Scan Injection Opportunities ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Trajectories: {traj_path}")
    logger.info(f"Output: {output_path}")

    # Load trajectories
    trajectories = CleanTrajectoryCollector.load(str(traj_path))
    validate_trajectories_for_task(trajectories, args.task)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Load task config and create environment
    task_config_path = task_info.get('task_config', config['paths']['task_config'])
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)
    env = create_env(task_config, dataset_path)

    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    # Create skills
    skills_config = config.get('error_skills', {})
    enabled_skills = config.get('scanning', {}).get('enabled_skills', None)
    all_skills = get_all_skills(skills_config)

    if enabled_skills:
        all_skills = [s for s in all_skills if s.name in enabled_skills]

    logger.info(f"Using {len(all_skills)} skills: {[s.name for s in all_skills]}")

    # Scan
    scanner = OpportunityScanner(env_wrapper, all_skills, min_frame_gap=min_frame_gap)
    opportunities = scanner.scan_all(trajectories)

    # Filter by degree if requested
    if args.degrees:
        allowed = set(args.degrees)
        before = len(opportunities)
        opportunities = [o for o in opportunities if o.degree in allowed]
        logger.info(f"Degree filter {args.degrees}: {before} -> {len(opportunities)} opportunities")

    # Save
    output_file = output_path / "opportunities.jsonl"
    OpportunityScanner.save(opportunities, str(output_file))

    # Print summary
    by_subtype = {}
    for opp in opportunities:
        key = opp.subtype_id()
        by_subtype[key] = by_subtype.get(key, 0) + 1

    logger.info(f"\n=== Scan Summary ===")
    logger.info(f"Total opportunities: {len(opportunities)}")
    logger.info(f"Subtypes found: {len(by_subtype)}")
    for key, count in sorted(by_subtype.items(), key=lambda x: -x[1]):
        logger.info(f"  {key}: {count}")


if __name__ == "__main__":
    main()
