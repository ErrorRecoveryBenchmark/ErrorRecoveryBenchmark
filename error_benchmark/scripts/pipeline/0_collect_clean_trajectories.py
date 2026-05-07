#!/usr/bin/env python
"""
v5.0 Pipeline Step 0: Collect Clean Trajectories

Collects clean (successful) trajectories from:
    - Human demonstration HDF5 datasets
    - Successful VLA/BC policy rollouts (optional)

Usage:
    python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py \
        --config error_benchmark/configs/benchmark_v5.yaml \
        --task pick_place \
        --num_demos 20

Output:
    {output_dir}/clean_trajectories/
        manifest.json
        demo_pick_place_demo_0.npz
        demo_pick_place_demo_1.npz
        ...
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry


def main():
    parser = argparse.ArgumentParser(description="v5 Step 0: Collect clean trajectories")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/benchmark_v5.yaml")
    parser.add_argument("--task", type=str, default="pick_place",
                        help="Task name from task_registry.yaml")
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Override number of demos to collect")
    parser.add_argument("--policy", type=str, default=None,
                        choices=["bc_rnn", "vla_pi0", "vla_pi05"],
                        help="Also collect successful rollouts from this policy")
    parser.add_argument("--min_successes", type=int, default=10,
                        help="Min successful rollouts to collect (default: 10)")
    parser.add_argument("--max_attempts", type=int, default=100,
                        help="Max rollout attempts (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_phases", action="store_true",
                        help="Replay trajectories to label task phases")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load task registry
    task_info = load_task_registry(args.task)
    dataset_path = task_info['dataset_path']

    # Determine output directory (per-task subdirectory)
    base_output_dir = args.output_dir or config['paths'].get(
        'clean_trajectories_dir',
        "error_benchmark/outputs/v5/clean_trajectories"
    )
    # Append task name if not already in path
    if not base_output_dir.endswith(args.task):
        output_dir = f"{base_output_dir}/{args.task}"
    else:
        output_dir = base_output_dir
    output_path = PROJECT_ROOT / output_dir

    # Number of demos
    num_demos = args.num_demos or config.get('clean_trajectories', {}).get(
        'sources', {}
    ).get('demo', {}).get('num_demos', 20)

    logger.info(f"=== v5 Step 0: Collect Clean Trajectories ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Num demos: {num_demos}")
    logger.info(f"Output: {output_path}")

    # Load task config from YAML
    task_config_path = task_info.get('task_config', config['paths']['task_config'])
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)

    # Create environment
    env = create_env(task_config, dataset_path)

    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    # Collect trajectories
    collector = CleanTrajectoryCollector(env_wrapper, task_config, task_name=args.task)
    trajectories = collector.collect_from_demos(
        dataset_path=dataset_path,
        num_demos=num_demos,
        seed=args.seed,
    )

    # Collect policy rollouts if requested
    if args.policy:
        logger.info(f"Collecting rollouts from policy: {args.policy}")

        if args.policy == "bc_rnn":
            bc_ckpt = task_info.get('bc_rnn_checkpoint')
            if not bc_ckpt:
                logger.error("No bc_rnn_checkpoint in task_registry for %s", args.task)
                return 1
            from error_benchmark.framework.policy_adapter import RobomimicPolicyAdapter
            policy_adapter = RobomimicPolicyAdapter(
                ckpt_path=bc_ckpt, device="cuda:0", name="bc_rnn", seed=args.seed,
            )
        else:
            logger.error("Policy %s not yet supported in step 0", args.policy)
            return 1

        rollout_trajs = collector.collect_from_rollouts(
            policy_adapter=policy_adapter,
            min_successes=args.min_successes,
            max_attempts=args.max_attempts,
            policy_name=args.policy,
            seed=args.seed,
        )
        trajectories.extend(rollout_trajs)
        logger.info(f"Added {len(rollout_trajs)} rollout trajectories")

    # Optionally label phases
    if args.label_phases:
        logger.info("Labeling task phases for each trajectory...")
        for i, traj in enumerate(tqdm(trajectories, desc="Labeling phases")):
            trajectories[i] = collector.replay_and_label_phases(traj)

        logger.info("Segmenting trajectories by object interaction...")
        for i, traj in enumerate(tqdm(trajectories, desc="Segmenting")):
            trajectories[i] = collector.segment_interactions(traj)

    # Save
    CleanTrajectoryCollector.save(trajectories, str(output_path))

    logger.info(f"\n=== Done ===")
    logger.info(f"Collected {len(trajectories)} clean trajectories")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
