#!/usr/bin/env python
"""
v5.0 Master Pipeline: Run all v5 steps for all tasks.

Orchestrates:
    Step 0:  Collect clean trajectories (demos + optional BC-RNN rollouts)
    Step 0b: Scan injection opportunities
    Step 0c: Schedule injections (quota-driven)
    Step 1:  Execute injections (with optional video rendering)

Usage:
    # Smoke test (pick_place only, small scale)
    python error_benchmark/scripts/pipeline/run_v5_all_tasks.py \
        --tasks pick_place --num_demos 2 --min_successes 2 \
        --max_attempts 10 --target_per_subtype 3

    # Full run (all tasks, 10 per subtype, with BC-RNN rollouts)
    python error_benchmark/scripts/pipeline/run_v5_all_tasks.py \
        --with_policy bc_rnn --num_demos 20 --min_successes 10 \
        --target_per_subtype 10
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        items = list(iterable)
        n = len(items)
        for i, item in enumerate(items):
            print(f"[{desc}] {i+1}/{n}: {item}", flush=True)
            yield item

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Robosuite/MimicGen paths required by child processes
MIMICGEN_WS = str(PROJECT_ROOT / "shared" / "mimicgen_workspace")
_EXTRA_PATHS = [
    str(PROJECT_ROOT),
    os.path.join(MIMICGEN_WS, "robosuite"),
    os.path.join(MIMICGEN_WS, "mimicgen"),
]

ALL_TASKS = [
    "pick_place", "coffee", "stack",
    "stack_three", "threading", "three_piece_assembly",
]

PIPELINE_SCRIPTS = {
    "step0": "error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py",
    "step0b": "error_benchmark/scripts/pipeline/0b_scan_opportunities.py",
    "step0c": "error_benchmark/scripts/pipeline/0c_schedule_injections.py",
    "step1": "error_benchmark/scripts/pipeline/1_v5_execute_injections.py",
}


def run_command(cmd, dry_run=False, logger=None):
    """Run a subprocess command, return (success, elapsed_seconds)."""
    cmd_str = " ".join(cmd)
    if logger:
        logger.info(f"  CMD: {cmd_str}")
    if dry_run:
        return True, 0.0

    start = time.time()
    try:
        # Build env with PYTHONPATH for robosuite/mimicgen
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        extra = os.pathsep.join(_EXTRA_PATHS)
        env["PYTHONPATH"] = f"{extra}{os.pathsep}{existing}" if existing else extra

        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            env=env,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            if logger:
                logger.error(f"  FAILED (exit code {result.returncode}) "
                             f"after {elapsed:.1f}s")
            return False, elapsed
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        if logger:
            logger.error(f"  EXCEPTION: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="v5 Master Pipeline: all tasks, all steps")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/benchmark_v5.yaml")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to run (default: all 6)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from this task (skip earlier tasks)")
    parser.add_argument("--num_demos", type=int, default=20)
    parser.add_argument("--with_policy", type=str, default=None,
                        choices=["bc_rnn"],
                        help="Also collect rollouts from this policy")
    parser.add_argument("--min_successes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=100)
    parser.add_argument("--target_per_subtype", type=int, default=10)
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render videos for each scene (default: True)")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable video rendering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degrees", nargs="+", default=None,
                        help="Filter to specific degrees (e.g., --degrees D0)")
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip step 0 (reuse existing clean trajectories)")
    parser.add_argument("--skip_scan", action="store_true",
                        help="Skip step 0b (reuse existing opportunity maps)")
    parser.add_argument("--skip_schedule", action="store_true",
                        help="Skip step 0c (reuse existing schedules)")
    parser.add_argument("--only_scan", action="store_true",
                        help="Run only step 0 + 0b (collect + scan), then stop")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands only, don't execute")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("v5_master")

    tasks = args.tasks or ALL_TASKS
    if args.resume_from:
        if args.resume_from not in tasks:
            logger.error(f"resume_from task '{args.resume_from}' not in task list")
            return 1
        idx = tasks.index(args.resume_from)
        tasks = tasks[idx:]
        logger.info(f"Resuming from task: {args.resume_from}")

    do_render = args.render and not args.no_render
    python = args.python

    logger.info("=" * 60)
    logger.info("v5 MASTER PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Demos: {args.num_demos}, Policy: {args.with_policy}")
    logger.info(f"Target per subtype: {args.target_per_subtype}")
    logger.info(f"Degrees: {args.degrees or 'all'}")
    logger.info(f"Render: {do_render}")
    logger.info(f"Skip: collect={args.skip_collect}, scan={args.skip_scan}, schedule={args.skip_schedule}")
    logger.info(f"Only scan: {args.only_scan}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    results = {}
    total_start = time.time()

    STEPS = ["step0", "step0b", "step0c", "step1"]
    STEP_NAMES = {
        "step0": "Collect trajectories",
        "step0b": "Scan opportunities",
        "step0c": "Schedule injections",
        "step1": "Execute injections",
    }

    task_pbar = tqdm(tasks, desc="Tasks", unit="task", position=0, leave=True)
    for task in task_pbar:
        task_pbar.set_description(f"Task: {task}")
        task_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK: {task}")
        logger.info(f"{'='*60}")

        task_results = {}

        # Output directories
        base_out = f"error_benchmark/outputs/v5"
        traj_dir = f"{base_out}/clean_trajectories/{task}"
        opp_dir = f"{base_out}/opportunity_maps/{task}"
        sched_dir = f"{base_out}/schedules/{task}"
        scene_dir = f"{base_out}/{task}"

        step_pbar = tqdm(total=4, desc=f"  {task}", unit="step",
                         position=1, leave=False,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]")

        # Step 0: Collect clean trajectories
        if args.skip_collect:
            logger.info(f"\n--- Step 0: SKIPPED (--skip_collect) ---")
            task_results["step0"] = {"ok": True, "elapsed": 0.0}
        else:
            logger.info(f"\n--- Step 0: Collect Clean Trajectories ---")
            step_pbar.set_postfix_str("Collecting trajectories")
            cmd = [
                python, PIPELINE_SCRIPTS["step0"],
                "--config", args.config,
                "--task", task,
                "--num_demos", str(args.num_demos),
                "--output_dir", traj_dir,
                "--seed", str(args.seed),
                "--label_phases",
            ]
            if args.with_policy:
                cmd += [
                    "--policy", args.with_policy,
                    "--min_successes", str(args.min_successes),
                    "--max_attempts", str(args.max_attempts),
                ]
            ok, elapsed = run_command(cmd, args.dry_run, logger)
            task_results["step0"] = {"ok": ok, "elapsed": elapsed}
            step_pbar.update(1)
            if not ok:
                logger.error(f"Step 0 failed for {task}, skipping remaining steps")
                results[task] = task_results
                step_pbar.close()
                continue

        # Step 0b: Scan opportunities
        if args.skip_scan:
            logger.info(f"\n--- Step 0b: SKIPPED (--skip_scan) ---")
            task_results["step0b"] = {"ok": True, "elapsed": 0.0}
        else:
            logger.info(f"\n--- Step 0b: Scan Injection Opportunities ---")
            step_pbar.set_postfix_str("Scanning opportunities")
            cmd = [
                python, PIPELINE_SCRIPTS["step0b"],
                "--config", args.config,
                "--task", task,
                "--trajectories_dir", traj_dir,
                "--output_dir", opp_dir,
            ]
            if args.degrees:
                cmd += ["--degrees"] + args.degrees
            ok, elapsed = run_command(cmd, args.dry_run, logger)
            task_results["step0b"] = {"ok": ok, "elapsed": elapsed}
            step_pbar.update(1)
            if not ok:
                logger.error(f"Step 0b failed for {task}, skipping remaining steps")
                results[task] = task_results
                step_pbar.close()
                continue

        # Early exit if --only_scan
        if args.only_scan:
            task_elapsed = time.time() - task_start
            task_results["total_elapsed"] = task_elapsed
            results[task] = task_results
            step_pbar.close()
            logger.info(f"\nTask {task} scan completed in {task_elapsed:.1f}s (--only_scan)")
            continue

        # Step 0c: Schedule injections
        if args.skip_schedule:
            logger.info(f"\n--- Step 0c: SKIPPED (--skip_schedule) ---")
            task_results["step0c"] = {"ok": True, "elapsed": 0.0}
        else:
            logger.info(f"\n--- Step 0c: Schedule Injections ---")
            step_pbar.set_postfix_str("Scheduling injections")
            cmd = [
                python, PIPELINE_SCRIPTS["step0c"],
                "--config", args.config,
                "--task", task,
                "--opportunities", f"{opp_dir}/opportunities.jsonl",
                "--target", str(args.target_per_subtype),
                "--output_dir", sched_dir,
                "--seed", str(args.seed),
            ]
            ok, elapsed = run_command(cmd, args.dry_run, logger)
            task_results["step0c"] = {"ok": ok, "elapsed": elapsed}
            step_pbar.update(1)
            if not ok:
                logger.error(f"Step 0c failed for {task}, skipping remaining steps")
                results[task] = task_results
                step_pbar.close()
                continue

        # Step 1: Execute injections
        logger.info(f"\n--- Step 1: Execute Injections ---")
        step_pbar.set_postfix_str("Executing injections")
        cmd = [
            python, PIPELINE_SCRIPTS["step1"],
            "--config", args.config,
            "--task", task,
            "--schedule", f"{sched_dir}/schedule.jsonl",
            "--trajectories_dir", traj_dir,
            "--output_dir", scene_dir,
            "--seed", str(args.seed),
        ]
        if do_render:
            cmd.append("--render")
        ok, elapsed = run_command(cmd, args.dry_run, logger)
        task_results["step1"] = {"ok": ok, "elapsed": elapsed}
        step_pbar.update(1)
        step_pbar.close()

        task_elapsed = time.time() - task_start
        task_results["total_elapsed"] = task_elapsed
        results[task] = task_results
        logger.info(f"\nTask {task} completed in {task_elapsed:.1f}s")

    # Summary
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info("")

    for task, res in results.items():
        steps_ok = sum(1 for k, v in res.items()
                       if k.startswith("step") and isinstance(v, dict) and v.get("ok"))
        steps_total = sum(1 for k in res if k.startswith("step"))
        status = "OK" if steps_ok == steps_total else "PARTIAL"
        elapsed = res.get("total_elapsed", 0)
        logger.info(f"  {task}: {status} ({steps_ok}/{steps_total} steps, "
                     f"{elapsed:.1f}s)")

    failed_tasks = [t for t, r in results.items()
                    if any(k.startswith("step") and isinstance(v, dict) and not v.get("ok")
                           for k, v in r.items())]
    if failed_tasks:
        logger.warning(f"\nFailed tasks: {failed_tasks}")
        return 1

    logger.info(f"\nAll tasks completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
