#!/usr/bin/env python
"""
Evaluate Pi0.5 finetuned models on error scenes and clean rollouts.

For each task, this script:
  1. Starts a VLA server (openpi05 env) with the finetuned checkpoint
  2. Loads error scenes from v5_training/{task}/scenes/
  3. Runs the policy from each error state, measures success rate
  4. Runs clean rollouts from initial state for comparison
  5. Saves detailed results JSON

Usage:
  # Single task
  conda activate mimicgen_env
  MUJOCO_GL=egl python eval_pi05_error_scenes.py --task pick_place --gpu 0

  # All tasks (via launch script)
  bash launch_eval_all_tasks.sh
"""

import argparse
import json
import logging
import os
import pickle
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
OPENPI_DIR = Path(
    os.environ.get("OPENPI_DIR")
    or "${BENCHMARK_ROOT}/shared_deps/openpi"
)
CONDA_DIR = Path(
    os.environ.get("CONDA_DIR")
    or Path.home() / "miniconda3"
)
CONDA_BIN = CONDA_DIR / "bin" / "conda"
VLA_SERVER_PY = PROJECT_DIR / "archive" / "v4" / "framework" / "vla_server.py"
SCENES_BASE = PROJECT_DIR / "error_benchmark" / "outputs" / "v5_training"
CHECKPOINT_BASE = OPENPI_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "error_benchmark" / "outputs" / "eval_pi05_error_scenes"

# Ensure framework is importable
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Task definitions
# ═══════════════════════════════════════════════════════════════

TASK_PROMPTS = {
    "pick_place": "pick up the milk, cereal, bread, and can and place them in the correct bins",
    "coffee": "make coffee",
    "stack": "stack the red block on top of the green block",
    "stack_three": "stack three blocks",
    "threading": "insert the needle into the needle hole",
    "three_piece_assembly": "assemble the three pieces",
}

ALL_TASKS = list(TASK_PROMPTS.keys())


def config_name_finetune(task_name: str) -> str:
    return f"pi05_benchmark_{task_name}_merged_finetune"


def config_name_inference(task_name: str) -> str:
    return f"pi05_benchmark_{task_name}_merged_inference"


# ═══════════════════════════════════════════════════════════════
# VLA Server management
# ═══════════════════════════════════════════════════════════════

def find_latest_checkpoint(task_name: str) -> Path:
    """Auto-detect the latest checkpoint step for a task."""
    ckpt_dir = CHECKPOINT_BASE / config_name_finetune(task_name) / "merged"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory: {ckpt_dir}")
    steps = sorted(
        [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=int,
    )
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps in {ckpt_dir}")
    return ckpt_dir / steps[-1]


def start_vla_server(task_name: str, checkpoint_path: Path, port: int, gpu_id: int,
                     log_dir: Path = None) -> tuple:
    """Start VLA server as subprocess in openpi05 env.

    Returns (proc, log_path) tuple. Server stdout/stderr go to a log file
    to avoid subprocess.PIPE buffer deadlock.
    """
    cfg_name = config_name_inference(task_name)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        str(CONDA_BIN), "run", "-n", "openpi05",
        "python", str(VLA_SERVER_PY),
        "--model_type", "pi05",
        "--config_name", cfg_name,
        "--checkpoint", str(checkpoint_path),
        "--port", str(port),
        "--device", "cuda:0",
    ]
    logger.info(f"Starting VLA server: {cfg_name} on GPU {gpu_id}, port {port}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Redirect server output to file to avoid PIPE buffer deadlock
    if log_dir is None:
        log_dir = DEFAULT_OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = log_dir / f"vla_server_{task_name}_{port}.log"
    server_log_file = open(server_log_path, "w")
    logger.info(f"Server log: {server_log_path}")

    proc = subprocess.Popen(cmd, stdout=server_log_file, stderr=subprocess.STDOUT, env=env)
    return proc, server_log_path


def wait_for_server(port: int, timeout: int = 180) -> bool:
    """Poll TCP port until VLA server accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("127.0.0.1", port))
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(3)
    return False


def shutdown_vla_server(port: int, proc: subprocess.Popen):
    """Gracefully shutdown VLA server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", port))
        msg = pickle.dumps({"cmd": "shutdown"})
        sock.sendall(struct.pack("!I", len(msg)) + msg)
        sock.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("VLA server stopped.")


# ═══════════════════════════════════════════════════════════════
# Evaluation loops
# ═══════════════════════════════════════════════════════════════

def evaluate_error_scenes(env, env_wrapper, policy, scenes_dir: Path,
                          task_name: str, max_steps: int = 500) -> dict:
    """Run policy on each error scene and measure recovery success."""
    scene_jsons = sorted(scenes_dir.glob("*.json"))
    logger.info(f"Found {len(scene_jsons)} error scenes in {scenes_dir}")

    results = []
    successes = 0

    for idx, json_path in enumerate(scene_jsons):
        scene_id = json_path.stem
        npz_path = json_path.with_suffix(".npz")

        # Load metadata
        with open(json_path) as f:
            meta = json.load(f)

        error_spec = meta.get("error_spec", {})
        labels = meta.get("labels", {})
        error_name = labels.get("error_name", error_spec.get("error_name", "unknown"))
        degree = labels.get("degree", error_spec.get("degree", "unknown"))
        subtype_id = labels.get("subtype_id", f"{error_name}_{degree}")

        if not npz_path.exists():
            logger.warning(f"  [{idx+1}/{len(scene_jsons)}] {scene_id}: NPZ missing, skip")
            results.append({
                "scene_id": scene_id, "subtype_id": subtype_id,
                "error_name": error_name, "degree": degree,
                "success": False, "steps": 0, "error": "npz_missing",
            })
            continue

        try:
            # Load sim state (prefer post-injection stable state for evaluation)
            npz_data = np.load(str(npz_path))
            if "post_sim_state" in npz_data:
                sim_state = npz_data["post_sim_state"]
            else:
                sim_state = npz_data["sim_state"]

            # Reset env to clear state, then restore error state
            obs = env.reset()
            env_wrapper.set_sim_state_flat(sim_state)
            # forward() is called inside set_sim_state_flat

            # Warm-up step to populate observation cache (especially camera images)
            obs, _, _, _ = env_wrapper.step(env_wrapper.get_neutral_action())

            # Verify we're in an error state (optional sanity check)
            already_success = env_wrapper.check_success()

            # Run policy
            policy.start_episode()
            success = already_success  # rare: some scenes might already be "solved"
            steps_taken = 0

            for step in range(max_steps):
                policy_result = policy.predict_from_obs(obs)
                obs, reward, done, info = env_wrapper.step(policy_result.action)
                steps_taken = step + 1

                if env_wrapper.check_success():
                    success = True
                    break

            if success:
                successes += 1

            results.append({
                "scene_id": scene_id,
                "subtype_id": subtype_id,
                "error_name": error_name,
                "degree": degree,
                "success": success,
                "steps": steps_taken,
                "already_success": already_success,
            })

            status = "OK" if success else "FAIL"
            logger.info(f"  [{idx+1}/{len(scene_jsons)}] {scene_id} ({subtype_id}): "
                        f"{status} in {steps_taken} steps")

        except Exception as e:
            logger.error(f"  [{idx+1}/{len(scene_jsons)}] {scene_id}: ERROR {e}")
            results.append({
                "scene_id": scene_id, "subtype_id": subtype_id,
                "error_name": error_name, "degree": degree,
                "success": False, "steps": 0, "error": str(e),
            })

    # Aggregate
    total = len(results)
    overall_sr = successes / total if total > 0 else 0.0

    # By subtype
    by_subtype = {}
    for r in results:
        sid = r["subtype_id"]
        if sid not in by_subtype:
            by_subtype[sid] = {"total": 0, "successes": 0}
        by_subtype[sid]["total"] += 1
        if r["success"]:
            by_subtype[sid]["successes"] += 1
    for v in by_subtype.values():
        v["sr"] = v["successes"] / v["total"] if v["total"] > 0 else 0.0

    # By degree
    by_degree = {}
    for r in results:
        d = r["degree"]
        if d not in by_degree:
            by_degree[d] = {"total": 0, "successes": 0}
        by_degree[d]["total"] += 1
        if r["success"]:
            by_degree[d]["successes"] += 1
    for v in by_degree.values():
        v["sr"] = v["successes"] / v["total"] if v["total"] > 0 else 0.0

    # By error_name
    by_error_name = {}
    for r in results:
        en = r["error_name"]
        if en not in by_error_name:
            by_error_name[en] = {"total": 0, "successes": 0}
        by_error_name[en]["total"] += 1
        if r["success"]:
            by_error_name[en]["successes"] += 1
    for v in by_error_name.values():
        v["sr"] = v["successes"] / v["total"] if v["total"] > 0 else 0.0

    return {
        "overall_sr": overall_sr,
        "total": total,
        "successes": successes,
        "by_subtype": by_subtype,
        "by_degree": by_degree,
        "by_error_name": by_error_name,
        "per_scene": results,
    }


def evaluate_clean_rollouts(env, env_wrapper, policy, num_rollouts: int = 50,
                            max_steps: int = 500) -> dict:
    """Run clean rollouts from initial state."""
    logger.info(f"Running {num_rollouts} clean rollouts...")
    results = []
    successes = 0

    for i in range(num_rollouts):
        obs = env.reset()
        policy.start_episode()
        success = False
        steps_taken = 0

        for step in range(max_steps):
            policy_result = policy.predict_from_obs(obs)
            obs, reward, done, info = env_wrapper.step(policy_result.action)
            steps_taken = step + 1

            if env_wrapper.check_success():
                success = True
                break

        if success:
            successes += 1

        results.append({
            "rollout_id": i,
            "success": success,
            "steps": steps_taken,
        })
        status = "OK" if success else "FAIL"
        logger.info(f"  Clean [{i+1}/{num_rollouts}]: {status} in {steps_taken} steps")

    sr = successes / num_rollouts if num_rollouts > 0 else 0.0
    return {
        "sr": sr,
        "total": num_rollouts,
        "successes": successes,
        "per_rollout": results,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 on error scenes")
    parser.add_argument("--task", type=str, required=True, choices=ALL_TASKS)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for VLA server")
    parser.add_argument("--port", type=int, default=None, help="VLA server port (default: 5560+gpu)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per rollout")
    parser.add_argument("--num_clean", type=int, default=50, help="Number of clean rollouts")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_error", action="store_true")
    parser.add_argument("--scenes_dir", type=str, default=None, help="Override scenes directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()

    task_name = args.task
    gpu_id = args.gpu
    port = args.port or (5560 + gpu_id)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(task_name)
    checkpoint_step = checkpoint_path.name
    logger.info(f"Task: {task_name}, GPU: {gpu_id}, Port: {port}")
    logger.info(f"Checkpoint: {checkpoint_path} (step {checkpoint_step})")

    # Resolve scenes directory
    if args.scenes_dir:
        scenes_dir = Path(args.scenes_dir)
    else:
        scenes_dir = SCENES_BASE / task_name / "scenes"

    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        sys.exit(1)

    # Start VLA server
    server_proc, server_log_path = start_vla_server(
        task_name, checkpoint_path, port, gpu_id, log_dir=output_dir / "logs"
    )
    logger.info("Waiting for VLA server to start...")

    server_timeout = int(os.environ.get("VLA_SERVER_TIMEOUT", "1200"))
    if not wait_for_server(port, timeout=server_timeout):
        logger.error("VLA server failed to start within timeout")
        # Read server log for diagnostics
        try:
            log_tail = server_log_path.read_text()[-2000:]
            logger.error(f"Server log tail:\n{log_tail}")
        except Exception:
            pass
        if server_proc.poll() is None:
            server_proc.kill()
        server_proc.wait()
        sys.exit(1)
    logger.info("VLA server ready.")

    try:
        # Create environment
        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.framework.policy_adapter import PolicyServerAdapter

        task_reg = load_task_registry(task_name)
        task_config_path = os.path.join(str(PROJECT_DIR), task_reg["task_config"])
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)

        logger.info("Creating robosuite environment (cameras enabled)...")
        env = create_env(task_config, task_reg["dataset_path"],
                         enable_camera=True, camera_resolution=256)
        env_wrapper = EnvWrapper(env, task_config)

        # Create policy adapter (120s timeout for first inference JIT warmup)
        policy = PolicyServerAdapter(
            host="localhost",
            port=port,
            task_prompt=TASK_PROMPTS[task_name],
            replan_interval=5,
            connection_timeout=120.0,
        )

        # Warmup: first inference triggers JIT compilation and is very slow
        logger.info("Warming up VLA model (first inference is slow due to JIT)...")
        warmup_obs = env.reset()
        policy.start_episode()
        try:
            _ = policy.predict_from_obs(warmup_obs)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed: {e} — continuing anyway")

        # Results container
        combined_results = {
            "task": task_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint_step) if checkpoint_step.isdigit() else checkpoint_step,
            "timestamp": datetime.now().isoformat(),
            "max_steps": args.max_steps,
            "gpu": gpu_id,
            "port": port,
        }

        # Error scene evaluation
        if not args.skip_error:
            logger.info(f"\n{'='*60}")
            logger.info(f"Error Scene Evaluation: {task_name}")
            logger.info(f"{'='*60}")
            error_results = evaluate_error_scenes(
                env, env_wrapper, policy, scenes_dir, task_name, args.max_steps
            )
            combined_results["error_scenes"] = error_results
            logger.info(f"\nError Scene SR: {error_results['overall_sr']*100:.1f}% "
                        f"({error_results['successes']}/{error_results['total']})")

            # Print per-subtype summary
            logger.info("\nPer-subtype breakdown:")
            for st, sv in sorted(error_results["by_subtype"].items()):
                logger.info(f"  {st:40s} {sv['sr']*100:5.1f}% ({sv['successes']}/{sv['total']})")

        # Clean rollout evaluation
        if not args.skip_clean:
            logger.info(f"\n{'='*60}")
            logger.info(f"Clean Rollout Evaluation: {task_name}")
            logger.info(f"{'='*60}")
            clean_results = evaluate_clean_rollouts(
                env, env_wrapper, policy, args.num_clean, args.max_steps
            )
            combined_results["clean_rollouts"] = clean_results
            logger.info(f"\nClean SR: {clean_results['sr']*100:.1f}% "
                        f"({clean_results['successes']}/{clean_results['total']})")

        # Save results
        result_path = output_dir / f"{task_name}.json"
        with open(result_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {result_path}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {task_name} (step {checkpoint_step})")
        logger.info(f"{'='*60}")
        if "error_scenes" in combined_results:
            es = combined_results["error_scenes"]
            logger.info(f"  Error Scene SR: {es['overall_sr']*100:.1f}% ({es['successes']}/{es['total']})")
        if "clean_rollouts" in combined_results:
            cr = combined_results["clean_rollouts"]
            logger.info(f"  Clean SR:       {cr['sr']*100:.1f}% ({cr['successes']}/{cr['total']})")
        if "error_scenes" in combined_results and "clean_rollouts" in combined_results:
            delta = (es['overall_sr'] - cr['sr']) * 100
            logger.info(f"  Delta:          {delta:+.1f}%")

        # Cleanup policy
        policy.close()

    finally:
        shutdown_vla_server(port, server_proc)


if __name__ == "__main__":
    main()
