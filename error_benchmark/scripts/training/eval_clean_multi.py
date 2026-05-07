#!/usr/bin/env python
"""Multi-worker clean rollouts evaluation.

Runs a batched VLA server and 48 parallel rollout workers.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
VLA_SERVER_PY = PROJECT_DIR / "archive" / "v4" / "framework" / "vla_server.py"
CONDA_BIN = Path(os.environ.get("CONDA_DIR", Path.home() / "miniconda3")) / "bin" / "conda"

TASK_PROMPTS = {
    "threading": "insert the needle into the needle hole",
    "coffee": "make coffee",
    "pick_place": "pick up the milk, cereal, bread, and can and place them in the correct bins",
    "stack": "stack the red block on top of the green block",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def worker_rollout(worker_id: int, port: int, task: str, max_steps: int, prompt: str, output_dir: Path):
    """Run a single clean rollout and save result."""
    result_file = output_dir / f"rollout_{worker_id}.json"

    try:
        sys.path.insert(0, str(PROJECT_DIR))
        sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"))
        sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"))

        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.framework.policy_adapter import PolicyServerAdapter
        import yaml

        task_reg = load_task_registry(task)
        task_config_path = str(PROJECT_DIR) + "/" + task_reg["task_config"]
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)

        env = create_env(task_config, task_reg["dataset_path"], enable_camera=True, camera_resolution=256)
        env_wrapper = EnvWrapper(env, task_config)

        policy = PolicyServerAdapter(
            host="localhost",
            port=port,
            task_prompt=prompt,
            replan_interval=5,
            connection_timeout=30.0,
        )

        obs = env.reset()
        policy.start_episode()
        success = False
        steps_taken = 0

        for step in range(max_steps):
            result = policy.predict_from_obs(obs)
            obs, reward, done, info = env_wrapper.step(result.action)
            steps_taken = step + 1
            if env_wrapper.check_success():
                success = True
                break

        policy.close()

        result = {
            "rollout_id": worker_id,
            "success": success,
            "steps": steps_taken,
        }
        with open(result_file, "w") as f:
            json.dump(result, f)

        status = "OK" if success else "FAIL"
        logger.info(f"  Worker {worker_id}: {status} in {steps_taken} steps")

    except Exception as e:
        logger.error(f"  Worker {worker_id}: ERROR {e}")
        result = {
            "rollout_id": worker_id,
            "success": False,
            "steps": 0,
            "error": str(e),
        }
        with open(result_file, "w") as f:
            json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description="Multi-worker clean rollouts eval")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_PROMPTS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num_clean", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    prompt = TASK_PROMPTS[args.task]

    # Start batched VLA server
    logger.info(f"Starting batched VLA server on GPU {args.gpu}, port {args.port}")
    server_cmd = [
        str(CONDA_BIN), "run", "-n", "openpi05",
        "python", str(VLA_SERVER_PY),
        "--model_type", "pi05",
        "--config_name", f"pi05_benchmark_{args.task}_merged_inference",
        "--checkpoint", args.checkpoint,
        "--port", str(args.port),
        "--device", "cuda:0",
        "--batched",
        "--max_clients", str(args.num_workers + 4),
        "--max_batch_size", str(args.num_workers),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    server_log = logs_dir / "vla_server.log"
    server_proc = subprocess.Popen(server_cmd, stdout=open(server_log, "w"), stderr=subprocess.STDOUT, env=env)

    # Wait for server
    logger.info("Waiting for VLA server to start...")
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            import socket
            s = socket.socket()
            s.settimeout(2)
            s.connect(("127.0.0.1", args.port))
            s.close()
            logger.info("VLA server ready.")
            break
        except:
            time.sleep(3)
    else:
        logger.error("VLA server failed to start")
        server_proc.kill()
        sys.exit(1)

    # Run workers
    logger.info(f"Running {args.num_clean} clean rollouts with {args.num_workers} workers...")

    workers = []
    for i in range(args.num_clean):
        p = mp.Process(target=worker_rollout, args=(i, args.port, args.task, args.max_steps, prompt, output_dir))
        p.start()
        workers.append(p)

    # Wait for all workers
    for p in workers:
        p.join()

    # Aggregate results
    results = []
    for i in range(args.num_clean):
        f = output_dir / f"rollout_{i}.json"
        if f.exists():
            with open(f) as fp:
                results.append(json.load(fp))

    successes = sum(1 for r in results if r.get("success"))
    total = len(results)
    sr = successes / total if total > 0 else 0.0

    logger.info(f"Clean SR: {sr*100:.1f}% ({successes}/{total})")

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "checkpoint_step": Path(args.checkpoint).name,
        "max_steps": args.max_steps,
        "num_workers": args.num_workers,
        "clean_rollouts": {
            "sr": sr,
            "total": total,
            "successes": successes,
            "per_rollout": sorted(results, key=lambda x: x["rollout_id"]),
        }
    }
    with open(output_dir / f"{args.task}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir / args.task}.json")

    # Stop server
    logger.info("Stopping VLA server...")
    try:
        import socket
        s = socket.socket()
        s.connect(("127.0.0.1", args.port))
        msg = pickle.dumps({"cmd": "shutdown"})
        s.sendall(struct.pack("!I", len(msg)) + msg)
        s.close()
    except:
        server_proc.kill()
    server_proc.wait()


if __name__ == "__main__":
    import pickle
    import struct
    mp.set_start_method("spawn", force=True)
    main()