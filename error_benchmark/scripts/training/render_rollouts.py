#!/usr/bin/env python
"""Render clean rollouts for debugging."""

import argparse
import json
import logging
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


def run_and_render_rollout(port: int, task: str, max_steps: int, prompt: str,
                            output_dir: Path, rollout_id: int, render: bool = True):
    """Run a single rollout and render video."""
    import yaml
    sys.path.insert(0, str(PROJECT_DIR))
    sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"))
    sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"))

    from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.policy_adapter import PolicyServerAdapter
    from error_benchmark.framework.video_recorder import VideoRecorder

    task_reg = load_task_registry(task)
    task_config_path = str(PROJECT_DIR) + "/" + task_reg["task_config"]
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)

    # Create environment with rendering
    env = create_env(task_config, task_reg["dataset_path"],
                     enable_camera=True, camera_resolution=256,
                     has_renderer=render)

    env_wrapper = EnvWrapper(env, task_config)

    policy = PolicyServerAdapter(
        host="localhost",
        port=port,
        task_prompt=prompt,
        replan_interval=5,
        connection_timeout=30.0,
    )

    # Setup video recorder
    video_path = output_dir / f"rollout_{rollout_id}.mp4"
    recorder = VideoRecorder(video_path) if render else None

    obs = env.reset()
    if recorder:
        recorder.capture_frame(env_wrapper)

    policy.start_episode()
    success = False
    steps_taken = 0
    actions = []
    observations = []

    for step in range(max_steps):
        result = policy.predict_from_obs(obs)
        actions.append(result.action.copy())
        # Simple observation recording
        eef_pos_key = "eef_pos" if "eef_pos" in obs else "robot0_eef_pos"
        grip_key = "gripper_qpos_raw" if "gripper_qpos_raw" in obs else "robot0_gripper_qpos"
        observations.append({
            "eef_pos": np.copy(obs[eef_pos_key]) if eef_pos_key in obs else None,
            "gripper_qpos": np.copy(obs[grip_key]) if grip_key in obs else None,
        })

        obs, reward, done, info = env_wrapper.step(result.action)
        steps_taken = step + 1

        if recorder:
            recorder.capture_frame(env_wrapper)

        if env_wrapper.check_success():
            success = True
            break

    if recorder:
        recorder.close()

    policy.close()

    # Save trajectory data
    traj_path = output_dir / f"rollout_{rollout_id}.npz"
    np.savez(traj_path,
             actions=np.array(actions),
             observations=observations,
             success=success,
             steps=steps_taken)

    result = {
        "rollout_id": rollout_id,
        "success": success,
        "steps": steps_taken,
    }
    result_file = output_dir / f"rollout_{rollout_id}_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f)

    status = "OK" if success else "FAIL"
    logger.info(f"  Rollout {rollout_id}: {status} in {steps_taken} steps, video: {video_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Render clean rollouts for debugging")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_PROMPTS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    prompt = TASK_PROMPTS[args.task]

    # Start VLA server
    logger.info(f"Starting VLA server on GPU {args.gpu}, port {args.port}")
    server_cmd = [
        str(CONDA_BIN), "run", "-n", "openpi05",
        "python", str(VLA_SERVER_PY),
        "--model_type", "pi05",
        "--config_name", f"pi05_benchmark_{args.task}_merged_inference",
        "--checkpoint", args.checkpoint,
        "--port", str(args.port),
        "--device", "cuda:0",
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

    # Run rollouts with rendering
    logger.info(f"Running {args.num_rollouts} rollouts with rendering...")

    render = not args.no_render
    results = []
    for i in range(args.num_rollouts):
        r = run_and_render_rollout(args.port, args.task, args.max_steps, prompt,
                                    output_dir, i, render=render)
        results.append(r)

    # Summary
    successes = sum(1 for r in results if r.get("success"))
    total = len(results)
    sr = successes / total if total > 0 else 0.0
    logger.info(f"SR: {sr*100:.1f}% ({successes}/{total})")
    logger.info(f"Videos saved to {output_dir}")

    # Stop server
    logger.info("Stopping VLA server...")
    try:
        import socket
        import pickle
        import struct
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
    main()