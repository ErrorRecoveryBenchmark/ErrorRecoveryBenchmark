#!/usr/bin/env python3
"""
Evaluate a VLA policy (served via OpenPI) on 9 MimicGen tasks.
Based on Phoenix CVPR 2025 evaluation protocol.

Usage:
    # Start policy server first (openpi05 env):
    conda run -n openpi05 python scripts/start_vla_server.py \
        --model_type pi0 --config_name pi05_base_mimicgen_phoenix \
        --checkpoint /path/to/checkpoint --port 8000

    # Run evaluation (mimicgen_env):
    MUJOCO_GL=egl python evaluate_mimicgen.py \
        --host 0.0.0.0 --port 8000 \
        --tasks all --num-trials-per-task 50

    # Evaluate a single task:
    MUJOCO_GL=egl python evaluate_mimicgen.py \
        --host 0.0.0.0 --port 8000 \
        --tasks Coffee_D0 --num-trials-per-task 50
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import robosuite as suite
from robosuite.controllers import load_controller_config
import tqdm
import tyro

# Register MimicGen environments
import mimicgen  # noqa: F401

# ── Task configuration ───────────────────────────────────────────────────────
TASK_CONFIGS = {
    "PickPlace_D0":           {"prompt": "pick up the milk, cereal, bread, and can and place them in the correct bins", "max_steps": 500},
    "Coffee_D0":              {"prompt": "make coffee",                                    "max_steps": 400},
    "Coffee_D1":              {"prompt": "make coffee",                                    "max_steps": 400},
    "Stack_D0":               {"prompt": "stack the red block on top of the green block",  "max_steps": 400},
    "Stack_D1":               {"prompt": "stack the red block on top of the green block",  "max_steps": 400},
    "StackThree_D0":          {"prompt": "stack three blocks",                             "max_steps": 400},
    "StackThree_D1":          {"prompt": "stack three blocks",                             "max_steps": 400},
    "Threading_D0":           {"prompt": "insert the needle into the needle hole",         "max_steps": 400},
    "ThreePieceAssembly_D0":  {"prompt": "assemble the three pieces",                      "max_steps": 500},
    "ThreePieceAssembly_D1":  {"prompt": "assemble the three pieces",                      "max_steps": 500},
}

DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Evaluation
    tasks: str = "all"  # Comma-separated task names or "all"
    num_trials_per_task: int = 50
    num_steps_wait: int = 10  # Steps to wait for objects to settle
    max_steps_override: Optional[int] = None  # Override per-task max_steps

    # Output
    video_out_path: str = "data/mimicgen_eval/videos"
    results_out_path: str = "data/mimicgen_eval/results.json"
    save_videos: bool = True

    # Reproducibility
    seed: int = 7


def _make_env(task_name: str):
    """Create a robosuite environment for a MimicGen task."""
    env = suite.make(
        env_name=task_name,
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=224,
        camera_widths=224,
        reward_shaping=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    return env


def _quat2axisangle(quat):
    """Convert quaternion (x, y, z, w) to axis-angle representation."""
    quat = np.array(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def eval_mimicgen(args: Args) -> None:
    np.random.seed(args.seed)

    # Parse task list
    if args.tasks.lower() == "all":
        task_list = list(TASK_CONFIGS.keys())
    else:
        task_list = [t.strip() for t in args.tasks.split(",")]
        for t in task_list:
            if t not in TASK_CONFIGS:
                raise ValueError(f"Unknown task: {t}. Available: {list(TASK_CONFIGS.keys())}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    all_results = {}
    total_episodes = 0
    total_successes = 0

    for task_name in task_list:
        config = TASK_CONFIGS[task_name]
        prompt = config["prompt"]
        max_steps = args.max_steps_override or config["max_steps"]

        logging.info(f"\n{'='*60}")
        logging.info(f"Task: {task_name} | Prompt: '{prompt}' | Max steps: {max_steps}")
        logging.info(f"{'='*60}")

        env = _make_env(task_name)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_name):
            obs = env.reset()
            action_plan = collections.deque()
            replay_images = []
            reward = False
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to settle
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess images (vertical flip to match training)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        # Build observation dict for policy server
                        state = np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))
                        assert state.shape == (8,), f"Expected 8D state, got {state.shape}"

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": prompt,
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Policy predicts {len(action_chunk)} steps, need >= {args.replan_steps}"
                        )
                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done or reward:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception in {task_name} ep {episode_idx}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            if args.save_videos and replay_images:
                suffix = "success" if (done or reward) else "failure"
                video_path = (
                    pathlib.Path(args.video_out_path)
                    / f"{task_name}_{suffix}_ep{episode_idx}.mp4"
                )
                imageio.mimwrite(str(video_path), replay_images, fps=20)

        # Log per-task results
        sr = task_successes / max(task_episodes, 1)
        all_results[task_name] = {
            "success_rate": sr,
            "successes": task_successes,
            "episodes": task_episodes,
        }
        logging.info(f"  {task_name}: SR = {sr:.1%} ({task_successes}/{task_episodes})")

        env.close()

    # Summary
    overall_sr = total_successes / max(total_episodes, 1)
    all_results["_overall"] = {
        "success_rate": overall_sr,
        "successes": total_successes,
        "episodes": total_episodes,
    }

    logging.info(f"\n{'='*60}")
    logging.info(f"OVERALL: SR = {overall_sr:.1%} ({total_successes}/{total_episodes})")
    logging.info(f"{'='*60}")
    for name, res in all_results.items():
        if name != "_overall":
            logging.info(f"  {name:30s} {res['success_rate']:.1%}  ({res['successes']}/{res['episodes']})")

    # Save results JSON
    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_mimicgen)
