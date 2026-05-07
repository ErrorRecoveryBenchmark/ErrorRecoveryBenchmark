#!/usr/bin/env python3
"""
Generate success videos for all MimicGen source datasets.
Replays demos via state-based playback in robosuite with offscreen rendering.

Usage:
    MUJOCO_GL=egl python3 error_benchmark/scripts/utils/generate_success_videos.py \
        --output_dir shared/mimicgen_workspace/mimicgen/datasets/source_videos \
        --height 512 --width 512 --video_skip 3
"""
import os
import sys
import json
import argparse
import glob
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
import imageio

# Add mimicgen/robosuite to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

import robosuite
import mimicgen  # register mimicgen envs (PickPlace_D0, Coffee_D0, etc.)


def create_env_from_hdf5(hdf5_path, camera_name="agentview", height=512, width=512):
    """Create robosuite env from HDF5 metadata with offscreen rendering."""
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["render_gpu_device_id"] = 0
    env_kwargs["camera_names"] = [camera_name]
    env_kwargs["camera_heights"] = height
    env_kwargs["camera_widths"] = width
    env_kwargs["use_camera_obs"] = True

    env_name = env_args["env_name"]
    print(f"  Creating env: {env_name}")
    env = robosuite.make(env_name, **env_kwargs)
    return env


def get_demo_rewards(hdf5_path):
    """Return dict of demo_name -> total_reward."""
    rewards = {}
    with h5py.File(hdf5_path, "r") as f:
        for demo_name in sorted(f["data"].keys()):
            if demo_name.startswith("demo"):
                r = f["data"][demo_name]["rewards"][:].sum()
                rewards[demo_name] = float(r)
    return rewards


def render_demo_video(env, hdf5_path, demo_name, output_path,
                      camera_name="agentview", video_skip=3):
    """Render one demo to MP4 via state-based replay."""
    with h5py.File(hdf5_path, "r") as f:
        states = f["data"][demo_name]["states"][:]

    frames = []
    env.reset()
    # Set initial state
    mujoco_env = env.sim
    # Use state from the first timestep
    mj_state = states[0]
    env.sim.set_state_from_flattened(mj_state)
    env.sim.forward()

    for t in range(len(states)):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        # Render offscreen image
        img = env.sim.render(
            height=env.camera_heights[0],
            width=env.camera_widths[0],
            camera_name=camera_name,
        )
        frames.append(img[::-1, :, :])  # Flip for imageio

    if len(frames) == 0:
        print(f"  WARNING: no frames for {demo_name}")
        return

    # Subsample frames
    frames = frames[::video_skip]
    imageio.mimwrite(output_path, frames, fps=int(20 / video_skip), quality=8)
    print(f"  Saved: {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="shared/mimicgen_workspace/mimicgen/datasets/source")
    parser.add_argument("--output_dir", default="shared/mimicgen_workspace/mimicgen/datasets/source_videos")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_skip", type=int, default=3,
                        help="Save every Nth frame to reduce video size")
    parser.add_argument("--camera", default="agentview")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(source_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} source datasets")

    for hdf5_path in hdf5_files:
        task_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {task_name}")

        # Get rewards to pick best demo
        rewards = get_demo_rewards(hdf5_path)
        if not rewards:
            print(f"  No demos found, skipping")
            continue

        # Pick the demo with highest reward
        best_demo = max(rewards, key=rewards.get)
        print(f"  Best demo: {best_demo} (reward={rewards[best_demo]:.1f})")

        # Also create a combined video of all demos
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        try:
            env = create_env_from_hdf5(
                hdf5_path,
                camera_name=args.camera,
                height=args.height,
                width=args.width,
            )

            # Render all demos
            for demo_name in sorted(rewards.keys()):
                idx = demo_name.replace("demo_", "")
                out_path = os.path.join(task_output_dir, f"{demo_name}_r{rewards[demo_name]:.0f}.mp4")
                try:
                    render_demo_video(
                        env, hdf5_path, demo_name, out_path,
                        camera_name=args.camera,
                        video_skip=args.video_skip,
                    )
                except Exception as e:
                    print(f"  ERROR rendering {demo_name}: {e}")

            env.close()

        except Exception as e:
            print(f"  ERROR creating env for {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
