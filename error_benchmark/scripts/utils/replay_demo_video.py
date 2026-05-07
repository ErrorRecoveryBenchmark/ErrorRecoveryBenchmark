#!/usr/bin/env python
"""
Replay a human recovery demo (or augmented demo) and record MP4 video.

Supports two replay modes:
  - state: Load each sim state from NPZ sequentially (exact visual replay)
  - action: Load initial state, replay actions through physics (determinism check)

Usage:
    # State replay (default, exact)
    python error_benchmark/scripts/utils/replay_demo_video.py \
        --task stack --npz path/to/demo.npz

    # Action replay
    python error_benchmark/scripts/utils/replay_demo_video.py \
        --task stack --npz path/to/demo.npz --mode action

    # Custom output path and resolution
    python error_benchmark/scripts/utils/replay_demo_video.py \
        --task stack --npz path/to/demo.npz --output my_video.mp4 --resolution 512
"""

import argparse
import logging
import sys
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.framework.video_recorder import VideoRecorder
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry


def replay_states(env_wrapper, states, recorder, demo_id=""):
    """Replay by loading each sim state sequentially."""
    logger = logging.getLogger(__name__)
    total = len(states)
    logger.info(f"State replay: {total} frames")

    for i, state in enumerate(states):
        env_wrapper.set_sim_state_flat(np.asarray(state))
        env_wrapper.forward()

        overlay = [
            f"demo: {demo_id}" if demo_id else "state replay",
            f"frame: {i}/{total}",
        ]

        if i == total - 1:
            success = env_wrapper.check_success()
            overlay.append(f"final: {'SUCCESS' if success else 'FAIL'}")

        recorder.capture_frame(env_wrapper, overlay_text=overlay)

    logger.info(f"State replay complete: {total} frames recorded")


def replay_actions(env_wrapper, initial_state, actions, recorder, demo_id=""):
    """Replay by loading initial state and stepping actions through physics."""
    logger = logging.getLogger(__name__)
    total = len(actions)
    logger.info(f"Action replay: {total} steps")

    env_wrapper.set_sim_state_flat(np.asarray(initial_state))
    env_wrapper.forward()

    recorder.capture_frame(env_wrapper, overlay_text=[
        f"demo: {demo_id}" if demo_id else "action replay",
        "step: 0 (init)",
    ])

    for i, action in enumerate(actions):
        env_wrapper.step(action)

        overlay = [
            f"demo: {demo_id}" if demo_id else "action replay",
            f"step: {i + 1}/{total}",
        ]

        if i == total - 1:
            success = env_wrapper.check_success()
            overlay.append(f"final: {'SUCCESS' if success else 'FAIL'}")

        recorder.capture_frame(env_wrapper, overlay_text=overlay)

    logger.info(f"Action replay complete: {total} steps recorded")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recovery demo NPZ and record MP4 video")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., stack, pick_place)")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to demo NPZ file")
    parser.add_argument("--mode", type=str, default="state",
                        choices=["state", "action"],
                        help="Replay mode: 'state' (load each state) or "
                             "'action' (replay actions through physics)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: alongside NPZ)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Video resolution (square)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video FPS")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera name for rendering")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        logger.error(f"NPZ file not found: {npz_path}")
        sys.exit(1)

    # Load NPZ
    data = np.load(str(npz_path), allow_pickle=True)
    states = data.get('states')
    actions = data.get('actions')
    demo_id = str(data.get('demo_id', npz_path.stem))

    if args.mode == "state" and (states is None or len(states) == 0):
        logger.error("NPZ has no 'states' array. Use --mode action instead.")
        sys.exit(1)
    if args.mode == "action" and (actions is None or len(actions) == 0):
        logger.error("NPZ has no 'actions' array.")
        sys.exit(1)
    if args.mode == "action" and (states is None or len(states) == 0):
        logger.error("NPZ has no 'states' array for initial state.")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        demo_replay_dir = PROJECT_ROOT / "all_videos" / "demo_replay"
        demo_replay_dir.mkdir(parents=True, exist_ok=True)
        output_path = demo_replay_dir / f"{npz_path.stem}.{args.mode}_replay.mp4"

    # Load task config
    task_info = load_task_registry(args.task)
    dataset_path = task_info['dataset_path']
    task_config_path = task_info.get('task_config', '')
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)

    # Create environment with offscreen renderer
    env = create_env(task_config, dataset_path, enable_camera=True)
    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    logger.info(f"Task: {args.task}")
    logger.info(f"NPZ: {npz_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {output_path}")

    recorder = VideoRecorder(
        output_path,
        fps=args.fps,
        resolution=(args.resolution, args.resolution),
        camera_name=args.camera,
    )

    if args.mode == "state":
        replay_states(env_wrapper, states, recorder, demo_id=demo_id)
    else:
        replay_actions(env_wrapper, states[0], actions, recorder, demo_id=demo_id)

    recorder.close()
    logger.info(f"Video saved: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
