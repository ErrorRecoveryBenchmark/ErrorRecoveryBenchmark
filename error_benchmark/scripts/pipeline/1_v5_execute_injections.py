#!/usr/bin/env python
"""
v5.0 Pipeline Step 1: Execute Injections

Executes the scheduled injections using context-replay + error skills.

Usage:
    python error_benchmark/scripts/pipeline/1_v5_execute_injections.py \
        --config error_benchmark/configs/benchmark_v5.yaml \
        --task pick_place \
        --schedule error_benchmark/outputs/v5/schedules/pick_place/schedule.jsonl \
        --trajectories_dir error_benchmark/outputs/v5/clean_trajectories/pick_place

Output:
    {output_dir}/v5/{task}/
        scenes/
            *.json (scene metadata)
            *.npz (sim state)
        meta.json
"""

import argparse
import logging
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
from error_benchmark.framework.quota_scheduler import QuotaScheduler
from error_benchmark.framework.context_replay import InjectionEngine
from error_benchmark.framework.error_skills import get_skill
from error_benchmark.framework.core import QuotaStatus
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry, validate_trajectories_for_task


def render_frame(env, camera_name="agentview", height=480, width=480):
    """Render a frame from the environment's sim."""
    try:
        frame = env.sim.render(height=height, width=width, camera_name=camera_name)
        if frame is not None:
            return frame[::-1]  # Flip vertical for correct orientation
    except Exception:
        pass
    # Fallback: try through env_wrapper._env
    try:
        raw_env = env._env if hasattr(env, '_env') else env
        frame = raw_env.sim.render(height=height, width=width, camera_name=camera_name)
        if frame is not None:
            return frame[::-1]
    except Exception:
        pass
    return None


def annotate_frame(frame, phase, info_text, frame_idx, total_frames):
    """Annotate a single frame with phase banner and info box."""
    import cv2

    h, w = frame.shape[:2]
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Phase banner (top bar)
    banner_h = 36
    phase_map = {
        'context': ((34, 139, 34), "CONTEXT REPLAY"),
        'pre_inject': ((34, 139, 34), "PRE-INJECTION"),
        'injection': ((200, 30, 30), "ERROR INJECTED!"),
        'post_inject': ((200, 30, 30), "ERROR INJECTED!"),
        'post_error': ((220, 140, 20), "POST-ERROR SETTLEMENT"),
    }
    banner_color, banner_text = phase_map.get(phase, ((100, 100, 100), phase.upper()))

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    (tw, th), _ = cv2.getTextSize(banner_text, font, 0.8, 2)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2
    cv2.putText(out, banner_text, (tx, ty), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Info box (bottom-left)
    info_lines = info_text if isinstance(info_text, list) else [info_text]
    line_h = 22
    box_padding = 8
    max_tw = 0
    for line in info_lines:
        (ltw, _), _ = cv2.getTextSize(line, font, 0.45, 1)
        max_tw = max(max_tw, ltw)
    box_w = max_tw + box_padding * 2
    box_h = len(info_lines) * line_h + box_padding * 2
    box_x, box_y = 10, h - box_h - 10

    overlay = out.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for i, line in enumerate(info_lines):
        ly = box_y + box_padding + (i + 1) * line_h - 4
        cv2.putText(out, line, (box_x + box_padding, ly), font, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Frame counter (bottom-right)
    counter_text = f"Frame {frame_idx}/{total_frames}"
    (ctw, cth), _ = cv2.getTextSize(counter_text, font, 0.5, 1)
    cx, cy = w - ctw - 16, h - 16
    overlay = out.copy()
    cv2.rectangle(overlay, (cx - 6, cy - cth - 6), (cx + ctw + 6, cy + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, counter_text, (cx, cy), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="v5 Step 1: Execute injections")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/benchmark_v5.yaml")
    parser.add_argument("--task", type=str, default="pick_place")
    parser.add_argument("--schedule", type=str, default=None)
    parser.add_argument("--trajectories_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Maximum total scenes to generate")
    parser.add_argument("--render", action="store_true",
                        help="Render each scene as annotated MP4 video")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera name for rendering (default: agentview)")
    parser.add_argument("--resolution", type=int, default=480,
                        help="Render resolution (default: 480)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video FPS (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
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
    schedule_path = args.schedule or (
        f"error_benchmark/outputs/v5/schedules/{args.task}/schedule.jsonl"
    )
    output_dir = args.output_dir or config['paths'].get(
        'output_dir',
        f"error_benchmark/outputs/v5/{args.task}"
    )

    full_output = PROJECT_ROOT / output_dir
    scenes_dir = full_output / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== v5 Step 1: Execute Injections ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output: {full_output}")

    # Load trajectories
    trajectories = CleanTrajectoryCollector.load(str(PROJECT_ROOT / traj_dir))
    validate_trajectories_for_task(trajectories, args.task)
    traj_map = {t.trajectory_id: t for t in trajectories}
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Load schedule
    schedule = QuotaScheduler.load_schedule(str(PROJECT_ROOT / schedule_path))
    if args.max_scenes:
        schedule = schedule[:args.max_scenes]
    logger.info(f"Schedule: {len(schedule)} injections")

    # Load task config and create environment
    task_config_path = task_info.get('task_config', config['paths']['task_config'])
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)
    env = create_env(task_config, dataset_path, enable_camera=args.render)

    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    # Setup rendering
    if args.render:
        import robosuite.macros as macros
        macros.IMAGE_CONVENTION = "opencv"
        import imageio.v2 as imageio

        videos_dir = full_output / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering enabled: camera={args.camera}, "
                     f"resolution={args.resolution}, fps={args.fps}")

    # Create context replay engine
    injection_config = config.get('injection', {})
    engine = InjectionEngine(
        env=env_wrapper,
        task_config=task_config,
        render_window=injection_config.get('context_window', 50),
        post_injection_steps=injection_config.get('post_injection_steps', 20),
    )

    # Create skills
    skills_config = config.get('error_skills', {})
    rng = np.random.RandomState(args.seed)

    # Execute injections
    successful_scenes = []
    failed = 0

    pbar = tqdm(schedule, desc=f"Injecting ({args.task})")
    for i, opp in enumerate(pbar):
        trajectory = traj_map.get(opp.trajectory_id)
        if trajectory is None:
            logger.warning(f"Trajectory {opp.trajectory_id} not found, skipping")
            failed += 1
            continue

        # Get or create skill
        try:
            skill_params = skills_config.get(opp.error_name, {})
            skill = get_skill(opp.error_name, skill_params)
        except KeyError:
            logger.warning(f"Unknown skill {opp.error_name}, skipping")
            failed += 1
            continue

        # Build render_fn if rendering enabled
        captured_frames = []
        render_fn = None
        if args.render:
            injection_frame = opp.frame_index
            context_window = injection_config.get('context_window', 50)
            post_steps = injection_config.get('post_injection_steps', 20)
            total_frames_est = min(injection_frame, context_window) + 1 + post_steps

            info_lines = [
                f"Skill: {opp.error_name}",
                f"Degree: {opp.degree}",
                f"Phase: {opp.task_phase}",
                f"Frame: {injection_frame}",
            ]

            def make_render_fn(cap_frames, info, total_est, raw_env):
                def _render_fn(phase, frame_idx):
                    frame = render_frame(raw_env, args.camera,
                                         args.resolution, args.resolution)
                    if frame is not None:
                        annotated = annotate_frame(
                            frame, phase, info, frame_idx, total_est)
                        cap_frames.append(annotated)
                return _render_fn

            raw_env = env_wrapper._env
            render_fn = make_render_fn(captured_frames, info_lines,
                                       total_frames_est, raw_env)

        # Execute with optional render callback
        scene = engine.execute(trajectory, opp, skill, rng, render_fn=render_fn)

        if scene is not None:
            # Save scene
            scene_json_path = scenes_dir / f"{scene.scene_id}.json"
            with open(scene_json_path, 'w') as f:
                json.dump(scene.to_dict(), f, indent=2, cls=NumpyEncoder)

            if scene._pre_state is not None:
                npz_path = scenes_dir / f"{scene.scene_id}.npz"
                save_dict = {'sim_state': scene._pre_state}
                if scene._post_state is not None:
                    save_dict['post_sim_state'] = scene._post_state
                np.savez_compressed(npz_path, **save_dict)
                scene.replay['pre_state_npz'] = str(npz_path)

            # Save video if frames were captured
            if args.render and captured_frames:
                video_path = videos_dir / f"{scene.scene_id}.mp4"
                imageio.mimsave(str(video_path), captured_frames, fps=args.fps)
                logger.debug(f"Saved video: {video_path.name} "
                             f"({len(captured_frames)} frames)")

            successful_scenes.append(scene)
            pbar.set_postfix(ok=len(successful_scenes), fail=failed)
        else:
            failed += 1
            pbar.set_postfix(ok=len(successful_scenes), fail=failed)

    # Save meta
    meta = {
        'version': 'v5.0',
        'created': datetime.now().isoformat(),
        'task': args.task,
        'total_scenes': len(successful_scenes),
        'total_attempts': len(schedule),
        'failed': failed,
        'success_rate': len(successful_scenes) / len(schedule) if schedule else 0,
        'scenes': [
            {
                'scene_id': s.scene_id,
                'error_name': s.labels.get('error_name', ''),
                'degree': s.labels.get('degree', ''),
                'task_phase': s.labels.get('task_phase', ''),
            }
            for s in successful_scenes
        ],
    }

    meta_path = full_output / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Summary
    by_subtype = {}
    for s in successful_scenes:
        key = s.labels.get('subtype_id', 'unknown')
        by_subtype[key] = by_subtype.get(key, 0) + 1

    logger.info(f"\n=== Injection Summary ===")
    logger.info(f"Total scenes: {len(successful_scenes)} / {len(schedule)} attempts")
    logger.info(f"Success rate: {meta['success_rate']:.1%}")
    logger.info(f"Subtypes generated: {len(by_subtype)}")
    for key, count in sorted(by_subtype.items(), key=lambda x: -x[1]):
        logger.info(f"  {key}: {count}")


if __name__ == "__main__":
    main()
