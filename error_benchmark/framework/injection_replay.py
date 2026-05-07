"""
Injection Replay - Replay error injection process before teleop starts

Purpose:
    1. When collecting recovery demos, show injection animation first (operator understands how error occurred)
    2. Deterministic injection replay (using saved _pre_state + error_spec.seed + rng_state)

Usage:
    replay = InjectionReplay(env_wrapper, task_config, skills_config)
    replay.replay_animation(scene_data, scene_npz_path, clean_traj_dir,
                            render_fn=lambda: env.render(), speed=1.0)
    # Environment is now in post-injection state, ready for teleop
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict

from error_benchmark.framework.error_skills import get_skill

logger = logging.getLogger(__name__)


class InjectionReplay:
    """Replay injection process animation before teleop starts."""

    def __init__(
        self,
        env_wrapper,
        task_config: dict,
        skills_config: Optional[dict] = None,
    ):
        """
        Args:
            env_wrapper: EnvWrapper instance (must have renderer enabled)
            task_config: Task configuration dict
            skills_config: Optional skill-specific config (from benchmark_v5.yaml)
        """
        self.env = env_wrapper
        self.task_config = task_config
        self.skills_config = skills_config or {}

    def replay_animation(
        self,
        scene_data: dict,
        scene_npz_path: str,
        clean_traj_dir: Path,
        render_fn: Optional[Callable] = None,
        speed: float = 1.0,
        context_frames: int = 50,
        pause_after: float = 1.5,
    ) -> bool:
        """
        Replay injection process animation. Environment is in post-injection state after completion.

        Workflow:
        1. Load clean trajectory, replay normal execution frame by frame before injection (context)
        2. Restore _pre_state, re-execute skill.inject() (operator sees error occur)
        3. Pause, then ready for teleop

        Args:
            scene_data: Complete ErrorScene JSON dict
            scene_npz_path: Corresponding NPZ file path (contains sim_state)
            clean_traj_dir: clean trajectories directory
                            (e.g., outputs/v5/clean_trajectories/{task}/)
            render_fn: Render callback (called once per frame, typically env.render())
            speed: Playback speed multiplier (1.0 = real-time 20Hz, 2.0 = double speed)
            context_frames: Maximum context frames (normal execution frames before injection)
            pause_after: Seconds to pause after injection completes

        Returns:
            True if replay succeeded, False otherwise
        """
        # --- Parse scene metadata ---
        clean_traj_ref = scene_data.get('clean_trajectory_ref', '')
        injection_step = scene_data.get('injection_step', 0)
        render_window = scene_data.get('render_window',
                                       scene_data.get('replay', {}).get('render_window', 50))
        error_spec = scene_data.get('error_spec', {})
        error_name = error_spec.get('error_name', '')
        degree = error_spec.get('degree', 'D0')
        seed = error_spec.get('seed', 0)

        if not clean_traj_ref or not error_name:
            logger.warning("Scene data missing clean_trajectory_ref or error_name, "
                           "falling back to direct state load")
            return self._fallback_load(scene_npz_path)

        # --- Load clean trajectory ---
        clean_traj_states = self._load_clean_trajectory(clean_traj_dir, clean_traj_ref)
        if clean_traj_states is None:
            logger.warning(f"Clean trajectory '{clean_traj_ref}' not found, "
                           f"falling back to direct state load")
            return self._fallback_load(scene_npz_path)

        # --- Load scene NPZ (pre-injection state) ---
        try:
            npz_data = np.load(scene_npz_path)
            pre_state = npz_data['sim_state']
        except Exception as e:
            logger.error(f"Failed to load scene NPZ: {e}")
            return False

        # --- Phase 1: Replay context frames (normal execution) ---
        actual_context = min(context_frames, render_window, injection_step)
        context_start = max(0, injection_step - actual_context)

        frame_delay = 1.0 / (20.0 * speed)  # 20Hz control freq

        if actual_context > 0 and context_start < len(clean_traj_states):
            logger.info(f"  [Replay] Playing context: frames {context_start}→{injection_step} "
                        f"({actual_context} frames, speed={speed}x)")
            for i in range(context_start, min(injection_step, len(clean_traj_states))):
                self.env.set_sim_state_flat(clean_traj_states[i])
                self.env.forward()
                if render_fn is not None:
                    render_fn()
                time.sleep(frame_delay)

        # --- Phase 2: Restore pre-injection state and re-execute injection ---
        logger.info(f"  [Replay] Injecting {error_name}_{degree} at frame {injection_step}...")

        # Restore exact pre-injection MuJoCo state
        self.env.set_sim_state_flat(pre_state)
        self.env.forward()

        if render_fn is not None:
            render_fn()
            time.sleep(0.3)  # Brief pause to mark injection start

        # Rebuild RNG (ensure deterministic replay)
        rng = np.random.RandomState(seed)

        # Instantiate error skill
        skill_config = self.skills_config.get(error_name, {})
        skill = get_skill(error_name, skill_config)

        # Build frame_state (consistent with InjectionEngine.execute())
        frame_state = self._extract_state_info()
        frame_state['step'] = injection_step
        frame_state['trajectory_id'] = clean_traj_ref

        # Get target_object from segment info
        target_object = error_spec.get('target_object', '')
        if not target_object:
            target_object = scene_data.get('labels', {}).get('target_object', '')
        frame_state['target_object'] = target_object

        # Provide graspable_objects
        frame_state['graspable_objects'] = self.env._get_graspable_objects()

        # Build injection render callback (with frame delay)
        inject_render = None
        if render_fn is not None:
            def inject_render():
                render_fn()
                time.sleep(frame_delay)

        # Execute injection
        try:
            skill.inject(
                env=self.env,
                degree=degree,
                frame_state=frame_state,
                rng=rng,
                render_fn=inject_render,
            )
        except Exception as e:
            logger.error(f"  [Replay] Injection re-execution failed: {e}")
            # Fall back to loading post-injection state directly
            return self._fallback_load(scene_npz_path)

        # --- Phase 3: Pause to let operator observe error state ---
        if render_fn is not None:
            render_fn()

        if pause_after > 0:
            logger.info(f"  [Replay] Error injected. Pausing {pause_after}s before teleop...")
            time.sleep(pause_after)

        logger.info("  [Replay] Animation complete. Ready for teleop.")
        return True

    def _load_clean_trajectory(self, clean_traj_dir: Path, traj_ref: str) -> Optional[np.ndarray]:
        """Load the states array of a clean trajectory."""
        npz_path = Path(clean_traj_dir) / f"{traj_ref}.npz"
        if not npz_path.exists():
            # Try name without task prefix
            for f in Path(clean_traj_dir).glob("*.npz"):
                if traj_ref in f.stem:
                    npz_path = f
                    break
            else:
                return None

        try:
            data = np.load(npz_path, allow_pickle=True)
            return data['states']
        except Exception as e:
            logger.warning(f"Failed to load clean trajectory {npz_path}: {e}")
            return None

    def _extract_state_info(self) -> dict:
        """Extract current environment state info (consistent with InjectionEngine._extract_state_info)."""
        state_info = {
            'eef_pos': self.env.get_eef_pos(),
            'eef_quat': self.env.get_eef_quat(),
            'gripper_closed_norm': self.env.get_gripper_closed_norm(),
        }

        objects = {}
        for obj_name in self.env.get_all_object_names():
            try:
                pos, quat = self.env.get_object_pose(obj_name)
                linvel, angvel = self.env.get_object_velocity(obj_name)
                objects[obj_name] = {
                    'pos': pos,
                    'quat': quat,
                    'linvel': linvel,
                    'angvel': angvel,
                }
            except (ValueError, KeyError, AttributeError):
                pass
        state_info['objects'] = objects

        return state_info

    def _fallback_load(self, scene_npz_path: str) -> bool:
        """Fallback: directly load sim_state from NPZ (without playing animation)."""
        try:
            npz_data = np.load(scene_npz_path)
            self.env.set_sim_state_flat(npz_data['sim_state'])
            self.env.forward()
            logger.info("  [Replay] Fallback: loaded pre-injection state directly")
            return True
        except Exception as e:
            logger.error(f"  [Replay] Fallback load also failed: {e}")
            return False
