#!/usr/bin/env python
"""
Injection Engine - v5.0

Direct injection engine (no N-1 frame replay):
    Step A: Load pre-computed sim state at frame N directly
    Step B: Inject error at frame N via ErrorSkill.inject()
    Step C: Post-injection observation (validation + rendering)
    Step D: Record ErrorScene

Sim state at each frame is pre-computed during trajectory collection
(CleanTrajectory.states[N]) and loaded via set_sim_state_flat().
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.core import (
        CleanTrajectory, InjectionOpportunity, ErrorSpec, ErrorScene,
        PostRolloutStats, ValidationResult_v5,
    )

from error_benchmark.framework.error_skills import BaseErrorSkill, get_skill
from error_benchmark.framework.core import ErrorScene, PostRolloutStats
from error_benchmark.framework.utils.rollout_utils import collect_rollout_stats

logger = logging.getLogger(__name__)


class InjectionEngine:
    """
    Executes the direct injection + validation pipeline.

    For each injection opportunity:
        1. Load pre-computed sim state at injection frame
        2. Execute the error skill's inject()
        3. Run post-injection observation (for validation)
        4. Return the resulting ErrorScene

    Usage:
        engine = InjectionEngine(env, task_config)
        scene = engine.execute(trajectory, opportunity, skill, rng)
    """

    def __init__(
        self,
        env: 'EnvWrapper',
        task_config: dict,
        render_window: int = 50,
        post_injection_steps: int = 20,
    ):
        """
        Args:
            env: EnvWrapper instance
            task_config: Task configuration dict
            render_window: Kept for backward compatibility (no longer used
                since N-1 frame replay was removed)
            post_injection_steps: Steps to observe after injection (for validation)
        """
        self.env = env
        self.task_config = task_config
        self.render_window = render_window
        self.post_injection_steps = post_injection_steps
        self.last_failure_reason = None  # Populated when execute() returns None
        self.logger = logging.getLogger(__name__)
        self._env_fingerprint = env.get_fingerprint()

    def execute(
        self,
        trajectory: 'CleanTrajectory',
        opportunity: 'InjectionOpportunity',
        skill: BaseErrorSkill,
        rng: np.random.RandomState,
        policy_adapter=None,
        render_fn=None,
    ) -> Optional[ErrorScene]:
        """
        Execute direct injection + validation for one opportunity.

        Loads pre-computed sim state at injection frame, then injects.

        Args:
            trajectory: Clean trajectory (must have pre-computed states)
            opportunity: The injection opportunity (frame, skill, degree)
            skill: The error skill to use
            rng: Random state
            policy_adapter: Optional policy adapter — fed AFTER injection + settling,
                so it observes the post-error state
            render_fn: Optional callback(phase, frame_idx) for video capture.
                Phase is one of: 'pre_inject', 'injection',
                'post_inject', 'post_error'.

        Returns:
            ErrorScene if injection succeeded and validated, None otherwise
        """
        self.last_failure_reason = None  # Reset for each execute() call

        if trajectory.states is None or trajectory.actions is None:
            self.last_failure_reason = "trajectory_missing_states_or_actions"
            self.logger.error("Trajectory has no states/actions")
            return None

        injection_frame = opportunity.frame_index
        degree = opportunity.degree

        self.logger.info(
            f"Executing {skill.name}_{degree} at frame {injection_frame} "
            f"of {trajectory.trajectory_id}"
        )

        # ─── Step A: Load sim state at injection frame directly ───
        self.env.set_sim_state_flat(trajectory.states[injection_frame])

        # Verify environment consistency after state restore
        try:
            from error_benchmark.framework.env_wrapper import EnvWrapper
            current_fp = self.env.get_fingerprint()
            EnvWrapper.verify_fingerprint(current_fp, self._env_fingerprint)
        except Exception as e:
            self.last_failure_reason = f"env_fingerprint_mismatch: {e}"
            self.logger.error(f"Environment fingerprint changed after state restore: {e}")
            return None

        if policy_adapter is not None:
            policy_adapter.reset()

        render_frame_counter = 0

        # ─── Step C: Record pre-injection state and inject ───
        pre_state_info = self._extract_state_info()
        pre_sim_state = self.env.get_sim_state_flat().copy()
        pre_rng_states = self.env.get_rng_states()

        # Add target object info from segment (propagates to validate via pre_state)
        segment = trajectory.get_segment_at_frame(injection_frame)
        if segment is not None:
            pre_state_info['target_object'] = segment.target_object
            pre_state_info['task_phase'] = segment.phase
        else:
            pre_state_info['target_object'] = self.env.get_target_object()
            pre_state_info['task_phase'] = self.env.get_task_phase({'step': injection_frame})

        # Pass graspable objects list for skills that need it (e.g. wrong_object)
        pre_state_info['graspable_objects'] = self.env._get_graspable_objects()

        # Render pre-injection frame
        if render_fn is not None:
            render_frame_counter += 1
            render_fn('pre_inject', render_frame_counter)

        # Build injection render callback that auto-increments the frame counter
        inject_render = None
        if render_fn is not None:
            def inject_render():
                nonlocal render_frame_counter
                render_frame_counter += 1
                render_fn('injection', render_frame_counter)

        # Generate unique token for scene ID deduplication (allows same
        # opportunity to produce multiple distinct scenes with different seeds)
        unique_token = str(rng.randint(0, 2**63))

        # Execute injection
        target_object = pre_state_info.get('target_object')
        try:
            error_spec = skill.inject(
                env=self.env,
                degree=degree,
                frame_state={
                    **pre_state_info,
                    'step': injection_frame,
                    'trajectory_id': trajectory.trajectory_id,
                },
                rng=rng,
                render_fn=inject_render,
            )
        except Exception as e:
            self.last_failure_reason = f"injection_exception: {type(e).__name__}: {e}"
            self.logger.warning(f"Injection failed: {e}")
            return None

        # Propagate target_object to error_spec
        if not error_spec.target_object:
            error_spec.target_object = target_object or ''

        # ─── Step D: Post-injection observation ───
        post_state_info = self._extract_state_info()

        # Render post-injection frame
        if render_fn is not None:
            render_frame_counter += 1
            render_fn('post_inject', render_frame_counter)

        # ─── Step E: Collect post-injection stats (for validation) ───
        # Prefer error_spec.target (most accurate: reflects what the skill actually
        # targeted), then segment's target_object, then fallback to first object.
        spec_target = error_spec.target.get("body") if error_spec.target else None
        obj_name = spec_target or target_object
        all_objects = self.env.get_all_object_names()
        if obj_name is None or obj_name not in all_objects:
            obj_name = all_objects[0] if all_objects else None

        # Build step_callback for post-error rendering
        post_error_cb = None
        if render_fn is not None:
            _counter = render_frame_counter

            def post_error_cb(t):
                nonlocal _counter
                _counter += 1
                render_fn('post_error', _counter)

        post_stats = None
        if obj_name:
            post_stats = collect_rollout_stats(
                env=self.env,
                obj_name=obj_name,
                steps=self.post_injection_steps,
                pre_state_info=pre_state_info,
                step_callback=post_error_cb,
            )

        # ─── Step F: Feed policy adapter (post-error, env stabilized) ───
        # Policy should observe the post-error state, not the clean replay.
        if policy_adapter is not None:
            try:
                obs = self.env._get_obs()
                policy_adapter.predict(obs)
            except Exception:
                pass

        # ─── Step G: Validate ───
        validation_result = skill.validate(
            env=self.env,
            pre_state=pre_state_info,
            post_stats=post_stats or PostRolloutStats(),
            degree=degree,
        )

        if not validation_result.ok:
            self.last_failure_reason = f"validation: {validation_result.reason}"
            self.logger.info(
                f"Validation failed for {skill.name}_{degree}: "
                f"{validation_result.reason}"
            )
            return None

        # ─── Step H: Capture post-injection stable state ───
        post_sim_state = self.env.get_sim_state_flat().copy()
        post_injection_info = self._build_post_injection_info(obj_name)

        # ─── Build ErrorScene ───
        scene = self._build_scene(
            trajectory=trajectory,
            opportunity=opportunity,
            error_spec=error_spec,
            validation_result=validation_result,
            pre_sim_state=pre_sim_state,
            post_sim_state=post_sim_state,
            post_injection_info=post_injection_info,
            pre_rng_states=pre_rng_states,
            unique_suffix=unique_token,
        )

        self.logger.info(
            f"Successfully created scene {scene.scene_id}: "
            f"{skill.name}_{degree} at frame {injection_frame}"
        )

        return scene

    def execute_batch(
        self,
        trajectory: 'CleanTrajectory',
        opportunities: List['InjectionOpportunity'],
        skills_map: Dict[str, BaseErrorSkill],
        rng: np.random.RandomState,
        max_scenes: Optional[int] = None,
    ) -> List[ErrorScene]:
        """
        Execute multiple injections on the same trajectory.

        Args:
            trajectory: Clean trajectory
            opportunities: List of opportunities to try
            skills_map: Dict mapping skill name -> skill instance
            rng: Random state
            max_scenes: Maximum number of scenes to generate

        Returns:
            List of successfully generated ErrorScene objects
        """
        scenes = []

        for opp in opportunities:
            if max_scenes and len(scenes) >= max_scenes:
                break

            skill = skills_map.get(opp.error_name)
            if skill is None:
                continue

            scene = self.execute(trajectory, opp, skill, rng)
            if scene is not None:
                scenes.append(scene)

        return scenes

    def _extract_state_info(self) -> Dict:
        """Extract current state info from environment."""
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

    def _build_post_injection_info(self, target_object: str) -> Dict:
        """Build post-injection metadata: object poses, gripper state, id_eligible flag.

        Called after injection + validation (environment is in stable post-injection state).
        """
        obj_poses = {}
        for obj_name in self.env.get_all_object_names():
            try:
                pos, quat = self.env.get_object_pose(obj_name)
                obj_poses[obj_name] = {
                    'pos': pos.tolist() if hasattr(pos, 'tolist') else list(pos),
                    'quat': quat.tolist() if hasattr(quat, 'tolist') else list(quat),
                }
            except (ValueError, KeyError, AttributeError):
                pass

        gripper_norm = self.env.get_gripper_closed_norm()
        eef_pos = self.env.get_eef_pos()

        # Compute id_eligible: post-injection state equivalent to a new Initial Distribution?
        # Conditions: target object on table, stable, within workspace, gripper empty
        id_eligible = False
        if target_object and target_object in obj_poses:
            obj_pos = obj_poses[target_object]['pos']
            table_height = 0.82  # robosuite default table height
            on_table = abs(obj_pos[2] - table_height) < 0.05
            in_bounds = abs(obj_pos[0]) < 0.25 and abs(obj_pos[1]) < 0.25
            gripper_empty = gripper_norm < 0.05
            id_eligible = on_table and in_bounds and gripper_empty

        return {
            'obj_poses': obj_poses,
            'gripper_closed_norm': float(gripper_norm),
            'eef_pos': eef_pos.tolist() if hasattr(eef_pos, 'tolist') else list(eef_pos),
            'id_eligible': id_eligible,
            'target_object': target_object or '',
        }

    def _build_scene(
        self,
        trajectory: 'CleanTrajectory',
        opportunity: 'InjectionOpportunity',
        error_spec: 'ErrorSpec',
        validation_result: 'ValidationResult_v5',
        pre_sim_state: np.ndarray,
        post_sim_state: np.ndarray,
        post_injection_info: Dict,
        pre_rng_states: dict,
        unique_suffix: str = "",
    ) -> ErrorScene:
        """Build an ErrorScene from injection results."""
        import hashlib
        from datetime import datetime

        # Generate scene ID (unique_suffix ensures distinct IDs when the same
        # opportunity is reused with different random seeds)
        hash_input = (
            f"{trajectory.trajectory_id}_{opportunity.frame_index}_"
            f"{opportunity.error_name}_{opportunity.degree}_{unique_suffix}"
        )
        scene_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:12]
        scene_id = f"v5_{opportunity.error_name}_{opportunity.degree}_{scene_hash}"

        scene = ErrorScene(
            version="v5.0",
            scene_id=scene_id,
            dataset={
                'source': trajectory.source,
                'trajectory_id': trajectory.trajectory_id,
                'demo_key': trajectory.demo_key,
                'dataset_path': trajectory.dataset_path,
            },
            env_fingerprint=self.env.get_fingerprint(),
            replay={
                'init_state_source': 'trajectory',
                'injection_frame': opportunity.frame_index,
                'render_window': min(opportunity.frame_index, self.render_window),
            },
            error_spec=error_spec,
            labels={
                'error_name': opportunity.error_name,
                'degree': opportunity.degree,
                'subtype_id': opportunity.subtype_id(),
                'task_phase': opportunity.task_phase,
                'target_object': opportunity.metadata.get('target_object', '') if opportunity.metadata else '',
                'phase_group': opportunity.metadata.get('phase_group', '') if opportunity.metadata else '',
                'validation_metrics': validation_result.metrics,
            },
            detected_by={
                'method': 'offline_scan',
                'skill': opportunity.error_name,
            },
            validated_by={
                'skill': opportunity.error_name,
                'reason': validation_result.reason,
            },
            rng_state=pre_rng_states,
            clean_trajectory_ref=trajectory.trajectory_id,
            injection_step=opportunity.frame_index,
            render_window=min(opportunity.frame_index, self.render_window),
            post_injection=post_injection_info,
        )

        scene._pre_state = pre_sim_state
        scene._post_state = post_sim_state

        return scene
