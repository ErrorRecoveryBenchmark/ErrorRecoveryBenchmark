#!/usr/bin/env python
"""
Stage 3: MimicGen/IntervenGen-style Recovery Data Augmentation

Takes segmented recovery demos and generates augmented demos by:
  3A: Scene configuration augmentation (different error scenes as starting states)
  3B: Cross-degree augmentation (D0 → D1 via perturbation)
  3C: Cross-subtype augmentation (within same RBG)

Usage:
    # Augment all collected demos for a task
    python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
        --task stack \
        --target_per_subtype 100

    # Augment specific subtype
    python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
        --task stack \
        --subtype collision_holding_D0 \
        --target 100

Output:
    {augmented_demos_dir}/{task}/{subtype_id}/
        aug_*.npz   (augmented trajectory arrays)
        manifest.json
"""

import argparse
import h5py
import json
import logging
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, AugmentedRecovery, RecoverySubtask,
    RecoveryCollectionStatus, SUBTYPE_TO_RBG,
)
from error_benchmark.framework.recovery_mimicgen import (
    build_recovery_subtask_specs,
    make_pose_from_pos_quat,
    resolve_subtask_object_ref,
)
from error_benchmark.framework.recovery_segmenter import RecoverySegmenter
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.framework.error_scene_variants import (
    ErrorSceneVariantGenerator,
)
from error_benchmark.scripts.utils.script_utils import NumpyEncoder, create_env, load_task_registry, load_error_scenes
from error_benchmark.framework.utils.pose_transforms import (
    transform_source_data_segment_using_object_pose,
    unmake_pose,
    make_pose,
)


PREPARED_SOURCE_DIR = (
    PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success" / "prepared_source"
)


def _resolve_object_pose(object_poses, object_ref):
    """Look up an object pose with case-insensitive fallback.

    Scene labels may use capitalised names (e.g. 'Milk') while MimicGen's
    env_interface tracks lowercase names ('milk').
    """
    if object_ref in object_poses:
        return object_poses[object_ref]
    # Case-insensitive fallback
    ref_lower = object_ref.lower()
    for key in object_poses:
        if key.lower() == ref_lower:
            return object_poses[key]
    raise KeyError(
        f"Object '{object_ref}' not found in object_poses "
        f"(available: {list(object_poses.keys())})"
    )

_MIMICGEN_IMPORTS_DONE = False


def _lazy_mimicgen_imports():
    global _MIMICGEN_IMPORTS_DONE, make_interface, WaypointSequence, WaypointTrajectory
    if _MIMICGEN_IMPORTS_DONE:
        return

    from mimicgen.env_interfaces.base import make_interface as _make_interface
    from mimicgen.datagen.waypoint import (
        WaypointSequence as _WaypointSequence,
        WaypointTrajectory as _WaypointTrajectory,
    )

    make_interface = _make_interface
    WaypointSequence = _WaypointSequence
    WaypointTrajectory = _WaypointTrajectory
    _MIMICGEN_IMPORTS_DONE = True


class RecoveryAugmenter:
    """
    Augments recovery demonstrations using three strategies:
      A) Faithful MimicGen-style scene augmentation from recovery demos
      B) Cross-degree: add rotation/translation perturbation to D0 demos
      C) Cross-subtype: transfer recovery actions between subtypes in same RBG
    """

    NO_WARP_SUBTASKS = {"retract", "correct_position"}

    def __init__(self, env_wrapper, task_config: dict, aug_config: dict,
                 rng: np.random.RandomState):
        self.env = env_wrapper
        self.task_config = task_config
        self.aug_config = aug_config
        self.rng = rng

        self.scene_config = aug_config.get('scene_augmentation', {})
        self.cross_deg_config = aug_config.get('cross_degree', {})
        self.cross_sub_config = aug_config.get('cross_subtype', {})
        self.generator_mode = aug_config.get('generator_mode', 'faithful_mimicgen')
        self.scene_variant_mode = self.scene_config.get(
            'mode', 'error_scene_reset_sampler'
        )
        self.pos_noise_std = self.scene_config.get('position_noise_std', 0.005)
        self.validate_success = self.scene_config.get('success_validation', True)
        self.subtask_spec_config = aug_config.get('recovery_task_spec', {})
        self.subtask_specs = build_recovery_subtask_specs(self.subtask_spec_config)
        execution_cfg = self.subtask_spec_config.get('execution', {})
        self.transform_first_robot_pose = bool(
            execution_cfg.get('transform_first_robot_pose', False)
        )
        self.interpolate_from_last_target_pose = bool(
            execution_cfg.get('interpolate_from_last_target_pose', True)
        )
        self.expected_sequences_by_rbg = self.subtask_spec_config.get(
            'expected_sequences_by_rbg', {}
        )
        self.expected_sequences_by_subtype = self.subtask_spec_config.get(
            'expected_sequences_by_subtype', {}
        )
        self._variant_generator = ErrorSceneVariantGenerator(
            env_wrapper=self.env,
            task_config=self.task_config,
            rng=self.rng,
            scene_config=self.scene_config,
        )
        self._env_interface = None
        self._env_interface_task_name = None

    def augment_scene(
        self,
        source_demo: Union[RecoveryDemo, Sequence[RecoveryDemo]],
        target_scenes: List[dict],
        max_augments: int = 20,
        video_dir: Optional[Path] = None,
    ) -> List[AugmentedRecovery]:
        """
        3A: Scene configuration augmentation (same-scene variant mode).

        For each source demo, generate variants of its OWN error scene by
        randomising non-anchored object placements (analogous to MimicGen's
        env.reset()).  The robot pose and error state stay identical; only
        task-object positions change.

        Args:
            source_demo: One or more recovery demos with MimicGen-compatible target poses
            target_scenes: List of error scene dicts (used to look up each demo's source scene)
            max_augments: Maximum augmented demos to generate across all source demos
            video_dir: If set, record MP4 video for each attempt into this directory.

        Returns:
            List of successful AugmentedRecovery objects
        """
        logger = logging.getLogger(__name__)

        source_demos = self._normalize_source_demos(source_demo)
        if not source_demos:
            return []

        scenes_by_id = {
            scene.get("scene_id"): scene
            for scene in target_scenes
            if scene.get("_npz_path")
        }

        results = []
        max_cycles = max(1, int(self.scene_config.get('max_scene_cycles', 3)))
        max_ratio = int(self.scene_config.get('max_augment_ratio', 20))
        total_attempts = 0

        for demo in source_demos:
            if len(results) >= max_augments:
                break

            source_scene = self._find_source_scene(demo, scenes_by_id)
            if source_scene is None:
                logger.warning(
                    "Source scene %s not found for demo %s, skipping",
                    demo.scene_id, demo.demo_id,
                )
                continue

            per_demo_target = min(max_ratio, max_augments - len(results))
            per_demo_max_attempts = per_demo_target * max_cycles
            demo_attempts = 0
            demo_successes = 0

            while demo_successes < per_demo_target and demo_attempts < per_demo_max_attempts:
                demo_attempts += 1
                total_attempts += 1
                try:
                    aug = self._replay_with_warping(
                        [demo], source_scene, video_dir=video_dir,
                    )
                    if aug is not None:
                        results.append(aug)
                        demo_successes += 1
                except Exception as e:
                    logger.warning(
                        f"Scene augmentation failed for {demo.demo_id}: {e}",
                        exc_info=True,
                    )
                    continue

            logger.info(
                "  Demo %s: %d/%d successful (%d attempts)",
                demo.demo_id, demo_successes, per_demo_target, demo_attempts,
            )

        logger.info(
            "Scene augmentation: %d/%d successful (total attempts: %d)",
            len(results), max_augments, total_attempts,
        )
        return results

    @staticmethod
    def _find_source_scene(
        demo: RecoveryDemo, scenes_by_id: Dict[str, dict],
    ) -> Optional[dict]:
        """Find the error scene that *demo* was collected on."""
        if demo.scene_id and demo.scene_id in scenes_by_id:
            return scenes_by_id[demo.scene_id]
        # Fallback: use the demo's stored scene NPZ path directly
        if demo.scene_npz_path and Path(demo.scene_npz_path).exists():
            return {
                "scene_id": demo.scene_id,
                "_npz_path": demo.scene_npz_path,
            }
        return None

    def augment_cross_degree(
        self,
        source_demo: RecoveryDemo,
        target_degree: str,
    ) -> Optional[AugmentedRecovery]:
        """
        3B: Cross-degree augmentation.

        Take a D0 demo and add rotation perturbations to create D1 variants.

        Args:
            source_demo: Source demo (typically D0)
            target_degree: Target degree ("D1")

        Returns:
            AugmentedRecovery or None if failed
        """
        if source_demo.actions is None:
            return None

        source_degree = source_demo.degree
        if source_degree == target_degree:
            return None

        # Get perturbation parameters
        if target_degree == "D1":
            config = self.cross_deg_config.get('d0_to_d1', {})
            rot_range = config.get('rotation_range', [0.2, 0.8])
            pos_range = None
        else:
            return None

        # Perturb actions
        augmented_actions = source_demo.actions.copy()
        action_dim = augmented_actions.shape[1]

        # Add rotation perturbation to actions (dims 3:6 for OSC)
        if action_dim > 3 and rot_range is not None:
            rot_magnitude = self.rng.uniform(rot_range[0], rot_range[1])
            rot_axis = self.rng.randn(3)
            rot_axis /= max(np.linalg.norm(rot_axis), 1e-8)
            rot_perturbation = rot_magnitude * rot_axis

            # Apply to rotation action dimensions with temporal smoothing
            for t in range(len(augmented_actions)):
                # Smooth factor: larger in the middle of the trajectory
                smooth = np.sin(np.pi * t / max(len(augmented_actions), 1))
                augmented_actions[t, 3:6] += rot_perturbation * smooth * 0.1

        # Add position perturbation if specified
        if pos_range is not None:
            pos_magnitude = self.rng.uniform(pos_range[0], pos_range[1])
            pos_dir = self.rng.randn(3)
            pos_dir[2] = abs(pos_dir[2])  # Bias upward
            pos_dir /= max(np.linalg.norm(pos_dir), 1e-8)
            pos_perturbation = pos_magnitude * pos_dir

            for t in range(len(augmented_actions)):
                smooth = np.sin(np.pi * t / max(len(augmented_actions), 1))
                augmented_actions[t, :3] += pos_perturbation * smooth * 0.05

        # Clip actions
        augmented_actions = np.clip(augmented_actions, -1.0, 1.0)

        # Create new error_name (keep same, just change degree)
        new_subtype = f"{source_demo.error_name}_{target_degree}"

        return AugmentedRecovery(
            augmented_id=f"xdeg_{source_demo.demo_id}_{target_degree}_{self.rng.randint(10000):04d}",
            source_demo_id=source_demo.demo_id,
            augmentation_type="cross_degree",
            task_name=source_demo.task_name,
            error_name=source_demo.error_name,
            degree=target_degree,
            subtype_id=new_subtype,
            target_scene_id=source_demo.scene_id,
            success=source_demo.success,  # Assumed valid (not simulated)
            num_steps=len(augmented_actions),
            actions=augmented_actions,
            states=source_demo.states,  # Keep original states as reference
            subtasks=source_demo.subtasks,  # Keep same segmentation
            metadata={
                'source_degree': source_degree,
                'target_degree': target_degree,
                'rot_perturbation': rot_range,
                'pos_perturbation': pos_range,
            },
        )

    def augment_cross_subtype(
        self,
        source_demo: RecoveryDemo,
        target_subtype: str,
        target_scenes: List[dict],
    ) -> List[AugmentedRecovery]:
        """
        3C: Cross-subtype augmentation within same RBG.

        Transfer recovery actions from one subtype to another in the same
        Recovery Behavior Group (same motor primitive structure).  Uses the
        demo's own source scene (same-scene variant mode) to avoid cross-scene
        robot-pose mismatches; only the subtype label is changed.

        Args:
            source_demo: Source recovery demo
            target_subtype: Target subtype ID (e.g., "collision_empty_D0")
            target_scenes: Error scenes (used only to locate source scene)

        Returns:
            List of AugmentedRecovery objects
        """
        source_rbg = SUBTYPE_TO_RBG.get(source_demo.subtype_id, "")
        target_rbg = SUBTYPE_TO_RBG.get(target_subtype, "")

        if source_rbg != target_rbg:
            return []

        # Parse target subtype
        for d in ['D0', 'D1']:
            if target_subtype.endswith(f'_{d}'):
                target_error_name = target_subtype[:-(len(d) + 1)]
                target_degree = d
                break
        else:
            return []

        # Same-scene variant: use demo's own source scene
        scenes_by_id = {
            s.get("scene_id"): s for s in target_scenes if s.get("_npz_path")
        }
        source_scene = self._find_source_scene(source_demo, scenes_by_id)
        if source_scene is None:
            # Fallback: construct from demo's stored path
            if source_demo.scene_npz_path and Path(source_demo.scene_npz_path).exists():
                source_scene = {
                    "scene_id": source_demo.scene_id,
                    "_npz_path": source_demo.scene_npz_path,
                }
            else:
                logger.warning(
                    "Source scene %s not found for cross-subtype demo %s",
                    source_demo.scene_id, source_demo.demo_id,
                )
                return []

        results = []
        max_attempts = 5 * 3  # cap * max_cycles
        attempts = 0
        while len(results) < 5 and attempts < max_attempts:
            attempts += 1
            try:
                aug = self._replay_with_warping(source_demo, source_scene)
                if aug is not None:
                    aug.augmentation_type = "cross_subtype"
                    aug.error_name = target_error_name
                    aug.degree = target_degree
                    aug.subtype_id = target_subtype
                    aug.augmented_id = (
                        f"xsub_{source_demo.demo_id}_{target_subtype}_"
                        f"{self.rng.randint(10000):04d}"
                    )
                    results.append(aug)
            except Exception:
                continue

        return results

    @staticmethod
    def _decode_h5_attr(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _normalize_source_demos(
        source_demo: Union[RecoveryDemo, Sequence[RecoveryDemo]],
    ) -> List[RecoveryDemo]:
        if isinstance(source_demo, RecoveryDemo):
            return [source_demo]
        return [demo for demo in source_demo if demo is not None]

    def _get_env_interface(self, task_name: str):
        if (
            self._env_interface is not None
            and self._env_interface_task_name == task_name
        ):
            return self._env_interface

        _lazy_mimicgen_imports()
        source_dataset = PREPARED_SOURCE_DIR / f"{task_name}_src.hdf5"
        if not source_dataset.exists():
            raise FileNotFoundError(
                f"Missing prepared MimicGen source dataset for task '{task_name}': "
                f"{source_dataset}"
            )

        with h5py.File(source_dataset, "r") as f:
            demo_keys = sorted(f["data"].keys())
            if not demo_keys:
                raise ValueError(f"No demos found in MimicGen source dataset: {source_dataset}")
            datagen_info = f["data"][demo_keys[0]]["datagen_info"]
            env_interface_name = self._decode_h5_attr(
                datagen_info.attrs["env_interface_name"]
            )
            env_interface_type = self._decode_h5_attr(
                datagen_info.attrs["env_interface_type"]
            )

        self._env_interface = make_interface(
            name=env_interface_name,
            interface_type=env_interface_type,
            env=self.env._env,
        )
        self._env_interface_task_name = task_name
        return self._env_interface

    def _validate_source_demo(self, demo: RecoveryDemo) -> None:
        if demo.actions is None or len(demo.actions) == 0:
            raise ValueError(f"Recovery demo '{demo.demo_id}' is missing actions")
        if demo.target_poses is None:
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' is missing target_poses. "
                "Run backfill_recovery_target_poses.py first."
            )
        if demo.eef_positions is None or demo.eef_orientations is None:
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' is missing eef pose data. "
                "Run backfill_recovery_target_poses.py first."
            )
        if not demo.subtasks:
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' has no subtask segmentation"
            )
        if len(demo.target_poses) != len(demo.actions):
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' has {len(demo.actions)} actions but "
                f"{len(demo.target_poses)} target poses"
            )
        if len(demo.eef_positions) < len(demo.actions) + 1:
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' has incomplete eef_positions"
            )
        if len(demo.eef_orientations) < len(demo.eef_positions):
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' has incomplete eef_orientations"
            )

        observed_labels = [subtask.label for subtask in demo.subtasks]
        unknown_labels = [
            label for label in observed_labels if label not in self.subtask_specs
        ]
        if unknown_labels:
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' contains unsupported subtasks: "
                f"{unknown_labels}"
            )

        expected_sequence = self.expected_sequences_by_subtype.get(demo.subtype_id)
        if expected_sequence is None:
            expected_sequence = self.expected_sequences_by_rbg.get(demo.rbg)
        if expected_sequence is not None and observed_labels != list(expected_sequence):
            raise ValueError(
                f"Recovery demo '{demo.demo_id}' sequence {observed_labels} does not "
                f"match configured sequence {list(expected_sequence)}"
            )

    def _resolve_target_object(
        self,
        source_demos: List[RecoveryDemo],
        target_scene: dict,
    ) -> str:
        scene_labels = target_scene.get("labels", {})
        target_object = scene_labels.get("target_object", "")
        if target_object:
            return target_object

        for demo in source_demos:
            target_object = demo.metadata.get("target_object", "")
            if target_object:
                return target_object

        return self.env.get_target_object() or ""

    def _get_source_object_pose(
        self,
        source_demo: RecoveryDemo,
        object_name: str,
        step: int,
    ) -> np.ndarray:
        if (
            source_demo.object_positions is None
            or object_name not in source_demo.object_positions
        ):
            raise ValueError(
                f"Recovery demo '{source_demo.demo_id}' is missing object positions for "
                f"'{object_name}'"
            )
        if (
            source_demo.object_orientations is None
            or object_name not in source_demo.object_orientations
        ):
            raise ValueError(
                f"Recovery demo '{source_demo.demo_id}' is missing object orientations for "
                f"'{object_name}'. Run backfill_recovery_target_poses.py first."
            )
        pos = source_demo.object_positions[object_name][step]
        quat = source_demo.object_orientations[object_name][step]
        return make_pose_from_pos_quat(pos, quat)

    def _select_source_candidate_index(
        self,
        strategy_name: str,
        candidates: List[dict],
        current_object_pose: Optional[np.ndarray],
        selection_kwargs: Optional[dict],
    ) -> int:
        if len(candidates) == 1:
            return 0

        selection_kwargs = selection_kwargs or {}
        if strategy_name == "random":
            return int(self.rng.randint(0, len(candidates)))

        if strategy_name != "nearest_neighbor_object":
            raise ValueError(
                f"Unsupported recovery selection strategy '{strategy_name}'"
            )
        if current_object_pose is None:
            # No object pose available (e.g. pick_place bins not tracked) —
            # fall back to random selection.
            return int(self.rng.randint(0, len(candidates)))

        src_object_poses = np.array(
            [candidate["src_object_pose"] for candidate in candidates]
        )
        all_src_obj_pos, all_src_obj_rot = unmake_pose(src_object_poses)
        obj_pos, obj_rot = unmake_pose(current_object_pose)
        obj_pos = obj_pos.reshape(-1, 3)
        obj_rot_T = obj_rot.T.reshape(-1, 3, 3)

        pos_weight = float(selection_kwargs.get("pos_weight", 1.0))
        rot_weight = float(selection_kwargs.get("rot_weight", 1.0))
        nn_k = int(selection_kwargs.get("nn_k", 3))

        pos_dists = np.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(axis=-1))
        delta_rot = np.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (np.trace(delta_rot, axis1=-2, axis2=-1) - 1.0) / 2.0
        arc_cos_in = np.clip(arc_cos_in, -1.0, 1.0)
        rot_dists = np.arccos(arc_cos_in)
        dists = pos_weight * pos_dists + rot_weight * rot_dists

        nn_k = min(nn_k, len(dists))
        top_k = np.argsort(dists)[:nn_k]
        return int(top_k[self.rng.randint(0, nn_k)])

    def _select_source_subtask(
        self,
        source_demos: List[RecoveryDemo],
        subtask_index: int,
        label: str,
        object_ref: Optional[str],
        current_object_pose: Optional[np.ndarray],
    ) -> dict:
        spec = self.subtask_specs[label]
        candidates = []
        for demo in source_demos:
            source_subtask = demo.subtasks[subtask_index]
            if source_subtask.label != label:
                raise ValueError(
                    f"Recovery demo '{demo.demo_id}' has subtask '{source_subtask.label}' "
                    f"at index {subtask_index}, expected '{label}'"
                )
            src_object_pose = None
            if object_ref is not None:
                src_object_pose = self._get_source_object_pose(
                    demo, object_ref, source_subtask.start_step
                )
            candidates.append(
                {
                    "demo": demo,
                    "subtask": source_subtask,
                    "src_object_pose": src_object_pose,
                }
            )

        selected_idx = self._select_source_candidate_index(
            strategy_name=spec.selection_strategy,
            candidates=candidates,
            current_object_pose=current_object_pose,
            selection_kwargs=spec.selection_strategy_kwargs,
        )
        return candidates[selected_idx]

    def _build_source_segment(
        self,
        source_demo: RecoveryDemo,
        source_subtask: RecoverySubtask,
        include_first_robot_pose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start = source_subtask.start_step
        end = source_subtask.end_step

        src_target_poses = np.asarray(source_demo.target_poses[start:end])
        src_gripper_actions = np.asarray(source_demo.actions[start:end, -1:])
        if len(src_target_poses) == 0 or len(src_gripper_actions) == 0:
            raise ValueError(
                f"Recovery demo '{source_demo.demo_id}' has an empty source segment for "
                f"subtask '{source_subtask.label}'"
            )

        if include_first_robot_pose:
            start_pose = make_pose_from_pos_quat(
                source_demo.eef_positions[start],
                source_demo.eef_orientations[start],
            )[None]
            src_eef_poses = np.concatenate([start_pose, src_target_poses], axis=0)
        else:
            src_eef_poses = np.array(src_target_poses)

        src_gripper_actions = np.concatenate(
            [src_gripper_actions[0:1], src_gripper_actions], axis=0
        )
        return src_eef_poses, src_gripper_actions

    def _execute_waypoint_trajectory(self, traj_to_execute, env_interface,
                                     render_fn=None, subtask_label="") -> dict:
        _exec_logger = logging.getLogger(__name__)
        actions = []
        states = []
        eef_positions = []
        gripper_states = []
        target_poses = []
        success = False
        _step_count = 0

        for seq_idx, seq in enumerate(traj_to_execute.waypoint_sequences):
            for wp_idx, waypoint in enumerate(seq):
                action_pose = env_interface.target_pose_to_action(
                    target_pose=waypoint.pose
                )
                if waypoint.noise is not None:
                    action_pose = action_pose + waypoint.noise * self.rng.randn(
                        *action_pose.shape
                    )
                    action_pose = np.clip(action_pose, -1.0, 1.0)

                play_action = np.concatenate(
                    [action_pose, waypoint.gripper_action], axis=0
                )

                # Per-step log for first 10 steps and every 20th step
                _pre_eef = self.env.get_eef_pos()
                if _step_count < 10 or _step_count % 20 == 0:
                    _wp_pos, _ = unmake_pose(waypoint.pose[None])
                    _exec_logger.info(
                        f"    [{subtask_label}] step={_step_count} seq={seq_idx} wp={wp_idx} | "
                        f"waypoint_pos={_wp_pos[0].round(4)} | "
                        f"pre_eef={_pre_eef.round(4)} | "
                        f"action={action_pose.round(4)} | "
                        f"gap={np.linalg.norm(_wp_pos[0]-_pre_eef)*100:.1f}cm"
                    )

                datagen_info = env_interface.get_datagen_info(action=play_action)
                self.env.step(play_action)

                actions.append(play_action.copy())
                target_poses.append(np.array(datagen_info.target_pose))
                states.append(self.env.get_sim_state_flat())
                eef_positions.append(self.env.get_eef_pos())
                gripper_states.append(self.env.get_gripper_closed_norm())
                success = success or self.env.check_success()

                if render_fn is not None:
                    render_fn(step=len(actions), success=success)
                _step_count += 1

        return {
            "actions": np.array(actions),
            "target_poses": np.array(target_poses),
            "states": states,
            "eef_positions": eef_positions,
            "gripper_states": gripper_states,
            "success": success,
        }

    def _diagnose_task_completion(self, augmented_subtasks=None) -> dict:
        """Inspect object/EEF state to diagnose why check_success() failed."""
        diag = {
            "eef_pos": self.env.get_eef_pos().tolist(),
            "gripper_closed": float(self.env.get_gripper_closed_norm()),
        }
        for obj_name in self.env.get_all_object_names():
            try:
                pos, quat = self.env.get_object_pose(obj_name)
                diag[f"obj_{obj_name}_pos"] = pos.tolist()
            except Exception:
                pass
        if augmented_subtasks:
            diag["subtasks_executed"] = [
                {"label": st.label, "actions": st.duration}
                for st in augmented_subtasks
            ]
        # stack-specific: relative cube positions
        a_pos = diag.get("obj_cubeA_pos")
        b_pos = diag.get("obj_cubeB_pos")
        if a_pos is not None and b_pos is not None:
            a = np.array(a_pos)
            b = np.array(b_pos)
            diag["cubeA_cubeB_xy_dist"] = float(np.linalg.norm(a[:2] - b[:2]))
            diag["cubeA_height"] = float(a[2])
            diag["cubeA_above_cubeB"] = float(a[2] - b[2])
        return diag

    def _replay_with_warping(
        self,
        source_demo: Union[RecoveryDemo, Sequence[RecoveryDemo]],
        target_scene: dict,
        diagnose: bool = False,
        video_dir: Optional[Path] = None,
    ) -> Optional[AugmentedRecovery]:
        if self.generator_mode == "legacy_replay":
            source_demos = self._normalize_source_demos(source_demo)
            if not source_demos:
                return None
            return self._legacy_replay_with_warping(source_demos[0], target_scene)
        return self._replay_with_mimicgen_targets(
            source_demo, target_scene, diagnose=diagnose,
            video_dir=video_dir,
        )

    def _make_failed_augmented(
        self,
        fail_reason: str,
        base_demo: Optional[RecoveryDemo] = None,
        scene_id: str = "",
        variant=None,
        actions_list=None,
        augmented_subtasks=None,
    ) -> AugmentedRecovery:
        """Build a failed AugmentedRecovery with diagnosis metadata."""
        diag = self._diagnose_task_completion(augmented_subtasks)
        diag["_fail_reason"] = fail_reason
        meta = {"_fail_reason": fail_reason, "_diagnosis": diag}
        if variant is not None:
            meta.update({
                "variant_id": variant.variant_id,
                "variant_index": variant.variant_index,
                "base_scene_id": variant.base_scene_id,
                "randomized_objects": list(variant.randomized_objects),
                "anchored_objects": list(variant.anchored_objects),
            })
        return AugmentedRecovery(
            augmented_id=f"diag_{scene_id}",
            source_demo_id=base_demo.demo_id if base_demo else "",
            augmentation_type="scene",
            task_name=base_demo.task_name if base_demo else "",
            error_name=base_demo.error_name if base_demo else "",
            degree=base_demo.degree if base_demo else "",
            subtype_id=base_demo.subtype_id if base_demo else "",
            target_scene_id=scene_id,
            success=False,
            num_steps=len(actions_list) if actions_list else 0,
            actions=np.array(actions_list) if actions_list else None,
            subtasks=augmented_subtasks,
            metadata=meta,
        )

    def _replay_with_mimicgen_targets(
        self,
        source_demo: Union[RecoveryDemo, Sequence[RecoveryDemo]],
        target_scene: dict,
        diagnose: bool = False,
        video_dir: Optional[Path] = None,
    ) -> Optional[AugmentedRecovery]:
        logger = logging.getLogger(__name__)
        source_demos = self._normalize_source_demos(source_demo)
        if not source_demos:
            return None
        for demo in source_demos:
            self._validate_source_demo(demo)

        base_demo = source_demos[0]
        task_name = base_demo.task_name or self.task_config.get("task_name", "")
        env_interface = self._get_env_interface(task_name)

        scene_id = target_scene.get("scene_id", "")
        # Use demo's states[0] as base state — this is the actual robot
        # configuration when the human started teleoperating, which may differ
        # from the scene NPZ snapshot (after injection replay / OSC warm-up).
        demo_base_state = (
            np.asarray(base_demo.states[0], dtype=np.float64)
            if base_demo.states is not None and len(base_demo.states) > 0
            else None
        )
        variant = self._variant_generator.generate_variant(
            target_scene, override_base_state=demo_base_state,
        )
        if variant is None:
            if diagnose:
                return self._make_failed_augmented(
                    "variant_generation_failed", base_demo, scene_id,
                )
            return None
        self.env.set_sim_state_flat(variant.sim_state)
        self.env.forward()

        target_object = self._resolve_target_object(source_demos, target_scene)
        if not target_object:
            if diagnose:
                return self._make_failed_augmented(
                    "empty_target_object", base_demo, scene_id, variant,
                )
            return None

        scene_labels = {}
        scene_labels.update(base_demo.metadata.get("scene_labels", {}))
        scene_labels.update(target_scene.get("labels", {}))

        # Video recording setup
        _video_recorder = None
        _current_subtask_label = ""
        if video_dir is not None:
            from error_benchmark.framework.video_recorder import (
                VideoRecorder, build_augmentation_overlay,
            )
            _attempt_id = f"{base_demo.demo_id}_{scene_id}_{self.rng.randint(10000):04d}"
            _video_path = video_dir / f"{_attempt_id}.mp4"
            _video_recorder = VideoRecorder(
                _video_path, fps=20, resolution=(512, 512),
                camera_names=["agentview", "robot0_eye_in_hand"],
            )
            # Capture initial frame
            _video_recorder.capture_frame(self.env, overlay_text=[
                f"scene: {scene_id}",
                f"source: {base_demo.demo_id}",
                "INIT",
            ])

        def _render_fn(step, success):
            if _video_recorder is None:
                return
            overlay = build_augmentation_overlay(
                step=step, total_steps=step,
                subtask_label=_current_subtask_label,
                success=success if success else None,
                eef_pos=self.env.get_eef_pos(),
            )
            _video_recorder.capture_frame(self.env, overlay_text=overlay)

        # Settle: step zero actions until EEF stabilizes
        _settle_max = 50
        _settle_thresh = 1e-4  # m per step
        _neutral = self.env.get_neutral_action()
        _prev_eef = self.env.get_eef_pos().copy()
        for _si in range(_settle_max):
            self.env.step(_neutral)
            _cur_eef = self.env.get_eef_pos()
            if np.linalg.norm(_cur_eef - _prev_eef) < _settle_thresh:
                break
            _prev_eef = _cur_eef.copy()
        logger.info(f"  Settled after {_si+1} steps, EEF={self.env.get_eef_pos().round(4)}")

        states_list = [self.env.get_sim_state_flat()]
        eef_list = [self.env.get_eef_pos()]
        grip_list = [self.env.get_gripper_closed_norm()]
        target_pose_list = []
        actions_list = []
        augmented_subtasks = []
        selected_sources = []
        prev_executed_traj = None

        for subtask_index, template_subtask in enumerate(base_demo.subtasks):
            label = template_subtask.label
            spec = self.subtask_specs[label]
            cur_datagen_info = env_interface.get_datagen_info()
            object_ref = resolve_subtask_object_ref(
                spec=spec,
                task_name=task_name,
                task_config=self.task_config,
                target_object=target_object,
                scene_labels=scene_labels,
            )
            if spec.object_ref_strategy != "none" and object_ref is None:
                # Placement ref unavailable (e.g. pick_place bins, stack cubeB).
                # Proceed without object-centric warping for this subtask.
                logger.debug(
                    "No object reference for subtask '%s' in task '%s' "
                    "(strategy=%s) — skipping object-centric warping",
                    label, task_name, spec.object_ref_strategy,
                )
            current_object_pose = (
                _resolve_object_pose(cur_datagen_info.object_poses, object_ref)
                if object_ref is not None
                else None
            )

            selected = self._select_source_subtask(
                source_demos=source_demos,
                subtask_index=subtask_index,
                label=label,
                object_ref=object_ref,
                current_object_pose=current_object_pose,
            )
            source_subtask = selected["subtask"]
            selected_demo = selected["demo"]

            include_first_robot_pose = (
                subtask_index == 0 or self.transform_first_robot_pose
            )
            src_eef_poses, src_gripper_actions = self._build_source_segment(
                source_demo=selected_demo,
                source_subtask=source_subtask,
                include_first_robot_pose=include_first_robot_pose,
            )

            if object_ref is not None:
                transformed_eef_poses = transform_source_data_segment_using_object_pose(
                    obj_pose=current_object_pose,
                    src_eef_poses=src_eef_poses,
                    src_obj_pose=selected["src_object_pose"],
                )
            else:
                transformed_eef_poses = src_eef_poses

            # ── Diagnostic logging: warping inputs/outputs ──
            _src_obj_pos, _ = unmake_pose(selected["src_object_pose"]) if object_ref else (None, None)
            _tgt_obj_pos, _ = unmake_pose(current_object_pose) if object_ref is not None else (None, None)
            _src_eef_first_pos, _ = unmake_pose(src_eef_poses[0:1])
            _src_eef_last_pos, _ = unmake_pose(src_eef_poses[-1:])
            _warp_first_pos, _ = unmake_pose(transformed_eef_poses[0:1])
            _warp_last_pos, _ = unmake_pose(transformed_eef_poses[-1:])
            _cur_eef_pos = self.env.get_eef_pos()
            logger.info(
                f"  [{label}] object_ref={object_ref} | "
                f"src_obj={_src_obj_pos[0].round(4) if _src_obj_pos is not None else 'N/A'} -> "
                f"tgt_obj={_tgt_obj_pos[0].round(4) if _tgt_obj_pos is not None else 'N/A'} | "
                f"obj_delta={np.linalg.norm((_tgt_obj_pos[0]-_src_obj_pos[0]))*100:.1f}cm"
                if _tgt_obj_pos is not None and _src_obj_pos is not None else
                f"  [{label}] object_ref=None (no warping)"
            )
            logger.info(
                f"  [{label}] src_eef: first={_src_eef_first_pos[0].round(4)} last={_src_eef_last_pos[0].round(4)} | "
                f"warped_eef: first={_warp_first_pos[0].round(4)} last={_warp_last_pos[0].round(4)} | "
                f"cur_eef={_cur_eef_pos.round(4)}"
            )

            traj_to_execute = WaypointTrajectory()
            if (
                self.interpolate_from_last_target_pose
                and prev_executed_traj is not None
            ):
                init_sequence = WaypointSequence(
                    sequence=[prev_executed_traj.last_waypoint]
                )
            else:
                init_sequence = WaypointSequence.from_poses(
                    poses=cur_datagen_info.eef_pose[None],
                    gripper_actions=src_gripper_actions[0:1],
                    action_noise=spec.action_noise,
                )
            traj_to_execute.add_waypoint_sequence(init_sequence)

            transformed_seq = WaypointSequence.from_poses(
                poses=transformed_eef_poses,
                gripper_actions=src_gripper_actions,
                action_noise=spec.action_noise,
            )
            transformed_traj = WaypointTrajectory()
            transformed_traj.add_waypoint_sequence(transformed_seq)
            traj_to_execute.merge(
                transformed_traj,
                num_steps_interp=spec.num_interpolation_steps,
                num_steps_fixed=spec.num_fixed_steps,
                action_noise=(
                    float(spec.apply_noise_during_interpolation)
                    * spec.action_noise
                ),
            )
            traj_to_execute.pop_first()

            _current_subtask_label = label
            start_step = len(actions_list)
            exec_results = self._execute_waypoint_trajectory(
                traj_to_execute, env_interface, subtask_label=label,
                render_fn=_render_fn if _video_recorder is not None else None,
            )
            if len(exec_results["actions"]) == 0:
                if _video_recorder is not None:
                    _video_recorder.capture_frame(self.env, overlay_text=[
                        f"FAIL: subtask_empty_actions:{label}",
                    ])
                    _video_recorder.close()
                if diagnose:
                    return self._make_failed_augmented(
                        f"subtask_empty_actions:{label}",
                        base_demo, scene_id, variant,
                        actions_list, augmented_subtasks,
                    )
                return None

            actions_list.extend(exec_results["actions"])
            target_pose_list.extend(exec_results["target_poses"])
            states_list.extend(exec_results["states"])
            eef_list.extend(exec_results["eef_positions"])
            grip_list.extend(exec_results["gripper_states"])

            # ── Diagnostic logging: execution results ──
            _exec_eef = np.array(exec_results["eef_positions"])
            _exec_eef_final = _exec_eef[-1] if len(_exec_eef) > 0 else np.zeros(3)
            _warp_target_pos, _ = unmake_pose(transformed_eef_poses[-1:])
            _eef_vs_warp = np.linalg.norm(_exec_eef_final - _warp_target_pos[0]) * 100
            if object_ref is not None:
                _obj_pos_after = self.env.get_object_pose(object_ref)[0]
                _eef_vs_obj = np.linalg.norm(_exec_eef_final - _obj_pos_after) * 100
                _min_dist_to_obj = np.min(np.linalg.norm(_exec_eef - _obj_pos_after, axis=1)) * 100
                logger.info(
                    f"  [{label}] EXEC: {len(_exec_eef)} steps | "
                    f"eef_final={_exec_eef_final.round(4)} | "
                    f"eef_vs_warped_target={_eef_vs_warp:.1f}cm | "
                    f"eef_vs_{object_ref}={_eef_vs_obj:.1f}cm | "
                    f"min_dist_to_{object_ref}={_min_dist_to_obj:.1f}cm | "
                    f"gripper={exec_results['gripper_states'][-1]:.3f}"
                )
            else:
                logger.info(
                    f"  [{label}] EXEC: {len(_exec_eef)} steps | "
                    f"eef_final={_exec_eef_final.round(4)} | "
                    f"eef_vs_warped_target={_eef_vs_warp:.1f}cm"
                )

            end_step = len(actions_list)
            augmented_subtasks.append(
                RecoverySubtask(
                    label=label,
                    start_step=start_step,
                    end_step=end_step,
                    duration=end_step - start_step,
                    metadata={
                        "source_demo_id": selected_demo.demo_id,
                        "source_start_step": source_subtask.start_step,
                        "source_end_step": source_subtask.end_step,
                        "object_ref": object_ref,
                    },
                )
            )
            selected_sources.append(
                {
                    "label": label,
                    "source_demo_id": selected_demo.demo_id,
                    "object_ref": object_ref,
                }
            )
            prev_executed_traj = traj_to_execute

        success = bool(self.env.check_success())

        # Finalize video: add final frame with result, rename on failure
        if _video_recorder is not None:
            fail_reason = "" if success else "success_check_failed"
            final_overlay = build_augmentation_overlay(
                step=len(actions_list), total_steps=len(actions_list),
                success=success, fail_reason=fail_reason,
                eef_pos=self.env.get_eef_pos(),
            )
            _video_recorder.capture_frame(self.env, overlay_text=final_overlay)
            _video_recorder.close()
            # Rename: prefix with success/fail for easy filtering
            old_path = _video_recorder.output_path
            tag = "ok" if success else f"FAIL_{fail_reason}"
            new_name = f"{tag}__{old_path.stem}{old_path.suffix}"
            new_path = old_path.parent / new_name
            try:
                old_path.rename(new_path)
            except OSError:
                pass

        if self.validate_success and not success:
            if diagnose:
                return self._make_failed_augmented(
                    "success_check_failed",
                    base_demo, scene_id, variant,
                    actions_list, augmented_subtasks,
                )
            return None

        all_actions = (
            np.array(actions_list)
            if actions_list
            else np.empty((0, self.env._env.action_dim))
        )
        return AugmentedRecovery(
            augmented_id=(
                f"aug_{base_demo.demo_id}_{variant.variant_id}_{self.rng.randint(10000):04d}"
            ),
            source_demo_id=base_demo.demo_id,
            augmentation_type="scene",
            task_name=base_demo.task_name,
            error_name=base_demo.error_name,
            degree=base_demo.degree,
            subtype_id=base_demo.subtype_id,
            target_scene_id=scene_id,
            success=success,
            num_steps=len(all_actions),
            actions=all_actions,
            states=np.array(states_list) if states_list else None,
            eef_positions=np.array(eef_list) if eef_list else None,
            target_poses=np.array(target_pose_list) if target_pose_list else None,
            gripper_states=np.array(grip_list) if grip_list else None,
            subtasks=augmented_subtasks,
            metadata={
                "target_object": target_object,
                "scene_labels": scene_labels,
                "generator_mode": self.generator_mode,
                "scene_variant_mode": self.scene_variant_mode,
                "base_scene_id": variant.base_scene_id,
                "variant_id": variant.variant_id,
                "variant_index": variant.variant_index,
                "randomized_objects": list(variant.randomized_objects),
                "anchored_objects": list(variant.anchored_objects),
                "variant_generation_attempts": variant.generation_attempts,
                "source_demo_ids": sorted(
                    {entry["source_demo_id"] for entry in selected_sources}
                ),
                "selected_sources": selected_sources,
            },
        )

    def _target_pose_to_action(self, target_pose_4x4: np.ndarray) -> np.ndarray:
        """Convert target EEF 4x4 pose to normalized OSC action (MimicGen-style).

        Computes relative delta from current EEF state, normalizes by
        controller output_max, and clips to [-1, 1].
        """
        from error_benchmark.framework.utils.pose_transforms import _mat2quat, quat2axisangle

        target_pos = target_pose_4x4[:3, 3]
        target_rot = target_pose_4x4[:3, :3]

        cur_pos = self.env.get_eef_pos()
        cur_quat = self.env.get_eef_quat()  # (w, x, y, z)
        cur_rot = self._quat_to_rot(cur_quat)

        # Controller bounds
        max_dpos = self.env._env.robots[0].controller.output_max[0]
        max_drot = self.env._env.robots[0].controller.output_max[3]

        # Normalized position delta
        delta_pos = np.clip((target_pos - cur_pos) / max_dpos, -1.0, 1.0)

        # Normalized rotation delta (axis-angle)
        delta_rot_mat = target_rot @ cur_rot.T
        delta_quat = _mat2quat(delta_rot_mat)  # (x, y, z, w)
        axis, angle = quat2axisangle(delta_quat)
        delta_rot_aa = axis * angle if abs(angle) > 1e-6 else np.zeros(3)
        delta_rot = np.clip(delta_rot_aa / max_drot, -1.0, 1.0)

        action = np.zeros(self.env._env.action_dim)
        action[:3] = delta_pos
        action[3:6] = delta_rot
        return action

    def _get_src_obj_pose_at_step(
        self, source_demo: RecoveryDemo, target_object: str, step: int,
    ) -> np.ndarray:
        """Get source object 4x4 pose at a given step."""
        pos = source_demo.object_positions[target_object][step]
        if (source_demo.object_orientations is not None
                and target_object in source_demo.object_orientations):
            quat = source_demo.object_orientations[target_object][step]
            rot = self._quat_to_rot(quat)
        else:
            rot = np.eye(3)
        return self._make_pose_4x4(pos, rot)

    def _build_segment_eef_poses(
        self, source_demo: RecoveryDemo, start: int, end: int,
    ) -> np.ndarray:
        """Build (N, 4, 4) EEF pose matrices for a trajectory segment.

        Indices [start, end] inclusive for poses (end+1 poses for end-start actions).
        """
        seg_end = min(end + 1, len(source_demo.eef_positions))
        seg_pos = source_demo.eef_positions[start:seg_end]
        n = len(seg_pos)
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, :3, 3] = seg_pos

        has_orient = (source_demo.eef_orientations is not None
                      and len(source_demo.eef_orientations) >= seg_end)
        if has_orient:
            for i in range(n):
                quat = source_demo.eef_orientations[start + i]
                poses[i, :3, :3] = self._quat_to_rot(quat)

        return poses

    def _legacy_replay_with_warping(
        self,
        source_demo: RecoveryDemo,
        target_scene: dict,
    ) -> Optional[AugmentedRecovery]:
        """
        Per-subtask closed-loop replay (MimicGen-style).

        For each subtask segment:
          1. Query current actual object pose
          2. Warp that segment's source EEF targets via object-centric transform
          3. Track warped targets with closed-loop control (target_pose_to_action)
        """
        from error_benchmark.framework.utils.pose_transforms import (
            transform_source_data_segment_using_object_pose,
        )

        npz_path = target_scene.get('_npz_path', '')
        scene_id = target_scene.get('scene_id', '')

        # Identify the target object
        target_object = (
            target_scene.get('labels', {}).get('target_object', '')
            or source_demo.metadata.get('target_object', '')
        )

        # Load target scene state (post-injection = error state)
        npz_data = np.load(npz_path)
        if 'post_sim_state' in npz_data:
            sim_state = npz_data['post_sim_state']
        else:
            sim_state = npz_data['sim_state']
        self.env.set_sim_state_flat(sim_state)
        self.env.forward()

        # Resolve target object name
        if not target_object or target_object not in self.env.get_all_object_names():
            target_object = self.env.get_target_object() or ''
            if not target_object:
                return None

        # Validate source demo has required data
        if (source_demo.object_positions is None
                or target_object not in source_demo.object_positions
                or source_demo.eef_positions is None
                or source_demo.actions is None):
            return None

        # Build subtask segments; fallback to single whole-trajectory segment
        if source_demo.subtasks:
            segments = [(st.start_step, st.end_step, st.label)
                        for st in source_demo.subtasks]
        else:
            segments = [(0, len(source_demo.actions), 'full')]

        # Execution buffers
        actions_list = []
        states_list = [self.env.get_sim_state_flat()]
        eef_list = [self.env.get_eef_pos()]
        grip_list = [self.env.get_gripper_closed_norm()]
        success = False

        for seg_start, seg_end, seg_label in segments:
            if success:
                break

            num_steps = seg_end - seg_start
            if num_steps <= 0:
                continue

            # NO_WARP subtasks: replay original actions directly
            if seg_label in self.NO_WARP_SUBTASKS:
                for t in range(seg_start, min(seg_end, len(source_demo.actions))):
                    action = source_demo.actions[t].copy()
                    noise = self.rng.randn(3) * self.pos_noise_std
                    action[:3] += noise
                    action = np.clip(action, -1.0, 1.0)
                    try:
                        self.env.step(action)
                    except Exception:
                        break
                    actions_list.append(action)
                    states_list.append(self.env.get_sim_state_flat())
                    eef_list.append(self.env.get_eef_pos())
                    grip_list.append(self.env.get_gripper_closed_norm())
                    if self.env.check_success():
                        success = True
                        break
                continue

            # ── Closed-loop warped segment ──

            # Query CURRENT actual object pose at this subtask boundary
            cur_obj_pos, cur_obj_quat = self.env.get_object_pose(target_object)
            cur_obj_pose = self._make_pose_4x4(
                cur_obj_pos, self._quat_to_rot(cur_obj_quat))

            # Source object pose at this segment's start
            src_obj_pose = self._get_src_obj_pose_at_step(
                source_demo, target_object, seg_start)

            # Source EEF poses for this segment (poses[0] = state at seg_start)
            src_eef_poses = self._build_segment_eef_poses(
                source_demo, seg_start, seg_end)

            # Object-centric warp
            warped_eef = transform_source_data_segment_using_object_pose(
                obj_pose=cur_obj_pose,
                src_eef_poses=src_eef_poses,
                src_obj_pose=src_obj_pose,
            )

            # Closed-loop tracking: for each step, compute action from
            # current EEF to warped target (MimicGen target_pose_to_action)
            for t in range(min(num_steps, len(warped_eef) - 1)):
                target_pose = warped_eef[t + 1]  # target for this step

                action = self._target_pose_to_action(target_pose)

                # Preserve gripper action from source demo
                src_action_idx = seg_start + t
                if src_action_idx < len(source_demo.actions):
                    action[-1] = source_demo.actions[src_action_idx, -1]

                # Add diversity noise
                noise = self.rng.randn(3) * self.pos_noise_std
                action[:3] += noise
                action = np.clip(action, -1.0, 1.0)

                try:
                    self.env.step(action)
                except Exception:
                    break

                actions_list.append(action)
                states_list.append(self.env.get_sim_state_flat())
                eef_list.append(self.env.get_eef_pos())
                grip_list.append(self.env.get_gripper_closed_norm())

                if self.env.check_success():
                    success = True
                    break

        if self.validate_success and not success:
            return None

        all_actions = np.array(actions_list) if actions_list else np.empty((0, self.env._env.action_dim))

        return AugmentedRecovery(
            augmented_id=f"aug_{source_demo.demo_id}_{scene_id}_{self.rng.randint(10000):04d}",
            source_demo_id=source_demo.demo_id,
            augmentation_type="scene",
            task_name=source_demo.task_name,
            error_name=source_demo.error_name,
            degree=source_demo.degree,
            target_scene_id=scene_id,
            success=success,
            num_steps=len(all_actions),
            actions=all_actions,
            states=np.array(states_list) if states_list else None,
            eef_positions=np.array(eef_list) if eef_list else None,
            gripper_states=np.array(grip_list) if grip_list else None,
            subtasks=source_demo.subtasks,
            metadata={
                'target_object': target_object,
                'pos_noise_std': self.pos_noise_std,
                'warp_method': 'per_subtask_closed_loop',
            },
        )

    def _warp_actions(
        self,
        source_demo: RecoveryDemo,
        src_obj_pose: np.ndarray,
        target_obj_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Warp recovery demo actions using object-centric transform.

        D0: single-pass transform over entire trajectory.
        D1: per-subtask transform, zeroing delta for no-warp subtasks.

        Applies both position and rotation corrections when EEF orientation
        data is available; falls back to position-only for older demos.
        """
        from error_benchmark.framework.utils.pose_transforms import (
            transform_source_data_segment_using_object_pose,
            unmake_pose,
            _mat2quat,
            quat2axisangle,
        )

        actions = source_demo.actions.copy()
        eef_positions = source_demo.eef_positions

        if eef_positions is None or len(eef_positions) < 2:
            return self._warp_actions_pos_delta(actions, src_obj_pose, target_obj_pose)

        # Build source EEF poses as 4x4 matrices
        src_eef_poses = np.tile(np.eye(4), (len(eef_positions), 1, 1))
        src_eef_poses[:, :3, 3] = eef_positions

        # Fill in actual EEF rotations if available
        has_orientations = (source_demo.eef_orientations is not None
                           and len(source_demo.eef_orientations) >= len(eef_positions))
        if has_orientations:
            for i in range(len(eef_positions)):
                quat = source_demo.eef_orientations[i]
                src_eef_poses[i, :3, :3] = self._quat_to_rot(quat)

        # Transform EEF trajectory
        transformed_eef = transform_source_data_segment_using_object_pose(
            obj_pose=target_obj_pose,
            src_eef_poses=src_eef_poses,
            src_obj_pose=src_obj_pose,
        )

        # Compute per-step position correction
        src_pos, _ = unmake_pose(src_eef_poses)
        new_pos, _ = unmake_pose(transformed_eef)
        pos_delta = new_pos - src_pos

        # Compute per-step rotation correction (axis-angle delta for OSC actions[3:6])
        rot_delta = None
        action_dim = actions.shape[1]
        if has_orientations and action_dim > 3:
            T_rot = min(len(actions), len(src_eef_poses))
            rot_delta = np.zeros((T_rot, 3))
            for t in range(T_rot):
                src_rot = src_eef_poses[t, :3, :3]
                new_rot = transformed_eef[t, :3, :3]
                # delta_rot = new_rot @ src_rot^T
                delta_rot = new_rot @ src_rot.T
                # Convert rotation matrix to axis-angle
                delta_quat = _mat2quat(delta_rot)  # (x, y, z, w)
                axis, angle = quat2axisangle(delta_quat)
                if abs(angle) > 1e-4:
                    rot_delta[t] = axis * angle

        # For D1: zero out delta for no-warp subtasks (retract, correct_position)
        if source_demo.degree == "D1" and source_demo.subtasks:
            for st in source_demo.subtasks:
                if st.label in self.NO_WARP_SUBTASKS:
                    start = min(st.start_step, len(pos_delta))
                    end = min(st.end_step, len(pos_delta))
                    pos_delta[start:end] = 0.0
                    if rot_delta is not None:
                        rot_delta[start:min(st.end_step, len(rot_delta))] = 0.0

        # Apply position correction to actions
        T = min(len(actions), len(pos_delta))
        actions[:T, :3] += pos_delta[:T]

        # Apply rotation correction to actions (OSC rotation dims 3:6)
        if rot_delta is not None:
            T_r = min(len(actions), len(rot_delta))
            actions[:T_r, 3:6] += rot_delta[:T_r] * 0.3  # scale factor to avoid over-correction

        return actions

    def _warp_actions_pos_delta(
        self,
        actions: np.ndarray,
        src_obj_pose: np.ndarray,
        target_obj_pose: np.ndarray,
    ) -> np.ndarray:
        """Fallback: simple position delta distribution (like original code)."""
        pos_delta = target_obj_pose[:3, 3] - src_obj_pose[:3, 3]
        if np.linalg.norm(pos_delta) < 1e-4:
            return actions

        approach_end = len(actions) // 2
        if approach_end > 0:
            per_step = pos_delta / approach_end
            actions[:approach_end, :3] += per_step

        return actions

    @staticmethod
    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        """Convert (w, x, y, z) quaternion to 3x3 rotation matrix.
        Note: robosuite uses (w,x,y,z) while MimicGen uses (x,y,z,w)."""
        from error_benchmark.framework.utils.pose_transforms import _quat2mat
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        return _quat2mat(xyzw)

    @staticmethod
    def _make_pose_4x4(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Build a 4x4 homogeneous matrix from pos (3,) and rot (3,3)."""
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return pose


def load_recovery_demos(demos_dir: Path, subtype_id: str) -> List[RecoveryDemo]:
    """Load collected recovery demos for a subtype."""
    demos = []
    subtype_dir = demos_dir / subtype_id

    manifest_path = demos_dir / "manifest.json"
    if not manifest_path.exists():
        return demos

    with open(manifest_path) as f:
        manifest = json.load(f)

    for demo_info in manifest.get('demos', []):
        if demo_info.get('subtype_id') != subtype_id:
            continue

        demo = RecoveryDemo.from_dict(demo_info)

        # Load arrays — prefer local path relative to manifest dir
        local_npz = subtype_dir / f"{demo.demo_id}.npz"
        stored_npz = demo_info.get('npz_path', '')
        if local_npz.exists():
            npz_path = str(local_npz)
        elif stored_npz and Path(stored_npz).exists():
            npz_path = stored_npz
        else:
            npz_path = str(local_npz)

        if Path(npz_path).exists():
            data = np.load(npz_path, allow_pickle=True)
            demo.actions = data.get('actions')
            demo.states = data.get('states')
            demo.eef_positions = data.get('eef_positions')
            demo.eef_orientations = data.get('eef_orientations')
            demo.target_poses = data.get('target_poses')
            demo.gripper_states = data.get('gripper_states')

            # Load object positions and orientations
            obj_positions = {}
            obj_orientations = {}
            for key in data.files:
                if key.startswith('obj_') and key.endswith('_quat'):
                    obj_name = key[4:-5]  # strip 'obj_' prefix and '_quat' suffix
                    obj_orientations[obj_name] = data[key]
                elif key.startswith('obj_'):
                    obj_name = key[4:]
                    obj_positions[obj_name] = data[key]
            if obj_positions:
                demo.object_positions = obj_positions
            if obj_orientations:
                demo.object_orientations = obj_orientations

        demos.append(demo)

    return demos


# load_error_scenes imported from script_utils


def save_augmented_demo(aug: AugmentedRecovery, output_dir: Path) -> str:
    """Save augmented demo NPZ and return path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{aug.augmented_id}.npz"

    save_dict = {}
    if aug.actions is not None:
        save_dict['actions'] = aug.actions
    if aug.states is not None:
        save_dict['states'] = aug.states
    if aug.eef_positions is not None:
        save_dict['eef_positions'] = aug.eef_positions
    if aug.target_poses is not None:
        save_dict['target_poses'] = aug.target_poses
    if aug.gripper_states is not None:
        save_dict['gripper_states'] = aug.gripper_states

    np.savez_compressed(npz_path, **save_dict)
    return str(npz_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Recovery data augmentation")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subtype", type=str, default=None,
                        help="Specific subtype to augment")
    parser.add_argument("--target_per_subtype", type=int, default=100,
                        help="Target augmented demos per subtype")
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/recovery_collection.yaml")
    parser.add_argument("--demos_dir", type=str, default=None)
    parser.add_argument("--scenes_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip simulation validation of augmented demos")
    parser.add_argument("--render", action="store_true",
                        help="Record MP4 video for each augmentation attempt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        recovery_config = yaml.safe_load(f)

    task_info = load_task_registry(args.task)
    dataset_path = task_info['dataset_path']

    task_config_path = task_info.get('task_config', '')
    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)

    aug_config = recovery_config.get('augmentation', {})
    if args.no_validate:
        aug_config.setdefault('scene_augmentation', {})['success_validation'] = False

    # Paths
    demos_base = args.demos_dir or recovery_config['paths']['recovery_demos_dir']
    demos_dir = PROJECT_ROOT / demos_base / args.task

    scenes_base = args.scenes_dir or f"error_benchmark/outputs/v5_training/{args.task}/scenes"
    scenes_dir = PROJECT_ROOT / scenes_base

    output_base = args.output_dir or recovery_config['paths']['augmented_demos_dir']
    output_dir = PROJECT_ROOT / output_base / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine subtypes
    allocations = recovery_config.get('demo_allocations', {}).get(args.task, {})
    if args.subtype:
        subtypes = [args.subtype]
    else:
        subtypes = list(allocations.keys())

    # Create environment (needed for simulation validation)
    env = create_env(task_config, dataset_path,
                     enable_camera=args.render)
    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)

    rng = np.random.RandomState(args.seed)
    augmenter = RecoveryAugmenter(env_wrapper, task_config, aug_config, rng)

    logger.info(f"=== Recovery Data Augmentation ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Target per subtype: {args.target_per_subtype}")
    logger.info(f"Subtypes: {len(subtypes)}")

    total_augmented = 0
    aug_manifest = {'version': 'v1.0', 'task': args.task, 'subtypes': {}}

    for subtype_id in subtypes:
        logger.info(f"\n--- Augmenting {subtype_id} ---")

        # Load source demos
        source_demos = load_recovery_demos(demos_dir, subtype_id)
        successful_demos = [d for d in source_demos if d.success]

        if not successful_demos:
            logger.warning(f"  No successful demos found for {subtype_id}")
            continue

        logger.info(f"  Source demos: {len(successful_demos)} successful / {len(source_demos)} total")

        # Load target error scenes
        target_scenes = load_error_scenes(scenes_dir, subtype_id)
        logger.info(f"  Target scenes: {len(target_scenes)}")

        subtype_output = output_dir / subtype_id
        augmented_count = 0
        subtype_augs = []

        # 3A: Scene augmentation
        if aug_config.get('scene_augmentation', {}).get('enabled', True):
            remaining = args.target_per_subtype - augmented_count
            _video_dir = (PROJECT_ROOT / "all_videos" / "mimicgen") if args.render else None
            augs = augmenter.augment_scene(
                successful_demos, target_scenes, max_augments=remaining,
                video_dir=_video_dir)

            for aug in augs:
                npz_path = save_augmented_demo(aug, subtype_output)
                subtype_augs.append({**aug.to_dict(), 'npz_path': npz_path})
                augmented_count += 1

        # 3B: Cross-degree augmentation
        if aug_config.get('cross_degree', {}).get('enabled', True):
            # Find D0 demos to generate D1 variants
            d0_demos = [d for d in successful_demos if d.degree == 'D0']

            for demo in d0_demos:
                if augmented_count >= args.target_per_subtype:
                    break

                for target_deg in ['D1']:
                    target_sub = f"{demo.error_name}_{target_deg}"
                    if target_sub == subtype_id:
                        continue  # Don't augment to same subtype
                    # Only create if target_sub is a valid subtype
                    from error_benchmark.framework.error_taxonomy_v5 import is_valid_subtype
                    if not is_valid_subtype(demo.error_name, target_deg):
                        continue

                    aug = augmenter.augment_cross_degree(demo, target_deg)
                    if aug is not None:
                        npz_path = save_augmented_demo(aug, output_dir / target_sub)
                        subtype_augs.append({**aug.to_dict(), 'npz_path': npz_path})

        # 3C: Cross-subtype augmentation (within same RBG)
        if aug_config.get('cross_subtype', {}).get('enabled', True):
            source_rbg = SUBTYPE_TO_RBG.get(subtype_id, "")

            # Find other subtypes in same RBG that need augmentation
            for other_sub in subtypes:
                if other_sub == subtype_id:
                    continue
                other_rbg = SUBTYPE_TO_RBG.get(other_sub, "")
                if other_rbg != source_rbg:
                    continue

                other_scenes = load_error_scenes(scenes_dir, other_sub)
                if not other_scenes:
                    continue

                for demo in successful_demos[:2]:  # Use top 2 demos per cross-subtype
                    augs = augmenter.augment_cross_subtype(
                        demo, other_sub, other_scenes)
                    for aug in augs:
                        npz_path = save_augmented_demo(aug, output_dir / other_sub)
                        subtype_augs.append({**aug.to_dict(), 'npz_path': npz_path})

        aug_manifest['subtypes'][subtype_id] = {
            'source_demos': len(successful_demos),
            'augmented': augmented_count,
            'entries': subtype_augs,
        }
        total_augmented += augmented_count
        logger.info(f"  Augmented: {augmented_count}")

    # Save manifest
    manifest_path = output_dir / "augmentation_manifest.json"
    aug_manifest['total_augmented'] = total_augmented
    aug_manifest['created'] = datetime.now().isoformat()
    with open(manifest_path, 'w') as f:
        json.dump(aug_manifest, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\n=== Augmentation Summary ===")
    logger.info(f"Total augmented: {total_augmented}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
