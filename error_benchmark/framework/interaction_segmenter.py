#!/usr/bin/env python
"""
Interaction Segmenter - v5.0

Segments a clean trajectory by object interaction, producing a list of
InteractionSegment. Each segment has a definite target_object and phase,
eliminating ambiguity in multi-object tasks.

Segmentation logic:
    1. For each frame, find the closest object to the EEF.
    2. Detect grasping via EEF-object co-location + object Z following EEF Z.
    3. Cut a new segment when the target object changes or gripper state changes.
    4. Label each segment with a phase (reach / grasp / lift / transport / place).
"""

import logging
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper

from error_benchmark.framework.core import CleanTrajectory, InteractionSegment

logger = logging.getLogger(__name__)

# Thresholds
PROXIMITY_THRESHOLD = 0.08      # m — EEF within this distance counts as "near" object
GRASP_COLOCATION_THRESHOLD = 0.06  # m — EEF-object distance for grasp detection
CO_MOTION_WINDOW = 3            # frames — look-back for co-motion detection
CO_MOTION_THRESHOLD = 0.8       # correlation — EEF-object Z velocity correlation


class InteractionSegmenter:
    """
    Segments a trajectory into InteractionSegments based on object interaction.

    Usage:
        segmenter = InteractionSegmenter(task_config)
        segments = segmenter.segment(trajectory, env)
    """

    def __init__(self, task_config: dict):
        self.task_config = task_config
        thresholds = task_config.get('thresholds', {})
        self.proximity_threshold = thresholds.get('reach', PROXIMITY_THRESHOLD)
        self.grasp_closed_threshold = thresholds.get('grasp_closed', 0.05)
        self.lift_height = thresholds.get('lift_height', 0.85)

    def segment(
        self,
        trajectory: CleanTrajectory,
        env: 'EnvWrapper',
    ) -> List[InteractionSegment]:
        """
        Segment a trajectory by replaying it and tracking object interactions.

        Args:
            trajectory: CleanTrajectory with states and actions
            env: EnvWrapper for replaying and querying state

        Returns:
            List of InteractionSegment covering the full trajectory
        """
        if trajectory.states is None or trajectory.actions is None:
            logger.warning(f"Cannot segment {trajectory.trajectory_id}: missing states/actions")
            return []

        all_objects = env.get_all_object_names()
        if not all_objects:
            logger.warning(f"No objects in env for {trajectory.trajectory_id}")
            return []

        # Target candidates: graspable objects only (skip fixtures like 'base')
        target_candidates = env._get_graspable_objects()

        # Replay trajectory and collect per-frame data
        # Collect all object poses but find closest only among target candidates
        frame_data = self._collect_frame_data(trajectory, env, all_objects, target_candidates)

        # Build segments from frame data
        segments = self._build_segments(frame_data, all_objects, target_candidates, trajectory.num_steps)

        # Label phases
        segments = self._label_phases(segments, frame_data)

        logger.info(
            f"Segmented {trajectory.trajectory_id}: "
            f"{len(segments)} segments — "
            f"{[(s.target_object, s.phase, s.start_step, s.end_step) for s in segments]}"
        )

        return segments

    def _collect_frame_data(
        self,
        trajectory: CleanTrajectory,
        env: 'EnvWrapper',
        all_objects: List[str],
        target_candidates: List[str],
    ) -> List[Dict]:
        """Replay trajectory and collect per-frame EEF/object state."""
        env.set_sim_state_flat(trajectory.states[0])
        env.forward()

        frame_data = []
        # Track unique pose-read failures so we warn once per (object, error_type) pair
        # instead of flooding the log with one warning per frame.
        pose_failures: Dict[str, str] = {}
        for step in range(trajectory.num_steps + 1):
            eef_pos = env.get_eef_pos()
            gripper_norm = env.get_gripper_closed_norm()

            obj_poses = {}
            for name in all_objects:
                try:
                    pos, quat = env.get_object_pose(name)
                    obj_poses[name] = pos.copy()
                except (KeyError, AttributeError, ValueError) as e:
                    pose_failures.setdefault(name, f"{type(e).__name__}: {e}")

            # Find closest object among target candidates only (skip fixtures)
            closest_obj = None
            closest_dist = float('inf')
            for name in target_candidates:
                if name in obj_poses:
                    dist = np.linalg.norm(eef_pos - obj_poses[name])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_obj = name

            frame_data.append({
                'eef_pos': eef_pos.copy(),
                'gripper_norm': gripper_norm,
                'obj_poses': obj_poses,
                'closest_obj': closest_obj,
                'closest_dist': closest_dist,
            })

            # Execute action to advance (except after last step)
            if step < trajectory.num_steps:
                env.step(trajectory.actions[step])

        if pose_failures:
            logger.warning(
                "interaction_segmenter: dropped pose data for %d object(s) in trajectory %s: %s",
                len(pose_failures), trajectory.trajectory_id, pose_failures,
            )

        return frame_data

    def _detect_grasping(self, frame_data: List[Dict], frame_idx: int, obj_name: str) -> bool:
        """
        Detect if the robot is grasping an object at a given frame.

        Uses co-location (EEF near object) + co-motion (object Z follows EEF Z).
        """
        fd = frame_data[frame_idx]

        # Check co-location
        if obj_name not in fd['obj_poses']:
            return False
        dist = np.linalg.norm(fd['eef_pos'] - fd['obj_poses'][obj_name])
        if dist > GRASP_COLOCATION_THRESHOLD:
            return False

        # Check gripper is at least partially closed
        if fd['gripper_norm'] < self.grasp_closed_threshold:
            return False

        # Check co-motion: object Z should track EEF Z over recent frames
        if frame_idx < CO_MOTION_WINDOW:
            # Not enough history; rely on co-location + gripper
            return True

        eef_dz = []
        obj_dz = []
        for i in range(max(0, frame_idx - CO_MOTION_WINDOW), frame_idx):
            eef_dz.append(frame_data[i + 1]['eef_pos'][2] - frame_data[i]['eef_pos'][2])
            if obj_name in frame_data[i + 1]['obj_poses'] and obj_name in frame_data[i]['obj_poses']:
                obj_dz.append(
                    frame_data[i + 1]['obj_poses'][obj_name][2] -
                    frame_data[i]['obj_poses'][obj_name][2]
                )
            else:
                obj_dz.append(0.0)

        eef_dz = np.array(eef_dz)
        obj_dz = np.array(obj_dz)

        # If EEF barely moved vertically, co-location + gripper is enough
        if np.std(eef_dz) < 1e-4:
            return True

        # Correlation check
        if np.std(obj_dz) < 1e-6:
            # Object not moving vertically while EEF is → probably not grasped
            return np.std(eef_dz) < 1e-3

        corr = np.corrcoef(eef_dz, obj_dz)[0, 1]
        return corr > CO_MOTION_THRESHOLD

    def _build_segments(
        self,
        frame_data: List[Dict],
        all_objects: List[str],
        target_candidates: List[str],
        num_steps: int,
    ) -> List[InteractionSegment]:
        """Build segments by detecting target object and grasping state changes."""
        if not frame_data:
            return []

        segments = []
        # Initialize with first frame (fallback to first graspable candidate, not fixture)
        current_target = frame_data[0]['closest_obj'] or target_candidates[0]
        current_grasping = self._detect_grasping(frame_data, 0, current_target)
        segment_start = 0

        for frame_idx in range(1, len(frame_data)):
            fd = frame_data[frame_idx]

            # Determine target at this frame
            # If grasping, keep tracking the same object (don't switch mid-grasp)
            if current_grasping:
                new_target = current_target
                new_grasping = self._detect_grasping(frame_data, frame_idx, current_target)
            else:
                # Not grasping: target is the closest object within proximity
                if fd['closest_dist'] < self.proximity_threshold:
                    new_target = fd['closest_obj']
                else:
                    new_target = current_target  # Keep previous target if nothing close
                new_grasping = self._detect_grasping(frame_data, frame_idx, new_target)

            # Detect segment boundary: target change or grasp state change
            target_changed = new_target != current_target
            grasp_changed = new_grasping != current_grasping

            if target_changed or grasp_changed:
                # Close current segment
                other = [o for o in all_objects if o != current_target]
                segments.append(InteractionSegment(
                    target_object=current_target,
                    phase='',  # Will be labeled later
                    start_step=segment_start,
                    end_step=frame_idx,
                    gripper_grasping=current_grasping,
                    other_objects=other,
                ))
                segment_start = frame_idx
                current_target = new_target
                current_grasping = new_grasping

        # Close final segment
        other = [o for o in all_objects if o != current_target]
        segments.append(InteractionSegment(
            target_object=current_target,
            phase='',
            start_step=segment_start,
            end_step=len(frame_data),  # num_steps + 1 frames → end at num_steps + 1
            gripper_grasping=current_grasping,
            other_objects=other,
        ))

        # Merge very short segments (< 3 frames) into neighbors
        segments = self._merge_short_segments(segments, min_length=3)

        return segments

    def _merge_short_segments(
        self,
        segments: List[InteractionSegment],
        min_length: int = 3,
    ) -> List[InteractionSegment]:
        """Merge segments shorter than min_length into their neighbors."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.end_step - seg.start_step < min_length:
                # Merge into previous segment
                merged[-1].end_step = seg.end_step
            else:
                merged.append(seg)

        return merged

    # Number of consecutive frames above lift_height before 'lift' → 'transport'
    LIFT_TO_TRANSPORT_FRAMES = 10

    def _label_phases(
        self,
        segments: List[InteractionSegment],
        frame_data: List[Dict],
    ) -> List[InteractionSegment]:
        """
        Label segments with task phases, splitting at phase boundaries.

        Instead of assigning one phase per segment (which misses transitions
        in long segments), this walks frame-by-frame within each segment and
        splits at phase transitions:
            - Grasping segments: grasp → lift → transport
            - Non-grasping approach segments: pre_reach → reach → pre_grasp
            - Non-grasping post-grasp segments: place
        """
        grasp_eef_dist = self.task_config.get('thresholds', {}).get(
            'grasp_eef_dist', 0.06)
        grasped_objects: set = set()  # track which objects have been grasped
        result = []

        for seg in segments:
            if seg.gripper_grasping:
                grasped_objects.add(seg.target_object)
                sub_segs = self._split_grasping_segment(seg, frame_data)
                result.extend(sub_segs)
            else:
                had_grasped = seg.target_object in grasped_objects
                if had_grasped:
                    seg.phase = 'place'
                    result.append(seg)
                else:
                    sub_segs = self._split_approach_segment(
                        seg, frame_data, grasp_eef_dist)
                    result.extend(sub_segs)

        # Merge very short sub-segments to reduce noise
        result = self._merge_short_segments(result, min_length=2)
        return result

    def _split_grasping_segment(
        self,
        seg: InteractionSegment,
        frame_data: List[Dict],
    ) -> List[InteractionSegment]:
        """Split a grasping segment into grasp / lift / transport sub-segments."""
        sub_segments: List[InteractionSegment] = []
        current_phase = None
        phase_start = seg.start_step
        frames_above = 0

        for fi in range(seg.start_step, min(seg.end_step, len(frame_data))):
            obj_pos = frame_data[fi]['obj_poses'].get(seg.target_object)

            if obj_pos is None or obj_pos[2] <= self.lift_height:
                phase = 'grasp'
                frames_above = 0
            else:
                frames_above += 1
                phase = 'lift' if frames_above <= self.LIFT_TO_TRANSPORT_FRAMES else 'transport'

            if current_phase is not None and phase != current_phase:
                sub_segments.append(self._make_sub_segment(
                    seg, current_phase, phase_start, fi, grasping=True))
                phase_start = fi
            current_phase = phase

        # Final sub-segment
        if current_phase is not None:
            sub_segments.append(self._make_sub_segment(
                seg, current_phase, phase_start, seg.end_step, grasping=True))

        return sub_segments if sub_segments else [seg]

    def _split_approach_segment(
        self,
        seg: InteractionSegment,
        frame_data: List[Dict],
        grasp_eef_dist: float,
    ) -> List[InteractionSegment]:
        """Split an approach segment into pre_reach / reach / pre_grasp."""
        sub_segments: List[InteractionSegment] = []
        current_phase = None
        phase_start = seg.start_step

        for fi in range(seg.start_step, min(seg.end_step, len(frame_data))):
            fd = frame_data[fi]
            obj_pos = fd['obj_poses'].get(seg.target_object)

            if obj_pos is None:
                phase = 'pre_reach'
            else:
                dist = np.linalg.norm(fd['eef_pos'] - obj_pos)
                if dist < grasp_eef_dist:
                    phase = 'pre_grasp'
                elif dist < self.proximity_threshold:
                    phase = 'reach'
                else:
                    phase = 'pre_reach'

            if current_phase is not None and phase != current_phase:
                sub_segments.append(self._make_sub_segment(
                    seg, current_phase, phase_start, fi, grasping=False))
                phase_start = fi
            current_phase = phase

        if current_phase is not None:
            sub_segments.append(self._make_sub_segment(
                seg, current_phase, phase_start, seg.end_step, grasping=False))

        return sub_segments if sub_segments else [seg]

    @staticmethod
    def _make_sub_segment(
        parent: InteractionSegment,
        phase: str,
        start: int,
        end: int,
        grasping: bool,
    ) -> InteractionSegment:
        """Create a sub-segment inheriting target_object and other_objects."""
        return InteractionSegment(
            target_object=parent.target_object,
            phase=phase,
            start_step=start,
            end_step=end,
            gripper_grasping=grasping,
            other_objects=parent.other_objects,
        )
