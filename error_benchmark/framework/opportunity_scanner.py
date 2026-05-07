#!/usr/bin/env python
"""
Opportunity Scanner - v5.0

Offline scanner that replays each clean trajectory and calls every skill's
can_inject() at every frame to build an InjectionOpportunityMap.

This is Step 0b of the v5 pipeline:
    Step 0: Collect clean trajectories
    Step 0b: Scan opportunities (this module)  <--
    Step 0c: Schedule injections (quota)
    Step 1: Execute injections
"""

import logging
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper

from error_benchmark.framework.core import CleanTrajectory, InjectionOpportunity, map_phase_to_group
from error_benchmark.framework.error_skills import BaseErrorSkill

logger = logging.getLogger(__name__)


class OpportunityScanner:
    """
    Scans clean trajectories to find all injection opportunities.

    For each (trajectory, frame, skill, degree) combination, records
    whether injection is possible. The result is an InjectionOpportunityMap
    used by the QuotaScheduler.

    Usage:
        scanner = OpportunityScanner(env, skills)
        opportunities = scanner.scan_trajectory(trajectory)
        scanner.save(opportunities, output_path)
    """

    # Number of consecutive frames above lift_height before phase transitions
    # from 'lift' to 'transport'
    LIFT_TO_TRANSPORT_FRAMES = 10

    def __init__(
        self,
        env: 'EnvWrapper',
        skills: List[BaseErrorSkill],
        min_frame_gap: int = 10,
        show_progress: bool = True,
    ):
        """
        Args:
            env: EnvWrapper instance
            skills: List of enabled error skills
            min_frame_gap: Minimum frames between opportunities for same skill
            show_progress: Show tqdm progress bars
        """
        self.env = env
        self.skills = skills
        self.min_frame_gap = min_frame_gap
        self.show_progress = show_progress
        self.logger = logging.getLogger(__name__)

        # Per-frame phase detection thresholds (from task config)
        thresholds = env._task_config.get('thresholds', {})
        self._lift_height = thresholds.get('lift_height', 0.85)
        self._reach_dist = thresholds.get('reach', 0.08)
        self._grasp_dist = thresholds.get('grasp_eef_dist', 0.06)

    def _auto_segment(self, trajectory: CleanTrajectory) -> None:
        """
        Compute interaction segments for a trajectory that lacks them.

        Many existing clean trajectories were collected before the
        InteractionSegmenter was added.  This method runs segmentation
        on-the-fly so the per-frame phase detection in _detect_frame_phase
        can work correctly.
        """
        from error_benchmark.framework.interaction_segmenter import InteractionSegmenter
        segmenter = InteractionSegmenter(self.env._task_config)
        trajectory.interaction_segments = segmenter.segment(trajectory, self.env)
        self.logger.info(
            f"Auto-segmented {trajectory.trajectory_id}: "
            f"{len(trajectory.interaction_segments)} segments"
        )

    def scan_trajectory(
        self,
        trajectory: CleanTrajectory,
    ) -> List[InjectionOpportunity]:
        """
        Scan a single trajectory for all injection opportunities.

        Replays the trajectory frame by frame, calling can_inject() for
        each skill at each frame.

        Args:
            trajectory: CleanTrajectory to scan

        Returns:
            List of InjectionOpportunity objects
        """
        if trajectory.states is None or trajectory.actions is None:
            self.logger.warning(
                f"Cannot scan {trajectory.trajectory_id}: missing states/actions"
            )
            return []

        opportunities = []
        # Track last injection frame per skill (for min_frame_gap)
        last_frame: Dict[str, int] = {s.name: -self.min_frame_gap for s in self.skills}

        # Per-frame phase tracking state (reset per trajectory)
        self._frames_above: Dict[str, int] = {}   # consecutive frames above lift_height per object
        self._grasped_objects: set = set()          # objects that have been grasped

        self.logger.info(
            f"Scanning trajectory {trajectory.trajectory_id} "
            f"({trajectory.num_steps} frames, {len(self.skills)} skills)"
        )

        # Auto-segment if trajectory has no interaction segments
        if trajectory.interaction_segments is None:
            self._auto_segment(trajectory)

        # Restore initial state (set_sim_state_flat already calls forward())
        self.env.set_sim_state_flat(trajectory.states[0])

        # Cache values that don't change per frame
        graspable_objects = self.env._get_graspable_objects()

        for frame_idx in tqdm(range(trajectory.num_steps),
                              desc=f"  frames({trajectory.trajectory_id})",
                              leave=False, disable=not self.show_progress):
            # Build frame state info (uses interaction segments if available)
            frame_state = self._extract_frame_state(frame_idx, trajectory)

            # Build trajectory context
            target_object = frame_state.get('target_object') or self.env.get_target_object()
            trajectory_context = {
                'frame_index': frame_idx,
                'total_frames': trajectory.num_steps,
                'task_phase': frame_state.get('task_phase', ''),
                'objects': frame_state.get('objects', {}),
                'eef_pos': frame_state.get('eef_pos', np.zeros(3)),
                'gripper_closed_norm': frame_state.get('gripper_closed_norm', 0.0),
                'target_object': target_object,
                'other_objects': frame_state.get('other_objects', []),
                'graspable_objects': graspable_objects,
            }

            # Add target_pos for drop skill near/far distinction
            if target_object:
                target_pos = self.env._get_target_pos(target_object)
                if target_pos is not None:
                    trajectory_context['target_pos'] = target_pos

            # Add phase history (pass reference + index to avoid O(n²) slicing)
            if trajectory.phase_labels is not None:
                trajectory_context['phase_labels_ref'] = trajectory.phase_labels
                trajectory_context['phase_labels_up_to'] = frame_idx

            # Check each skill
            for skill in self.skills:
                # Enforce minimum frame gap
                if frame_idx - last_frame[skill.name] < self.min_frame_gap:
                    continue

                # Check if this frame is injectable
                injectable_degrees = skill.can_inject(frame_state, trajectory_context)

                for degree in injectable_degrees:
                    opp = InjectionOpportunity(
                        trajectory_id=trajectory.trajectory_id,
                        frame_index=frame_idx,
                        error_name=skill.name,
                        degree=degree,
                        task_phase=frame_state.get('task_phase', ''),
                        confidence=1.0,
                        metadata={
                            'target_object': target_object or '',
                            'phase_group': map_phase_to_group(
                                frame_state.get('task_phase', '')),
                        },
                    )
                    opportunities.append(opp)
                    last_frame[skill.name] = frame_idx

            # Execute action to advance to next frame
            if frame_idx < trajectory.num_steps:
                self.env.step(trajectory.actions[frame_idx])

        self.logger.info(
            f"Found {len(opportunities)} opportunities in "
            f"{trajectory.trajectory_id}"
        )

        # Log summary by skill
        from collections import Counter
        by_skill = Counter(opp.subtype_id() for opp in opportunities)
        for key, count in sorted(by_skill.items()):
            self.logger.debug(f"  {key}: {count} opportunities")

        return opportunities

    def scan_all(
        self,
        trajectories: List[CleanTrajectory],
    ) -> List[InjectionOpportunity]:
        """
        Scan all trajectories for injection opportunities.

        Args:
            trajectories: List of CleanTrajectory objects

        Returns:
            Combined list of all InjectionOpportunity objects
        """
        all_opportunities = []
        pbar = tqdm(trajectories, desc="Scanning trajectories",
                    disable=not self.show_progress)
        for traj in pbar:
            opps = self.scan_trajectory(traj)
            all_opportunities.extend(opps)
            pbar.set_postfix(opps=len(all_opportunities))

        self.logger.info(
            f"Total: {len(all_opportunities)} opportunities across "
            f"{len(trajectories)} trajectories"
        )
        return all_opportunities

    def _extract_frame_state(
        self,
        frame_idx: int,
        trajectory: CleanTrajectory,
    ) -> Dict:
        """
        Extract state info at the current frame from the environment.

        Uses interaction segments (if available) for accurate target_object
        and task_phase in multi-object tasks.

        Args:
            frame_idx: Current frame index
            trajectory: Source trajectory

        Returns:
            State info dict compatible with skill.can_inject()
        """
        state_info = {
            'step': frame_idx,
            'trajectory_id': trajectory.trajectory_id,
            'eef_pos': self.env.get_eef_pos(),
            'eef_quat': self.env.get_eef_quat(),
            'gripper_closed_norm': self.env.get_gripper_closed_norm(),
        }

        # Object states
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

        # Use interaction segment for target_object, then detect phase per-frame
        segment = trajectory.get_segment_at_frame(frame_idx)
        if segment is not None:
            state_info['target_object'] = segment.target_object
            state_info['other_objects'] = segment.other_objects
            state_info['gripper_grasping'] = segment.gripper_grasping
            # Per-frame phase detection instead of coarse segment.phase
            state_info['task_phase'] = self._detect_frame_phase(
                state_info, segment)
            # Track grasping history for place detection
            if segment.gripper_grasping:
                self._grasped_objects.add(segment.target_object)
        else:
            # Fallback: env-based phase detection
            # Determine target_object first so get_task_phase uses the correct object
            state_info['target_object'] = self.env.get_target_object()
            state_info['task_phase'] = self.env.get_task_phase(state_info)
            state_info['other_objects'] = [
                n for n in self.env.get_all_object_names()
                if n != state_info['target_object']
            ]
            # Note: do NOT override with trajectory.phase_labels here — they may
            # have been computed against the wrong target_object (e.g. fixture 'base').
            # Real-time detection with correct target_object is more accurate.

        return state_info

    def _detect_frame_phase(
        self,
        state_info: Dict,
        segment,
    ) -> str:
        """
        Detect the task phase for the current frame based on kinematics.

        Unlike the segment-level phase (which uses a midpoint heuristic and
        can miss transitions in long segments), this computes the phase
        per-frame using object height, EEF proximity, and gripper state.

        Phase vocabulary:
            pre_reach → reach → pre_grasp → grasp → lift → transport → place
        """
        target_obj = segment.target_object
        obj_info = state_info.get('objects', {}).get(target_obj)

        if segment.gripper_grasping:
            # Grasping: determine grasp / lift / transport
            if obj_info is None:
                return 'grasp'

            obj_z = obj_info['pos'][2]
            if obj_z <= self._lift_height:
                # Object below lift height — still in grasp phase
                self._frames_above.pop(target_obj, None)
                return 'grasp'
            else:
                # Object above lift height — count consecutive frames
                count = self._frames_above.get(target_obj, 0) + 1
                self._frames_above[target_obj] = count
                return 'lift' if count <= self.LIFT_TO_TRANSPORT_FRAMES else 'transport'
        else:
            # Not grasping
            # Check if this object was previously grasped → 'place'
            if target_obj in self._grasped_objects:
                return 'place'

            if obj_info is None:
                return 'pre_reach'

            eef_pos = state_info.get('eef_pos', np.zeros(3))
            obj_pos = obj_info['pos']
            dist = np.linalg.norm(eef_pos - obj_pos)

            if dist < self._grasp_dist:
                return 'pre_grasp'
            elif dist < self._reach_dist:
                return 'reach'
            else:
                return 'pre_reach'

    @staticmethod
    def save(
        opportunities: List[InjectionOpportunity],
        output_path: str,
    ):
        """
        Save opportunities to a JSONL file.

        Args:
            opportunities: List of InjectionOpportunity objects
            output_path: Output file path (.jsonl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for opp in opportunities:
                f.write(json.dumps(opp.to_dict()) + '\n')

        logger.info(f"Saved {len(opportunities)} opportunities to {output_path}")

    @staticmethod
    def load(input_path: str) -> List[InjectionOpportunity]:
        """
        Load opportunities from a JSONL file.

        Args:
            input_path: Input file path (.jsonl)

        Returns:
            List of InjectionOpportunity objects
        """
        opportunities = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    opportunities.append(InjectionOpportunity.from_dict(d))

        logger.info(f"Loaded {len(opportunities)} opportunities from {input_path}")
        return opportunities
