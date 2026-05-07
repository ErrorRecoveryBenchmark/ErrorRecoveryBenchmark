#!/usr/bin/env python
"""
Recovery Trajectory Segmenter

Segments recovery demonstrations into subtask sequences for MimicGen augmentation.
Each recovery trajectory is split into semantically meaningful subtasks based on
the Recovery Behavior Group (RBG) and kinematic/contact heuristics.

Usage:
    segmenter = RecoverySegmenter(task_config, segmentation_config)
    subtasks = segmenter.segment(recovery_demo)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, RecoverySubtask, SUBTYPE_TO_RBG,
    RBG_SUBTASK_SEQUENCES, RECOVERY_SUBTASKS,
)

logger = logging.getLogger(__name__)


class RecoverySegmenter:
    """
    Segments recovery trajectories into subtask sequences.

    The segmenter uses kinematic heuristics (EEF position, gripper state,
    object proximity) to identify transition points between recovery subtasks.
    """

    _SUSTAINED_GRASP_FRAMES = 10  # min frames to confirm a sustained grasp

    def __init__(self, task_config: dict, segmentation_config: Optional[dict] = None):
        """
        Args:
            task_config: Task configuration dict (thresholds, objects, etc.)
            segmentation_config: Segmentation-specific config (from recovery_collection.yaml)
        """
        self.task_config = task_config
        self.config = segmentation_config or {}

        # Extract thresholds
        thresholds = self.task_config.get('thresholds', {})
        self.retract_displacement = self.config.get('retract', {}).get(
            'min_eef_displacement', 0.05)
        self.nav_threshold = self.config.get('navigate_to_object', {}).get(
            'eef_to_obj_threshold', thresholds.get('reach', 0.06))
        re_grasp_cfg = self.config.get('re_grasp', {})
        if not re_grasp_cfg:
            re_grasp_cfg = self.config.get('re_acquire', {})
        re_transport_cfg = self.config.get('re_transport', {})
        if not re_transport_cfg:
            re_transport_cfg = self.config.get('re_deliver', {})

        self.grasp_threshold = re_grasp_cfg.get(
            'gripper_closed_threshold', thresholds.get('grasp_closed', 0.05))
        self.post_grasp_buffer = int(re_grasp_cfg.get(
            'post_grasp_buffer',
            self.config.get('re_acquire', {}).get('post_grasp_buffer', 15)))
        self.release_threshold = self.config.get('release', {}).get(
            'gripper_open_threshold', 0.02)
        self.pos_correction_threshold = self.config.get('correct_position', {}).get(
            'pos_error_threshold', 0.03)
        self.lift_height = thresholds.get('lift_height', 0.84)
        self.transport_threshold = re_transport_cfg.get(
            'min_xy_displacement', thresholds.get('transport', 0.05))

    def segment(self, demo: RecoveryDemo) -> List[RecoverySubtask]:
        """
        Segment a recovery demo into subtasks.

        Default: MimicGen-style 2-subtask segmentation (pre_grasp / post_grasp)
        based on the gripper close signal. This matches the original MimicGen
        subtask structure for simple tasks (coffee, stack, threading).

        Args:
            demo: RecoveryDemo with populated eef_positions, gripper_states,
                  object_positions arrays.

        Returns:
            List of RecoverySubtask with start/end step indices.
        """
        mode = self.config.get('segmentation_mode', 'mimicgen')

        if mode == 'mimicgen':
            subtasks = self._segment_mimicgen_style(demo)
        elif mode == 'rbg':
            subtasks = self._segment_by_rbg(demo)
        else:
            raise ValueError(f"Unknown segmentation_mode '{mode}'")

        subtasks = self._validate_and_fill(subtasks, demo.num_steps)
        logger.debug(
            "Segmented demo %s (mode=%s): %s",
            demo.demo_id, mode,
            [(s.label, s.start_step, s.end_step) for s in subtasks],
        )
        return subtasks

    def _segment_mimicgen_style(self, demo: RecoveryDemo) -> List[RecoverySubtask]:
        """MimicGen-style segmentation: split at the grasp boundary.

        Produces exactly 2 subtasks:
          - pre_grasp: everything before the gripper closes on the object
          - post_grasp: everything after (lift, transport, place)

        This matches the original MimicGen ``grasp`` subtask_term_signal.
        """
        T = demo.num_steps
        gripper = demo.gripper_states
        eef_pos = demo.eef_positions

        if gripper is None or T == 0:
            mid = max(1, T // 2)
            return [
                RecoverySubtask(label="pre_grasp", start_step=0, end_step=mid, duration=mid),
                RecoverySubtask(label="post_grasp", start_step=mid, end_step=T, duration=T - mid),
            ]

        grasp_step = self._find_gripper_close(gripper, 0, self.grasp_threshold)
        grasp_step = min(grasp_step + self.post_grasp_buffer, T)
        grasp_step = max(1, min(grasp_step, T - 1))

        boundaries = [
            ("pre_grasp", 0, grasp_step),
            ("post_grasp", grasp_step, T),
        ]
        return self._boundaries_to_subtasks(boundaries, eef_pos)

    def _segment_by_rbg(self, demo: RecoveryDemo) -> List[RecoverySubtask]:
        """RBG-based fine-grained segmentation (legacy / optional path)."""
        if demo.subtype_id == "stuck_no_progress_D1":
            return self._segment_stuck_no_progress_d1(demo)

        rbg = demo.rbg or SUBTYPE_TO_RBG.get(demo.subtype_id, "")
        if not rbg:
            logger.warning(f"Unknown RBG for subtype {demo.subtype_id}, using generic segmentation")
            return self._segment_generic(demo)

        expected_sequence = RBG_SUBTASK_SEQUENCES.get(rbg, [])
        if not expected_sequence:
            return self._segment_generic(demo)

        segmenter_map = {
            "RBG_A": self._segment_regrasp,
            "RBG_B": self._segment_retrieve,
            "RBG_C": self._segment_retract,
            "RBG_D": self._segment_redirect,
            "RBG_E": self._segment_realign,
        }

        segmenter_fn = segmenter_map.get(rbg, self._segment_generic)
        return segmenter_fn(demo, expected_sequence)

    def _segment_regrasp(self, demo: RecoveryDemo,
                         expected: List[str]) -> List[RecoverySubtask]:
        """RBG-A: retract → re_orient → re_grasp → re_lift → re_transport → re_place."""
        eef_pos = demo.eef_positions  # (T+1, 3)
        gripper = demo.gripper_states  # (T+1,)
        T = demo.num_steps

        if eef_pos is None or gripper is None:
            return self._segment_uniform(expected, T)

        retract_end = self._find_displacement_threshold(
            eef_pos, 0, self.retract_displacement)
        reorient_end = self._find_descent_start(eef_pos, retract_end, T)
        grasp_end = self._find_gripper_close(
            gripper, reorient_end, self.grasp_threshold)
        grasp_end = min(grasp_end + self.post_grasp_buffer, T)

        boundaries = [
            ("retract", 0, retract_end),
            ("re_orient", retract_end, reorient_end),
        ]
        boundaries.extend(
            self._segment_pick_and_deliver(
                eef_pos=eef_pos,
                pick_start=reorient_end,
                grasp_end=grasp_end,
                total_steps=T,
            )
        )
        subtasks = self._boundaries_to_subtasks(boundaries, eef_pos)
        return self._ensure_expected_sequence(subtasks, expected, T)

    def _segment_retrieve(self, demo: RecoveryDemo,
                          expected: List[str]) -> List[RecoverySubtask]:
        """RBG-B: navigate_to_object → re_grasp → re_lift → re_transport → re_place."""
        eef_pos = demo.eef_positions
        gripper = demo.gripper_states
        T = demo.num_steps

        if eef_pos is None or gripper is None:
            return self._segment_uniform(expected, T)

        navigate_end = self._find_descent_start(eef_pos, 0, T)
        grasp_end = self._find_gripper_close(
            gripper, navigate_end, self.grasp_threshold)
        grasp_end = min(grasp_end + self.post_grasp_buffer, T)

        boundaries = [("navigate_to_object", 0, navigate_end)]
        boundaries.extend(
            self._segment_pick_and_deliver(
                eef_pos=eef_pos,
                pick_start=navigate_end,
                grasp_end=grasp_end,
                total_steps=T,
            )
        )
        subtasks = self._boundaries_to_subtasks(boundaries, eef_pos)
        return self._ensure_expected_sequence(subtasks, expected, T)

    def _segment_retract(self, demo: RecoveryDemo,
                         expected: List[str]) -> List[RecoverySubtask]:
        """RBG-C: retract → navigate_to_object"""
        eef_pos = demo.eef_positions
        T = demo.num_steps

        if eef_pos is None:
            return self._segment_uniform(expected, T)

        boundaries = []
        current_step = 0

        # Phase 1: retract - back away
        retract_end = self._find_displacement_threshold(
            eef_pos, current_step, self.retract_displacement)
        boundaries.append(("retract", current_step, retract_end))
        current_step = retract_end

        # Phase 2: navigate_to_object - re-approach
        boundaries.append(("navigate_to_object", current_step, T))

        subtasks = self._boundaries_to_subtasks(boundaries, eef_pos)
        return self._ensure_expected_sequence(subtasks, expected, T)

    def _segment_redirect(self, demo: RecoveryDemo,
                          expected: List[str]) -> List[RecoverySubtask]:
        """RBG-D: release → navigate_to_object → re_grasp → re_lift → re_transport → re_place."""
        eef_pos = demo.eef_positions
        gripper = demo.gripper_states
        T = demo.num_steps

        if eef_pos is None or gripper is None:
            return self._segment_uniform(expected, T)

        release_end = self._find_gripper_open(gripper, 0, self.release_threshold)
        navigate_end = self._find_descent_start(eef_pos, release_end, T)
        grasp_end = self._find_gripper_close(
            gripper, navigate_end, self.grasp_threshold)
        grasp_end = min(grasp_end + self.post_grasp_buffer, T)

        boundaries = [
            ("release", 0, release_end),
            ("navigate_to_object", release_end, navigate_end),
        ]
        boundaries.extend(
            self._segment_pick_and_deliver(
                eef_pos=eef_pos,
                pick_start=navigate_end,
                grasp_end=grasp_end,
                total_steps=T,
            )
        )
        subtasks = self._boundaries_to_subtasks(boundaries, eef_pos)
        return self._ensure_expected_sequence(subtasks, expected, T)

    def _segment_realign(self, demo: RecoveryDemo,
                         expected: List[str]) -> List[RecoverySubtask]:
        """RBG-E: correct_position → resume_task"""
        eef_pos = demo.eef_positions
        T = demo.num_steps

        if eef_pos is None:
            return self._segment_uniform(expected, T)

        # Simple split: first half is correction, second half is resume
        # Better heuristic: find when EEF velocity drops (correction done)
        mid = self._find_velocity_minimum(eef_pos, 0, T)

        boundaries = [
            ("correct_position", 0, mid),
            ("resume_task", mid, T),
        ]
        subtasks = self._boundaries_to_subtasks(boundaries, eef_pos)
        return self._ensure_expected_sequence(subtasks, expected, T)

    def _segment_generic(self, demo: RecoveryDemo) -> List[RecoverySubtask]:
        """Fallback: uniform split into generic recovery phases."""
        T = demo.num_steps
        if T == 0:
            return []

        # Simple three-phase split: prepare → execute → complete
        phases = ["retract", "navigate_to_object", "resume_task"]
        return self._segment_uniform(phases, T)

    def _segment_stuck_no_progress_d1(self, demo: RecoveryDemo) -> List[RecoverySubtask]:
        """Special-case legacy D1 stall demos that use a 3-stage recovery sequence."""
        return self._segment_uniform(
            ["retract", "navigate_to_object", "resume_task"],
            demo.num_steps,
        )

    def _segment_pick_and_deliver(
        self,
        eef_pos: np.ndarray,
        pick_start: int,
        grasp_end: int,
        total_steps: int,
    ) -> List[Tuple[str, int, int]]:
        """Split the grasped-object recovery tail into pick, lift, transport, and place."""
        pick_start = self._ensure_step(pick_start, 0, total_steps)
        grasp_end = self._ensure_step(grasp_end, pick_start + 1, total_steps)

        transport_start = self._find_horizontal_displacement_threshold(
            eef_pos, grasp_end, self.transport_threshold
        )
        if transport_start <= grasp_end:
            transport_start = self._find_height_peak(eef_pos, grasp_end, total_steps)
        transport_start = self._ensure_step(transport_start, grasp_end + 1, total_steps)

        place_start = self._find_descent_start(eef_pos, transport_start, total_steps)
        if place_start <= transport_start:
            remaining = max(total_steps - transport_start, 1)
            place_start = transport_start + max(1, remaining // 2)
        place_start = self._ensure_step(place_start, transport_start + 1, total_steps)

        return [
            ("re_grasp", pick_start, grasp_end),
            ("re_lift", grasp_end, transport_start),
            ("re_transport", transport_start, place_start),
            ("re_place", place_start, total_steps),
        ]

    # ─── Heuristic helpers ───

    def _find_displacement_threshold(self, eef_pos: np.ndarray,
                                     start: int, threshold: float) -> int:
        """Find step where cumulative EEF displacement exceeds threshold."""
        T = len(eef_pos) - 1
        if start >= T:
            return T
        steps = np.linalg.norm(np.diff(eef_pos[start:], axis=0), axis=1)
        cumsum = np.cumsum(steps)
        hits = np.where(cumsum >= threshold)[0]
        return (start + hits[0] + 1) if len(hits) > 0 else T

    def _find_proximity_threshold(self, eef_pos: np.ndarray, obj_pos: np.ndarray,
                                  start: int, threshold: float) -> int:
        """Find step where EEF gets within threshold distance of object."""
        T = min(len(eef_pos), len(obj_pos)) - 1
        if start >= T:
            return T
        dists = np.linalg.norm(eef_pos[start:T] - obj_pos[start:T], axis=1)
        hits = np.where(dists < threshold)[0]
        return (start + hits[0] + 1) if len(hits) > 0 else T

    def _find_descent_start(self, eef_pos: np.ndarray,
                            start: int, end: Optional[int] = None,
                            eps: float = 1e-4,
                            min_run: int = 2) -> int:
        """Find the first sustained downward motion in EEF z."""
        T = len(eef_pos) - 1
        end = T if end is None else min(end, T)
        if start >= end:
            return end

        run = 0
        for t in range(start, end):
            if eef_pos[t + 1, 2] < eef_pos[t, 2] - eps:
                run += 1
                if run >= min_run:
                    return max(start, t - run + 2)
            else:
                run = 0
        return end

    def _find_horizontal_displacement_threshold(self, eef_pos: np.ndarray,
                                                start: int, threshold: float) -> int:
        """Find when xy motion exceeds threshold relative to the start pose."""
        T = len(eef_pos) - 1
        if start >= T:
            return T

        origin_xy = eef_pos[start, :2]
        for t in range(start, T):
            if np.linalg.norm(eef_pos[t, :2] - origin_xy) >= threshold:
                return t
        return T

    def _find_height_peak(self, eef_pos: np.ndarray,
                          start: int, end: int) -> int:
        """Find the timestep with the highest EEF z in [start, end]."""
        T = min(end, len(eef_pos) - 1)
        if start >= T:
            return T
        return start + int(np.argmax(eef_pos[start:T + 1, 2]))

    def _ensure_step(self, candidate: int, minimum: int, maximum: int) -> int:
        """Clamp a boundary while preserving an upper bound."""
        maximum = max(0, maximum)
        minimum = max(0, minimum)
        if maximum <= minimum:
            return maximum
        return max(minimum, min(candidate, maximum))

    def _ensure_expected_sequence(self, subtasks: List[RecoverySubtask],
                                  expected: List[str],
                                  total_steps: int) -> List[RecoverySubtask]:
        """Fallback to a uniform split when heuristics miss expected labels."""
        observed = [s.label for s in subtasks]
        if observed == expected or not expected:
            return subtasks
        if total_steps >= len(expected):
            return self._segment_uniform(expected, total_steps)
        return subtasks

    def _find_gripper_close(self, gripper: np.ndarray,
                            start: int, threshold: float) -> int:
        """Find step where gripper closes and STAYS closed (sustained grasp).

        Scans for the last 0→1 transition that is sustained for at least
        `_SUSTAINED_GRASP_FRAMES` consecutive frames, to skip failed grasp
        attempts where the gripper opens again immediately.
        """
        T = len(gripper)
        sustained_frames = self._SUSTAINED_GRASP_FRAMES

        # Find all 0→1 transitions
        last_sustained = None
        t = start
        while t < T:
            if gripper[t] > threshold:
                # Check if this close is sustained
                end = t
                while end < T and gripper[end] > threshold:
                    end += 1
                if end - t >= sustained_frames:
                    last_sustained = t + 1  # return 1-indexed end of transition
                t = end
            else:
                t += 1

        if last_sustained is not None:
            return last_sustained
        # Fallback: return first close if no sustained close found
        for t in range(start, T):
            if gripper[t] > threshold:
                return t + 1
        return T

    def _find_gripper_open(self, gripper: np.ndarray,
                           start: int, threshold: float) -> int:
        """Find step where gripper opens below threshold."""
        T = len(gripper)
        for t in range(start, T):
            if gripper[t] < threshold:
                return t + 1
        return T

    def _find_height_threshold(self, obj_pos: np.ndarray,
                               start: int, height: float) -> int:
        """Find step where object Z exceeds height."""
        T = len(obj_pos)
        for t in range(start, T):
            if obj_pos[t, 2] > height:
                return t + 1
        return T

    def _find_height_threshold_eef(self, eef_pos: np.ndarray,
                                   start: int, height: float) -> int:
        """Find step where EEF Z exceeds height (fallback when no object tracking)."""
        T = len(eef_pos)
        for t in range(start, T):
            if eef_pos[t, 2] > height:
                return t + 1
        return T

    def _find_velocity_minimum(self, eef_pos: np.ndarray,
                               start: int, end: int) -> int:
        """Find step with minimum EEF velocity (transition between correction and resume)."""
        if end - start < 3:
            return (start + end) // 2

        # Compute velocity magnitudes
        velocities = []
        for t in range(start, min(end, len(eef_pos) - 1)):
            v = np.linalg.norm(eef_pos[t + 1] - eef_pos[t])
            velocities.append(v)

        if not velocities:
            return (start + end) // 2

        # Smooth with window of 5 and find minimum in the middle third
        window = min(5, len(velocities))
        smoothed = np.convolve(velocities, np.ones(window) / window, mode='same')

        # Search in middle 60% of trajectory
        search_start = len(smoothed) // 5
        search_end = 4 * len(smoothed) // 5
        if search_start >= search_end:
            search_start = 0
            search_end = len(smoothed)

        min_idx = search_start + np.argmin(smoothed[search_start:search_end])
        return start + min_idx + 1

    def _get_primary_obj_positions(self, demo: RecoveryDemo) -> Optional[np.ndarray]:
        """Get position trajectory of the primary (first) object."""
        if demo.object_positions is None:
            return None
        for name, positions in demo.object_positions.items():
            return positions  # Return first object
        return None

    def _boundaries_to_subtasks(self, boundaries: List[Tuple[str, int, int]],
                                eef_pos: Optional[np.ndarray]) -> List[RecoverySubtask]:
        """Convert (label, start, end) tuples to RecoverySubtask objects."""
        subtasks = []
        for label, start, end in boundaries:
            if end <= start:
                continue  # Skip empty subtasks

            displacement = 0.0
            if eef_pos is not None and end < len(eef_pos):
                for t in range(start, end):
                    if t + 1 < len(eef_pos):
                        displacement += np.linalg.norm(eef_pos[t + 1] - eef_pos[t])

            subtasks.append(RecoverySubtask(
                label=label,
                start_step=start,
                end_step=end,
                duration=end - start,
                eef_displacement=float(displacement),
            ))
        return subtasks

    def _segment_uniform(self, labels: List[str], T: int) -> List[RecoverySubtask]:
        """Uniformly split T steps into len(labels) segments."""
        if not labels or T == 0:
            return []

        segment_len = max(1, T // len(labels))
        subtasks = []
        for i, label in enumerate(labels):
            start = i * segment_len
            end = (i + 1) * segment_len if i < len(labels) - 1 else T
            if start >= T:
                break
            subtasks.append(RecoverySubtask(
                label=label,
                start_step=start,
                end_step=min(end, T),
                duration=min(end, T) - start,
            ))
        return subtasks

    def _validate_and_fill(self, subtasks: List[RecoverySubtask],
                           total_steps: int) -> List[RecoverySubtask]:
        """Ensure subtasks cover [0, total_steps) without gaps or overlaps."""
        if not subtasks:
            return [RecoverySubtask(
                label="resume_task",
                start_step=0,
                end_step=total_steps,
                duration=total_steps,
            )]

        # Sort by start step
        subtasks.sort(key=lambda s: s.start_step)

        if subtasks[0].start_step > 0:
            subtasks[0].start_step = 0
            subtasks[0].duration = subtasks[0].end_step

        # Extend last subtask to cover total_steps
        if subtasks[-1].end_step < total_steps:
            subtasks[-1].end_step = total_steps
            subtasks[-1].duration = total_steps - subtasks[-1].start_step

        return subtasks
