"""Unit tests for per-frame phase detection in OpportunityScanner
and phase-boundary splitting in InteractionSegmenter."""

import numpy as np
import pytest

from error_benchmark.framework.core import InteractionSegment
from error_benchmark.framework.interaction_segmenter import InteractionSegmenter


# ─── InteractionSegmenter phase splitting ───


class TestSplitGraspingSegment:
    """Test _split_grasping_segment: grasp → lift → transport."""

    def _make_segmenter(self, lift_height=0.85):
        config = {
            'thresholds': {
                'reach': 0.08,
                'grasp_closed': 0.05,
                'lift_height': lift_height,
            }
        }
        return InteractionSegmenter(config)

    def _make_frame_data(self, z_values):
        """Create frame_data with object Z at given heights."""
        return [
            {
                'eef_pos': np.array([0.0, 0.0, z]),
                'gripper_norm': 0.5,
                'obj_poses': {'obj_a': np.array([0.0, 0.0, z])},
                'closest_obj': 'obj_a',
                'closest_dist': 0.0,
            }
            for z in z_values
        ]

    def _make_seg(self, start, end, target='obj_a'):
        return InteractionSegment(
            target_object=target,
            phase='',
            start_step=start,
            end_step=end,
            gripper_grasping=True,
            other_objects=[],
        )

    def test_all_below_lift_height(self):
        """Object stays below lift_height → entire segment is 'grasp'."""
        seg = InteractionSegmenter.__new__(InteractionSegmenter)
        seg.lift_height = 0.85
        segmenter = self._make_segmenter()
        parent = self._make_seg(0, 20)
        frame_data = self._make_frame_data([0.80] * 20)

        result = segmenter._split_grasping_segment(parent, frame_data)

        assert len(result) == 1
        assert result[0].phase == 'grasp'
        assert result[0].start_step == 0
        assert result[0].end_step == 20

    def test_grasp_then_lift_then_transport(self):
        """Object rises from table to above lift_height → grasp, lift, transport."""
        segmenter = self._make_segmenter(lift_height=0.85)
        parent = self._make_seg(0, 30)
        # 10 frames below, 10 frames above (lift), 10 more (transport)
        z_values = [0.80] * 10 + [0.90] * 20
        frame_data = self._make_frame_data(z_values)

        result = segmenter._split_grasping_segment(parent, frame_data)

        phases = [s.phase for s in result]
        assert 'grasp' in phases
        assert 'lift' in phases
        assert 'transport' in phases
        # Order should be grasp → lift → transport
        assert phases.index('grasp') < phases.index('lift')
        assert phases.index('lift') < phases.index('transport')

    def test_all_above_lift_height(self):
        """Object already above lift_height at start → lift then transport."""
        segmenter = self._make_segmenter(lift_height=0.85)
        parent = self._make_seg(0, 25)
        z_values = [0.90] * 25
        frame_data = self._make_frame_data(z_values)

        result = segmenter._split_grasping_segment(parent, frame_data)

        phases = [s.phase for s in result]
        assert 'lift' in phases
        assert 'transport' in phases
        assert 'grasp' not in phases

    def test_short_lift_no_transport(self):
        """Object above lift_height for < LIFT_TO_TRANSPORT_FRAMES → only lift."""
        segmenter = self._make_segmenter(lift_height=0.85)
        parent = self._make_seg(0, 15)
        z_values = [0.80] * 10 + [0.90] * 5  # only 5 frames above
        frame_data = self._make_frame_data(z_values)

        result = segmenter._split_grasping_segment(parent, frame_data)

        phases = [s.phase for s in result]
        assert 'grasp' in phases
        assert 'lift' in phases
        assert 'transport' not in phases

    def test_sub_segments_cover_full_range(self):
        """Sub-segments should cover the full parent range with no gaps."""
        segmenter = self._make_segmenter(lift_height=0.85)
        parent = self._make_seg(5, 35)
        z_values = [0.0] * 5 + [0.80] * 10 + [0.90] * 20
        frame_data = self._make_frame_data(z_values)

        result = segmenter._split_grasping_segment(parent, frame_data)

        assert result[0].start_step == 5
        assert result[-1].end_step == 35
        for i in range(1, len(result)):
            assert result[i].start_step == result[i - 1].end_step

    def test_inherits_parent_info(self):
        """Sub-segments inherit target_object and other_objects."""
        segmenter = self._make_segmenter()
        parent = InteractionSegment(
            target_object='piece_1',
            phase='',
            start_step=0,
            end_step=20,
            gripper_grasping=True,
            other_objects=['piece_2', 'base'],
        )
        frame_data = self._make_frame_data([0.80] * 20)
        # Rename object in frame_data
        for fd in frame_data:
            fd['obj_poses'] = {'piece_1': fd['obj_poses']['obj_a']}

        result = segmenter._split_grasping_segment(parent, frame_data)

        for seg in result:
            assert seg.target_object == 'piece_1'
            assert seg.other_objects == ['piece_2', 'base']
            assert seg.gripper_grasping is True


class TestSplitApproachSegment:
    """Test _split_approach_segment: pre_reach → reach → pre_grasp."""

    def _make_segmenter(self, reach=0.08, grasp_eef_dist=0.06):
        config = {
            'thresholds': {
                'reach': reach,
                'grasp_closed': 0.05,
                'lift_height': 0.85,
                'grasp_eef_dist': grasp_eef_dist,
            }
        }
        return InteractionSegmenter(config)

    def _make_frame_data_approach(self, distances):
        """Create frame_data where EEF approaches object at (0,0,0.8)."""
        obj_pos = np.array([0.0, 0.0, 0.80])
        return [
            {
                'eef_pos': np.array([d, 0.0, 0.80]),
                'gripper_norm': 0.0,
                'obj_poses': {'obj_a': obj_pos.copy()},
                'closest_obj': 'obj_a',
                'closest_dist': d,
            }
            for d in distances
        ]

    def test_far_then_near(self):
        """EEF approaches from far → pre_reach then reach."""
        segmenter = self._make_segmenter(reach=0.08, grasp_eef_dist=0.06)
        parent = InteractionSegment(
            target_object='obj_a', phase='', start_step=0,
            end_step=20, gripper_grasping=False, other_objects=[])
        # Approach: 0.15 → 0.05 (linear)
        distances = [0.15 - i * 0.005 for i in range(20)]
        frame_data = self._make_frame_data_approach(distances)

        result = segmenter._split_approach_segment(parent, frame_data, 0.06)

        phases = [s.phase for s in result]
        assert phases[0] == 'pre_reach'
        assert 'reach' in phases or 'pre_grasp' in phases

    def test_already_close(self):
        """EEF already close to object → pre_grasp."""
        segmenter = self._make_segmenter()
        parent = InteractionSegment(
            target_object='obj_a', phase='', start_step=0,
            end_step=10, gripper_grasping=False, other_objects=[])
        distances = [0.03] * 10
        frame_data = self._make_frame_data_approach(distances)

        result = segmenter._split_approach_segment(parent, frame_data, 0.06)

        assert all(s.phase == 'pre_grasp' for s in result)

    def test_full_approach_sequence(self):
        """EEF goes from far to very close → pre_reach, reach, pre_grasp."""
        segmenter = self._make_segmenter(reach=0.08, grasp_eef_dist=0.04)
        parent = InteractionSegment(
            target_object='obj_a', phase='', start_step=0,
            end_step=30, gripper_grasping=False, other_objects=[])
        # Far (0.15) → near reach (0.06) → very close (0.02)
        distances = [0.15] * 10 + [0.06] * 10 + [0.02] * 10
        frame_data = self._make_frame_data_approach(distances)

        result = segmenter._split_approach_segment(parent, frame_data, 0.04)

        phases = [s.phase for s in result]
        assert 'pre_reach' in phases
        assert 'reach' in phases
        assert 'pre_grasp' in phases


class TestLabelPhases:
    """Integration test for _label_phases with segment splitting."""

    def _make_segmenter(self):
        config = {
            'thresholds': {
                'reach': 0.08,
                'grasp_closed': 0.05,
                'lift_height': 0.85,
                'grasp_eef_dist': 0.06,
            }
        }
        return InteractionSegmenter(config)

    def test_coffee_like_trajectory(self):
        """Simulate coffee task: reach → grasp → lift → transport (→ insert).

        Previously this was labeled as one 'lift' segment, missing 'transport'.
        """
        segmenter = self._make_segmenter()
        obj_pos = np.array([0.0, 0.0, 0.80])

        # Build frame data for a coffee-like trajectory:
        # Frames 0-9: approaching (not grasping, far)
        # Frames 10-14: close to object (not grasping, near)
        # Frames 15-24: grasping, object on table (z < 0.85)
        # Frames 25-34: grasping, object rising (z crosses 0.85)
        # Frames 35-59: grasping, object above lift_height (transport)
        frame_data = []
        for i in range(60):
            if i < 10:
                eef = np.array([0.15 - i * 0.01, 0.0, 0.80])
                gn = 0.0
                oz = 0.80
                grasping = False
            elif i < 15:
                eef = np.array([0.05 - (i - 10) * 0.01, 0.0, 0.80])
                gn = 0.0
                oz = 0.80
                grasping = False
            elif i < 25:
                eef = np.array([0.0, 0.0, 0.80 + (i - 15) * 0.005])
                gn = 0.5
                oz = 0.80 + (i - 15) * 0.005
                grasping = True
            elif i < 35:
                eef = np.array([0.0, 0.0, 0.85 + (i - 25) * 0.005])
                gn = 0.5
                oz = 0.85 + (i - 25) * 0.005
                grasping = True
            else:
                eef = np.array([(i - 35) * 0.005, 0.0, 0.90])
                gn = 0.5
                oz = 0.90
                grasping = True

            frame_data.append({
                'eef_pos': eef,
                'gripper_norm': gn,
                'obj_poses': {'coffee_pod': np.array([eef[0], 0.0, oz])},
                'closest_obj': 'coffee_pod',
                'closest_dist': abs(eef[0]),
            })

        # Build coarse segments (simulating what _build_segments produces)
        segments = [
            InteractionSegment(
                target_object='coffee_pod', phase='', start_step=0,
                end_step=15, gripper_grasping=False, other_objects=[]),
            InteractionSegment(
                target_object='coffee_pod', phase='', start_step=15,
                end_step=60, gripper_grasping=True, other_objects=[]),
        ]

        result = segmenter._label_phases(segments, frame_data)
        phases = [s.phase for s in result]

        # Must produce 'transport' (was missing before fix)
        assert 'transport' in phases, f"Expected 'transport' in phases, got {phases}"
        # Must also have 'grasp' and 'lift'
        assert 'grasp' in phases, f"Expected 'grasp' in phases, got {phases}"
        assert 'lift' in phases, f"Expected 'lift' in phases, got {phases}"

    def test_three_piece_like_trajectory(self):
        """Simulate three_piece_assembly: two objects grasped sequentially.

        Previously labeled as 'lift'/'transport' only, missing 'grasp'/'reach'.
        """
        segmenter = self._make_segmenter()

        # Frames 0-9: approach piece_1 (not grasping)
        # Frames 10-19: grasp piece_1 (on table)
        # Frames 20-39: lift + transport piece_1
        # Frames 40-49: place piece_1 (not grasping)
        # Frames 50-59: approach piece_2 (not grasping)
        # Frames 60-69: grasp piece_2 (on table)
        # Frames 70-89: lift + transport piece_2
        frame_data = []
        for i in range(90):
            if i < 10:
                eef = np.array([0.15 - i * 0.01, 0.0, 0.80])
                obj1 = np.array([0.0, 0.0, 0.80])
                obj2 = np.array([0.2, 0.0, 0.80])
            elif i < 20:
                eef = np.array([0.0, 0.0, 0.80])
                obj1 = np.array([0.0, 0.0, 0.80])
                obj2 = np.array([0.2, 0.0, 0.80])
            elif i < 40:
                z = 0.80 + (i - 20) * 0.01
                eef = np.array([0.0, 0.0, min(z, 0.95)])
                obj1 = np.array([0.0, 0.0, min(z, 0.95)])
                obj2 = np.array([0.2, 0.0, 0.80])
            elif i < 50:
                eef = np.array([0.1, 0.0, 0.85])
                obj1 = np.array([0.0, 0.0, 0.82])
                obj2 = np.array([0.2, 0.0, 0.80])
            elif i < 60:
                eef = np.array([0.2 - (i - 50) * 0.01, 0.0, 0.80])
                obj1 = np.array([0.0, 0.0, 0.82])
                obj2 = np.array([0.2, 0.0, 0.80])
            elif i < 70:
                eef = np.array([0.2, 0.0, 0.80])
                obj1 = np.array([0.0, 0.0, 0.82])
                obj2 = np.array([0.2, 0.0, 0.80])
            else:
                z = 0.80 + (i - 70) * 0.01
                eef = np.array([0.2, 0.0, min(z, 0.95)])
                obj1 = np.array([0.0, 0.0, 0.82])
                obj2 = np.array([0.2, 0.0, min(z, 0.95)])

            frame_data.append({
                'eef_pos': eef,
                'gripper_norm': 0.5 if (10 <= i < 40 or 60 <= i < 90) else 0.0,
                'obj_poses': {'piece_1': obj1, 'piece_2': obj2},
                'closest_obj': 'piece_1' if i < 40 else 'piece_2',
                'closest_dist': np.linalg.norm(eef - (obj1 if i < 40 else obj2)),
            })

        segments = [
            InteractionSegment(
                target_object='piece_1', phase='', start_step=0,
                end_step=10, gripper_grasping=False, other_objects=['piece_2']),
            InteractionSegment(
                target_object='piece_1', phase='', start_step=10,
                end_step=40, gripper_grasping=True, other_objects=['piece_2']),
            InteractionSegment(
                target_object='piece_1', phase='', start_step=40,
                end_step=50, gripper_grasping=False, other_objects=['piece_2']),
            InteractionSegment(
                target_object='piece_2', phase='', start_step=50,
                end_step=60, gripper_grasping=False, other_objects=['piece_1']),
            InteractionSegment(
                target_object='piece_2', phase='', start_step=60,
                end_step=70, gripper_grasping=True, other_objects=['piece_1']),
            InteractionSegment(
                target_object='piece_2', phase='', start_step=70,
                end_step=90, gripper_grasping=True, other_objects=['piece_1']),
        ]

        result = segmenter._label_phases(segments, frame_data)
        phases = [s.phase for s in result]

        # Must have grasp for both piece_1 and piece_2
        grasp_targets = [s.target_object for s in result if s.phase == 'grasp']
        assert 'piece_1' in grasp_targets, f"piece_1 should have grasp phase, got {phases}"
        assert 'piece_2' in grasp_targets, f"piece_2 should have grasp phase, got {phases}"

        # Must have reach for piece_2 (not 'place' since it wasn't grasped before)
        reach_or_pregrasp = [s for s in result
                             if s.target_object == 'piece_2'
                             and s.phase in ('reach', 'pre_reach', 'pre_grasp')]
        assert len(reach_or_pregrasp) > 0, \
            f"piece_2 should have reach/pre_grasp, got {phases}"

        # piece_1 after release should be 'place'
        place_segs = [s for s in result
                      if s.target_object == 'piece_1' and s.phase == 'place']
        assert len(place_segs) > 0, f"piece_1 post-grasp should be place"


# ─── Scanner per-frame phase detection ───


class TestDetectFramePhase:
    """Test OpportunityScanner._detect_frame_phase logic.

    We test the logic directly without a full MuJoCo env by creating
    a minimal mock scanner with the required attributes.
    """

    class MockScanner:
        """Minimal mock of OpportunityScanner for phase detection tests."""
        def __init__(self, lift_height=0.85, reach_dist=0.08, grasp_dist=0.06):
            self._lift_height = lift_height
            self._reach_dist = reach_dist
            self._grasp_dist = grasp_dist
            self._frames_above = {}
            self._grasped_objects = set()
            self.LIFT_TO_TRANSPORT_FRAMES = 10

        def _detect_frame_phase(self, state_info, segment):
            # Import the actual method logic
            from error_benchmark.framework.opportunity_scanner import OpportunityScanner
            return OpportunityScanner._detect_frame_phase(self, state_info, segment)

    def _make_segment(self, target='obj_a', grasping=True):
        return InteractionSegment(
            target_object=target, phase='', start_step=0,
            end_step=100, gripper_grasping=grasping, other_objects=[])

    def _make_state(self, obj_z=0.80, eef_dist=0.0, target='obj_a'):
        eef_pos = np.array([eef_dist, 0.0, obj_z])
        return {
            'objects': {target: {'pos': np.array([0.0, 0.0, obj_z])}},
            'eef_pos': eef_pos,
            'gripper_closed_norm': 0.5,
        }

    def test_grasping_below_lift_height(self):
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=True)
        state = self._make_state(obj_z=0.80)
        assert scanner._detect_frame_phase(state, seg) == 'grasp'

    def test_grasping_above_lift_height_early(self):
        """First few frames above lift_height → 'lift'."""
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=True)
        state = self._make_state(obj_z=0.90)
        # First call: frames_above=1 → 'lift'
        assert scanner._detect_frame_phase(state, seg) == 'lift'
        assert scanner._frames_above['obj_a'] == 1

    def test_grasping_above_lift_height_late(self):
        """After LIFT_TO_TRANSPORT_FRAMES above → 'transport'."""
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=True)
        state = self._make_state(obj_z=0.90)
        # Simulate 11 consecutive frames above
        scanner._frames_above['obj_a'] = 10
        assert scanner._detect_frame_phase(state, seg) == 'transport'

    def test_grasping_dip_below_resets_counter(self):
        """Object dips below lift_height → resets to 'grasp'."""
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=True)
        # Build up counter
        scanner._frames_above['obj_a'] = 15
        # Object dips below
        state = self._make_state(obj_z=0.80)
        assert scanner._detect_frame_phase(state, seg) == 'grasp'
        assert 'obj_a' not in scanner._frames_above

    def test_not_grasping_far(self):
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=False)
        state = self._make_state(obj_z=0.80, eef_dist=0.15)
        assert scanner._detect_frame_phase(state, seg) == 'pre_reach'

    def test_not_grasping_near(self):
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=False)
        state = self._make_state(obj_z=0.80, eef_dist=0.07)
        assert scanner._detect_frame_phase(state, seg) == 'reach'

    def test_not_grasping_very_close(self):
        scanner = self.MockScanner()
        seg = self._make_segment(grasping=False)
        state = self._make_state(obj_z=0.80, eef_dist=0.03)
        assert scanner._detect_frame_phase(state, seg) == 'pre_grasp'

    def test_not_grasping_previously_grasped(self):
        """After placing object → 'place'."""
        scanner = self.MockScanner()
        scanner._grasped_objects.add('obj_a')
        seg = self._make_segment(grasping=False)
        state = self._make_state(obj_z=0.80, eef_dist=0.05)
        assert scanner._detect_frame_phase(state, seg) == 'place'

    def test_not_grasping_different_object_not_placed(self):
        """Grasped obj_a before, now approaching obj_b → reach, not place."""
        scanner = self.MockScanner()
        scanner._grasped_objects.add('obj_a')
        seg = self._make_segment(target='obj_b', grasping=False)
        state = self._make_state(obj_z=0.80, eef_dist=0.07, target='obj_b')
        assert scanner._detect_frame_phase(state, seg) == 'reach'
