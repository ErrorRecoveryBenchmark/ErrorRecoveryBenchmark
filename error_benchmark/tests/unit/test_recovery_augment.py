#!/usr/bin/env python
"""Unit tests for MimicGen-style recovery augmentation helpers."""

import numpy as np

from error_benchmark.framework.recovery_mimicgen import (
    build_recovery_subtask_specs,
    infer_target_pose_from_action,
    resolve_subtask_object_ref,
)
from error_benchmark.framework.recovery_types import RECOVERY_SUBTASKS
from error_benchmark.framework.utils.pose_transforms import (
    make_pose,
    transform_source_data_segment_using_object_pose,
)


class TestWarpingTargetObject:
    """Test that object-centric pose transforms follow the referenced object."""

    def test_target_object_selection(self):
        src_obj_pos = np.array([0.5, 0.0, 0.8])
        target_obj_pos = np.array([0.7, 0.0, 0.8])

        src_eef_pos = np.array([0.5, 0.0, 0.9])
        src_eef_pose = make_pose(src_eef_pos, np.eye(3))

        src_obj_pose = make_pose(src_obj_pos, np.eye(3))
        target_obj_pose = make_pose(target_obj_pos, np.eye(3))

        transformed = transform_source_data_segment_using_object_pose(
            obj_pose=target_obj_pose,
            src_eef_poses=src_eef_pose[None],
            src_obj_pose=src_obj_pose,
        )
        np.testing.assert_array_almost_equal(
            transformed[0, :3, 3], [0.7, 0.0, 0.9], decimal=4
        )


class TestRecoveryTaskSpec:
    def test_all_recovery_subtasks_have_specs(self):
        specs = build_recovery_subtask_specs(None)
        assert set(RECOVERY_SUBTASKS) == set(specs.keys())

    def test_default_object_ref_strategies(self):
        specs = build_recovery_subtask_specs(None)
        assert specs["retract"].object_ref_strategy == "none"
        assert specs["release"].object_ref_strategy == "none"
        assert specs["navigate_to_object"].object_ref_strategy == "target_object"
        assert specs["re_transport"].object_ref_strategy == "placement_ref"
        assert specs["correct_position"].object_ref_strategy == "none"
        assert specs["resume_task"].object_ref_strategy == "none"

    def test_stack_place_uses_placement_ref(self):
        specs = build_recovery_subtask_specs(None)
        task_config = {
            "recovery_augmentation": {
                "placement_refs": {
                    "cubeA": "cubeB",
                }
            }
        }
        object_ref = resolve_subtask_object_ref(
            spec=specs["re_place"],
            task_name="stack",
            task_config=task_config,
            target_object="cubeA",
            scene_labels={},
        )
        assert object_ref == "cubeB"


class TestTargetPoseInference:
    def test_relative_position_targets_match_action_scale(self):
        current_pose = make_pose(np.array([0.2, -0.1, 0.8]), np.eye(3))
        action = np.array([0.5, -0.25, 1.0, 0.0, 0.0, 0.0, -1.0])
        target_pose = infer_target_pose_from_action(
            current_pose=current_pose,
            action=action,
            max_dpos=0.1,
            max_drot=0.5,
            relative=True,
        )
        np.testing.assert_allclose(
            target_pose[:3, 3],
            np.array([0.25, -0.125, 0.9]),
            atol=1e-6,
        )

    def test_relative_rotation_targets_match_axis_angle(self):
        current_pose = make_pose(np.zeros(3), np.eye(3))
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
        target_pose = infer_target_pose_from_action(
            current_pose=current_pose,
            action=action,
            max_dpos=0.1,
            max_drot=1.0,
            relative=True,
        )
        expected = np.array(
            [
                [np.cos(0.5), -np.sin(0.5), 0.0],
                [np.sin(0.5), np.cos(0.5), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(target_pose[:3, :3], expected, atol=1e-6)
