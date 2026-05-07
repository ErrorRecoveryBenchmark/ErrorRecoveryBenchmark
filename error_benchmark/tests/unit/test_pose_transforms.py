#!/usr/bin/env python
"""Unit tests for extracted MimicGen pose transform utilities."""

import numpy as np
import pytest
from error_benchmark.framework.utils.pose_transforms import (
    make_pose, unmake_pose, pose_inv, pose_in_A_to_pose_in_B,
    transform_source_data_segment_using_object_pose,
    interpolate_poses, quat_slerp,
)


class TestMakePose:
    def test_identity(self):
        pos = np.zeros(3)
        rot = np.eye(3)
        pose = make_pose(pos, rot)
        assert pose.shape == (4, 4)
        np.testing.assert_array_almost_equal(pose, np.eye(4))

    def test_batch(self):
        pos = np.zeros((5, 3))
        rot = np.tile(np.eye(3), (5, 1, 1))
        poses = make_pose(pos, rot)
        assert poses.shape == (5, 4, 4)

    def test_roundtrip(self):
        pos = np.array([1.0, 2.0, 3.0])
        rot = np.eye(3)
        pose = make_pose(pos, rot)
        pos_out, rot_out = unmake_pose(pose)
        np.testing.assert_array_almost_equal(pos, pos_out)
        np.testing.assert_array_almost_equal(rot, rot_out)


class TestPoseInv:
    def test_identity(self):
        inv = pose_inv(np.eye(4))
        np.testing.assert_array_almost_equal(inv, np.eye(4))

    def test_inverse_property(self):
        """pose @ inv(pose) == identity"""
        pos = np.array([1.0, 2.0, 3.0])
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        pose = make_pose(pos, rot)
        inv = pose_inv(pose)
        product = pose @ inv
        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=6)


class TestTransformSegment:
    def test_identity_transform(self):
        """Same object pose => no change to EEF poses."""
        obj_pose = np.eye(4)
        obj_pose[:3, 3] = [0.5, 0.0, 0.8]
        src_eef = np.tile(np.eye(4), (10, 1, 1))
        src_eef[:, :3, 3] = np.linspace([0.4, 0.0, 0.9], [0.5, 0.0, 0.8], 10)

        transformed = transform_source_data_segment_using_object_pose(
            obj_pose=obj_pose,
            src_eef_poses=src_eef,
            src_obj_pose=obj_pose,
        )
        np.testing.assert_array_almost_equal(transformed, src_eef, decimal=6)

    def test_translation_shift(self):
        """Object moved +0.1 in X => EEF also shifts +0.1 in X."""
        src_obj = np.eye(4)
        src_obj[:3, 3] = [0.5, 0.0, 0.8]

        new_obj = np.eye(4)
        new_obj[:3, 3] = [0.6, 0.0, 0.8]

        src_eef = np.tile(np.eye(4), (5, 1, 1))
        src_eef[:, :3, 3] = [0.5, 0.0, 0.9]

        transformed = transform_source_data_segment_using_object_pose(
            obj_pose=new_obj, src_eef_poses=src_eef, src_obj_pose=src_obj,
        )
        expected_pos = np.array([0.6, 0.0, 0.9])
        np.testing.assert_array_almost_equal(
            transformed[0, :3, 3], expected_pos, decimal=5)


class TestQuatSlerp:
    def test_endpoints(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 0.7071, 0.7071])
        np.testing.assert_array_almost_equal(quat_slerp(q1, q2, 0.0), q1)
        np.testing.assert_array_almost_equal(quat_slerp(q1, q2, 1.0), q2)

    def test_midpoint_unit(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 1.0, 0.0])
        mid = quat_slerp(q1, q2, 0.5)
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-6


class TestInterpolatePoses:
    def test_num_steps(self):
        p1 = np.eye(4)
        p2 = np.eye(4)
        p2[:3, 3] = [1.0, 0.0, 0.0]
        poses, n = interpolate_poses(p1, p2, num_steps=5)
        assert n == 5
        assert poses.shape[0] == 7

    def test_endpoints_match(self):
        p1 = np.eye(4)
        p1[:3, 3] = [0.0, 0.0, 0.0]
        p2 = np.eye(4)
        p2[:3, 3] = [1.0, 0.0, 0.0]
        poses, _ = interpolate_poses(p1, p2, num_steps=3)
        np.testing.assert_array_almost_equal(poses[0, :3, 3], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(poses[-1, :3, 3], [1.0, 0.0, 0.0])
