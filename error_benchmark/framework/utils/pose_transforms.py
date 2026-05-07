#!/usr/bin/env python
"""
Pose transform utilities extracted from MimicGen pose_utils.py.

Pure numpy — no MimicGen or robosuite dependency. The quaternion helpers
(mat2quat, quat2mat) are inlined from robosuite.utils.transform_utils so
this module can be imported without robosuite on the path.

Original: shared/mimicgen_workspace/mimicgen/mimicgen/utils/pose_utils.py
License: NVIDIA Source Code License
"""

import math
import numpy as np


# ─── Quaternion helpers (inlined from robosuite) ────────────

def _mat2quat(rot):
    """Convert 3x3 rotation matrix to (x, y, z, w) quaternion."""
    # Shepperd's method
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def _quat2mat(quat):
    """Convert (x, y, z, w) quaternion to 3x3 rotation matrix."""
    x, y, z, w = quat
    n = np.dot(quat, quat)
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = w * s * x, w * s * y, w * s * z
    xx, xy, xz = x * s * x, x * s * y, x * s * z
    yy, yz, zz = y * s * y, y * s * z, z * s * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


# ─── Pose construction / decomposition ─────────────────────

def make_pose(pos, rot):
    """
    Make homogeneous pose matrices from position vectors and rotation matrices.

    Args:
        pos: (..., 3) position vectors
        rot: (..., 3, 3) rotation matrices

    Returns:
        pose: (..., 4, 4) homogeneous matrices
    """
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = np.zeros(pos.shape[:-1] + (4, 4))
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def unmake_pose(pose):
    """
    Split homogeneous pose matrices into position and rotation.

    Returns:
        (pos (..., 3), rot (..., 3, 3))
    """
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose):
    """
    Inverse of homogeneous pose matrices.
    [R t; 0 1]^-1 = [R.T  -R.T*t; 0  1]
    """
    num_axes = len(pose.shape)
    assert num_axes >= 2
    inv_pose = np.zeros_like(pose)
    inv_pose[..., :3, :3] = np.transpose(
        pose[..., :3, :3],
        tuple(range(num_axes - 2)) + (num_axes - 1, num_axes - 2),
    )
    inv_pose[..., :3, 3] = np.matmul(
        -inv_pose[..., :3, :3], pose[..., :3, 3:4]
    )[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def pose_in_A_to_pose_in_B(pose_in_A, pose_A_in_B):
    """Convert poses from frame A to frame B: pose_A_in_B @ pose_in_A."""
    return np.matmul(pose_A_in_B, pose_in_A)


# ─── Quaternion / rotation interpolation ────────────────────

def quat2axisangle(quat):
    """Convert (x, y, z, w) quaternion to (axis, angle) pair."""
    if quat[3] > 1.0:
        quat = quat.copy()
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat = quat.copy()
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3), 0.0
    return quat[:3] / den, 2.0 * math.acos(quat[3])


def axisangle2quat(axis, angle):
    """Convert axis-angle to (x, y, z, w) quaternion."""
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat_slerp(q1, q2, tau):
    """Spherical linear interpolation between two quaternions."""
    if tau == 0.0:
        return q1.copy()
    elif tau == 1.0:
        return q2.copy()
    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.0:
        return q1.copy()
    q2 = q2.copy()
    if d < 0.0:
        d = -d
        q2 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < np.finfo(float).eps * 4.0:
        return q1.copy()
    isin = 1.0 / math.sin(angle)
    return q1 * math.sin((1.0 - tau) * angle) * isin + q2 * math.sin(tau * angle) * isin


def interpolate_rotations(R1, R2, num_steps):
    """Interpolate between two 3x3 rotation matrices via axis-angle."""
    delta_rot_mat = R2.dot(R1.T)
    delta_quat = _mat2quat(delta_rot_mat)
    delta_axis, delta_angle = quat2axisangle(delta_quat)

    if delta_angle < 0.05:
        rot_steps = np.array([R2 for _ in range(num_steps)])
    else:
        rot_step_size = delta_angle / num_steps
        delta_rot_steps = [
            _quat2mat(axisangle2quat(delta_axis, i * rot_step_size))
            for i in range(num_steps)
        ]
        rot_steps = np.array([delta_rot_steps[i].dot(R1) for i in range(num_steps)])

    rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)
    return rot_steps


def interpolate_poses(pose_1, pose_2, num_steps):
    """
    Linear position + axis-angle rotation interpolation between two 4x4 poses.

    Args:
        pose_1: (4, 4) start pose
        pose_2: (4, 4) end pose
        num_steps: number of intermediate points (not counting start/end)

    Returns:
        (pose_steps (N+2, 4, 4), num_steps)
    """
    pos1, rot1 = unmake_pose(pose_1)
    pos2, rot2 = unmake_pose(pose_2)

    if num_steps == 0:
        poses = make_pose(
            np.stack([pos1, pos2]),
            np.stack([rot1, rot2]),
        )
        return poses, 0

    n = num_steps + 1  # include starting pose
    delta_pos = pos2 - pos1
    pos_step_size = delta_pos / n
    grid = np.arange(n).astype(np.float64)
    pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(n)])
    pos_steps = np.concatenate([pos_steps, pos2[None]], axis=0)

    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=n)

    pose_steps = make_pose(pos_steps, rot_steps)
    return pose_steps, num_steps


# ─── Object-centric segment transform (core MimicGen algorithm) ─

def transform_source_data_segment_using_object_pose(
    obj_pose,
    src_eef_poses,
    src_obj_pose,
):
    """
    Transform a source EEF trajectory so that relative poses w.r.t. the object
    frame are preserved. This is the core MimicGen warping algorithm.

    Args:
        obj_pose: (4, 4) object pose in current (target) scene
        src_eef_poses: (T, 4, 4) EEF poses from source demonstration
        src_obj_pose: (4, 4) object pose in source demonstration

    Returns:
        transformed_eef_poses: (T, 4, 4)
    """
    # EEF poses relative to source object frame
    src_eef_rel_obj = pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=pose_inv(src_obj_pose[None]),
    )
    # Apply to new object frame
    transformed = pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed
