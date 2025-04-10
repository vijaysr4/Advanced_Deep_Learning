"""
Code adapted from:
https://github.com/nv-tlabs/stmc/blob/main/src/tools/smplrifke_feats.py
"""

from typing import Tuple

import einops
import torch
from smplx import SMPLXLayer
from smplx.utils import SMPLXOutput
from torchtyping import TensorType

from utils.geometry_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
    to_matrix,
)

# ------------------------------------------------------------------------------------- #

num_frames, num_joints, num_poses, num_feats = None, None, None, None
Joints = TensorType["num_frames", "num_joints", 3]
Poses = TensorType["num_frames", "num_poses", 3, 3]
Feats = TensorType["num_frames", "num_feats"]
RootZ = TensorType["num_frames"]
VelLocal = TensorType["num_frames", 2]
VelAngles = TensorType["num_frames"]
PosesLocal = TensorType["num_frames", 132]
JointsLocal = TensorType["num_frames", 69]

# ------------------------------------------------------------------------------------- #


def group(
    root_z: RootZ,
    vel_root_xy_local: VelLocal,
    vel_angles: VelAngles,
    poses_6d_local: PosesLocal,
    joints_local: JointsLocal,
) -> Feats:
    """
    Pack the SMPLRIFKE features.
    1: root_z: (num_frames, 1)
    2: vel_root_xy_local: (num_frames, 2)
    1: vel_angles: (num_frames)
    132 = 22*6: poses_6d_local_flatten: (num_frames, num_joints, 6)
    69 = 23*3: joints_local_flatten: (num_frames, num_joints+1, 3)
    total: 205
    """
    # Flatten
    poses_6d_local_flatten = einops.rearrange(poses_6d_local, "k l t -> k (l t)")
    joints_local_flatten = einops.rearrange(joints_local, "k l t -> k (l t)")

    # Stack things together
    features, _ = einops.pack(
        [
            root_z,
            vel_root_xy_local,
            vel_angles,
            poses_6d_local_flatten,
            joints_local_flatten,
        ],
        "k *",
    )

    assert features.shape[-1] == 205
    return features


def ungroup(
    features: Feats, batch: bool = False
) -> Tuple[RootZ, VelLocal, VelAngles, PosesLocal, JointsLocal]:
    """
    Unpack the SMPLRIFKE features.
    1: root_z
    2: vel_root_xy_local
    1: vel_angles
    132 = 22*6: poses_6d_local_flatten
    69 = 23*3: joints_local_flatten
    total: 205
    """
    assert features.shape[-1] == 205

    ops_pattern = "b k *" if batch else "k *"
    (
        root_z,
        vel_root_xy_local,
        vel_angles,
        poses_6d_local_flatten,
        joints_local_flatten,
    ) = einops.unpack(features, [[], [2], [], [132], [69]], ops_pattern)

    ops_pattern = "b k (l t) -> b k l t" if batch else "k (l t) -> k l t"
    poses_6d_local = einops.rearrange(poses_6d_local_flatten, ops_pattern, t=6)
    joints_local = einops.rearrange(joints_local_flatten, ops_pattern, t=3)
    return root_z, vel_root_xy_local, vel_angles, poses_6d_local, joints_local


# ------------------------------------------------------------------------------------- #


def smpldata_to_smplrifkefeats(
    joints: Joints, poses: Poses, coord_system: str = "RIGHT_HAND_Z_UP"
) -> Feats:
    """Convert the SMPL data to SMPLRIFKE features."""
    assert len(joints.shape) == 3 and joints.shape[-2] == 24  # Translation + 24 joints
    assert len(poses.shape) == 4 and poses.shape[-3] == 22  # Rot matrix + 22 joints

    if coord_system == "RIGHT_HAND_Z_UP":  # Default cf humanml3d
        M = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=poses.device).float()
    elif coord_system == "RIGHT_HAND_Y_UP":  # Motionx
        M = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=poses.device
        ).float()
    elif coord_system == "RIGHT_HAND_Y_DOWN":  # E.T
        M = torch.tensor(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=poses.device
        ).float()
    else:
        raise ValueError(f"Unknown coord_system: {coord_system}")

    # Process poses
    # --------------------------------------------------------------------------------- #
    body_pose = poses[:, 1:].clone()
    root_orient = (M.T @ poses[:, 0]).clone()

    # Decompose the rot into 3 euler angles rotations to remove the Z rot for each frames
    global_euler = matrix_to_euler_angles(root_orient, "ZYX")
    rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)
    # Construct the rotations matrices
    rotX = axis_angle_rotation("X", rotX_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotZ = axis_angle_rotation("Z", rotZ_angle)
    # --------------------------------------------------------------------------------- #

    # Z (up) coordinate of the root
    # --------------------------------------------------------------------------------- #
    _joints = torch.einsum("ij,bvj->bvi", M.T, joints.clone())
    root_z = _joints[:, 0, [2]].clone()
    # --------------------------------------------------------------------------------- #

    # X, Y linear velocities of the root
    # --------------------------------------------------------------------------------- #
    # Translation without gravity axis (Z)
    root_xy = _joints[:, 0, [0, 1]].clone()
    vel_root_xy = torch.diff(root_xy, dim=0)
    vel_root_xy = torch.cat((root_xy[0][None], vel_root_xy), dim=0)
    # Rotate the vel_root_xy (rotation inverse in the indexes)
    vel_root_xy_local = torch.einsum("tkj,tk->tj", rotZ[:, :2, :2], vel_root_xy)
    # --------------------------------------------------------------------------------- #

    # Angular velocity of the Z angle
    # --------------------------------------------------------------------------------- #
    angles = matrix_to_axis_angle(rotZ)[:, 2]
    vel_angles = torch.diff(angles, dim=0)
    vel_angles = torch.cat((angles[0][None], vel_angles), dim=0)

    # --------------------------------------------------------------------------------- #

    # θ SMPL parameters in the root coordinate system
    # --------------------------------------------------------------------------------- #
    # construct the local global pose the one without the final Z rotation
    root_orient_local = rotY @ rotX
    # Replace the global orient with the one without rotation
    poses_local = torch.cat((root_orient_local[:, None], body_pose), dim=1)
    poses_6d_local = matrix_to_rotation_6d(poses_local)
    # --------------------------------------------------------------------------------- #

    # Joints in the root coordinate system
    # --------------------------------------------------------------------------------- #
    joints_p = _joints
    joints_p -= joints[:, [0]]
    # Delete the root from the local repr (already in root_z and root_xy)
    joints_p = joints_p[:, 1:]
    # Rotate the local_joints
    joints_local = torch.einsum("tkj,tlk->tlj", rotZ[:, :2, :2], joints_p[:, :, [0, 1]])
    joints_local = torch.stack(
        (joints_local[..., 0], joints_local[..., 1], joints_p[..., 2]), axis=-1
    )
    # --------------------------------------------------------------------------------- #

    # Stack things together
    features = group(
        root_z, vel_root_xy_local, vel_angles, poses_6d_local, joints_local
    )

    return features


def smplrifkefeats_to_smpldata(
    features: Feats, coord_system: str = "RIGHT_HAND_Z_UP"
) -> Tuple[Joints, Poses]:
    """Convert the SMPLRIFKE features to SMPL data."""
    # Unstack things
    (
        root_z,
        vel_root_xy_local,
        vel_angles,
        poses_6d_local,
        joints_local,
    ) = ungroup(features.clone())

    # --------------------------------------------------------------------------------- #
    poses_local = rotation_6d_to_matrix(poses_6d_local)
    root_orient_local = poses_local[:, 0]

    # Remove the predicted Z rotation inside root_orient_local
    # It is trained to be zero, but the network could produce non zeros outputs
    global_euler_local = matrix_to_euler_angles(root_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    # Integrate the angles
    angles = torch.cumsum(vel_angles, dim=0)

    # Construct the rotation matrix
    rotX = axis_angle_rotation("X", rotX_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotZ = axis_angle_rotation("Z", angles)
    # --------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------- #
    # Replace it with the one computed with velocities
    root_orient = rotZ @ rotY @ rotX
    poses = torch.cat(
        (root_orient[..., None, :, :], poses_local[..., 1:, :, :]), dim=-3
    )
    # --------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------- #
    # Rotate the root_xy (normal rotation in the indexes)
    vel_root_xy = torch.einsum("tjk,tk->tj", rotZ[:, :2, :2], vel_root_xy_local)
    # Integrate the root_xy
    root_xy = torch.cumsum(vel_root_xy, dim=-2)
    # --------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------- #
    joints = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
    joints = torch.stack(
        (joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1
    )
    # Add the root (which is still zero)
    joints = torch.cat((0 * joints[:, [0]], joints), axis=1)
    # Adding back the Z component
    joints[:, :, 2] += root_z[:, None]
    # Adding back the root_xy
    joints[:, :, [0, 1]] += root_xy[:, None]
    # --------------------------------------------------------------------------------- #

    # if coord_system == "RIGHT_HAND_Z_UP":  # Default cf humanml3d
    #     M = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=poses.device).float()
    # elif coord_system == "RIGHT_HAND_Y_UP":  # Motionx
    #     M = torch.tensor(
    #         [[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=poses.device
    #     ).float()
    # elif coord_system == "RIGHT_HAND_Y_DOWN":  # E.T.
    #     M = torch.tensor(
    #         [[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=poses.device
    #     ).float()
    # else:
    #     raise ValueError(f"Unknown coord_system: {coord_system}")
    # poses[:, 0] = M @ poses[:, 0].clone()
    # joints = torch.einsum("ij,bvj->bvi", M, joints.clone())

    return joints, poses


def smpldata_to_bodymodel(
    body_model: SMPLXLayer, joints: Joints, poses: Poses
) -> SMPLXOutput:
    """Infer the SMPLX model from the SMPLRIFKE features."""
    origin = torch.einsum("ik,ji->jk", [body_model.v_template, body_model.J_regressor])
    transl = joints[:, 0].clone() - origin[0]  # Shift by the root joint
    global_orient = poses[:, None, 0].clone()
    body_pose = poses[:, 1:22].clone()

    num_frames = joints.shape[0]
    left_hand_pose = body_model.left_hand_mean.reshape(15, 3)
    left_hand_pose = to_matrix("axisangle", left_hand_pose)
    left_hand_pose = left_hand_pose[None].repeat((num_frames, 1, 1, 1))
    right_hand_pose = body_model.right_hand_mean.reshape(15, 3)
    right_hand_pose = to_matrix("axisangle", right_hand_pose)
    right_hand_pose = right_hand_pose[None].repeat((num_frames, 1, 1, 1))

    parameters = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "transl": transl,
    }

    outputs = body_model(**parameters)

    return outputs
