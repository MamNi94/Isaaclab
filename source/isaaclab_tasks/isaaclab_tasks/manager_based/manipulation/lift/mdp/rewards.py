# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

#Cube - Robot
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


#Finger Tip - Cube
def fingertips_object_proximity(
    env: ManagerBasedRLEnv,
    std: float = 0.04,  # tune: smaller = sharper reward near contact
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("tips_frame"),
    tip_indices=(0,1,2),  # order matches your target_frames list
    aggregate: str = "min",  # "min" (good for grasp), or "mean" (smoother shaping)
    uniformity_scale: float = 0.02,            # spread scale (meters); smaller -> stricter equality
    uniformity_weight: float = 1.0,            # how much to value symmetry vs proximity
    gate_with_proximity: bool = True,  
) -> torch.Tensor:
    """
    Reward based on fingertip proximity to the cube center using a tanh kernel.
    Assumes your FrameTransformer has the three fingertips as the first three targets
    (in the same order you listed them in the config).
    """
    # scene handles
    obj: RigidObject = env.scene[object_cfg.name]
    tips: FrameTransformer = env.scene[ee_frame_cfg.name]

    # (num_envs, 3) world pos of cube center
    cube_pos_w = obj.data.root_pos_w

    # (num_envs, num_targets, 3) → select given fingertip indices
    tips_w = tips.data.target_pos_w[..., tip_indices, :]  # shape: (num_envs, 3, 3)

    # distances tip→cube_center: (num_envs, 3)
    dists = torch.norm(tips_w - cube_pos_w.unsqueeze(1), dim=-1)

    # aggregate over tips
    if aggregate == "min":
        agg = torch.min(dists, dim=1).values
    elif aggregate == "mean":
        agg = torch.mean(dists, dim=1)
    else:
        raise ValueError("aggregate must be 'min' or 'mean'")

    proximity = 1.0 - torch.tanh(agg / std)
    

    # ---- uniformity (small spread across the 3 distances) ----
    mean_d = torch.mean(dists, dim=1, keepdim=True)           # (N,1)
    spread = torch.sqrt(torch.mean((dists - mean_d) ** 2, dim=1) + 1e-9)  # (N,)
    uniformity = 1.0 - torch.tanh(spread / uniformity_scale)  # high when spread is small

    if gate_with_proximity:
        # Only care about symmetry once we're somewhat close
        uniformity = uniformity * proximity.detach()

    # final reward
    return proximity + uniformity_weight * uniformity

  


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
