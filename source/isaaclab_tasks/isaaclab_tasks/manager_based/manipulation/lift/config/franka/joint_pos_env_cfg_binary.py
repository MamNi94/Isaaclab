# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka_custom import FRANKA_PANDA_CFG  # isort: skip
#from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
     
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_0/Franka")
       

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        #self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
         #   asset_name="robot",
          #  joint_names=["panda_finger.*"],
           # open_command_expr={"panda_finger_.*": 0.04},
            #close_command_expr={"panda_finger_.*": 0.0},
        #)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
	    asset_name="robot",
	    joint_names=[
		 ".*Core_Bottom_Box_Right",
        ".*Core_Bottom_Umdrehung_104",
        ".*Core_compact_Box_Thumb",
        ".*Motorbox_D5021_right_Connector_Right",
        ".*Motorbox_D5021_left_Connector_Left",
        ".*Motorbox_D5021_thumb_Connector_Thumb",
        ".*Connector_Servo_Right_Finger_Right",
        ".*Connector_Servo_left_Finger_Left",
        ".*Connector_Servo_thumb_Finger_Thumb"
	    ],
	    open_command_expr={
		 ".*Core_Bottom_Box_Right":0.5,
        ".*Core_Bottom_Umdrehung_104":0.5,
        ".*Core_compact_Box_Thumb":0.5,
        ".*Motorbox_D5021_right_Connector_Right":0.5,
        ".*Motorbox_D5021_left_Connector_Left":0.5,
        ".*Motorbox_D5021_thumb_Connector_Thumb":0.5,
        ".*Connector_Servo_Right_Finger_Right":0.5,
        ".*Connector_Servo_left_Finger_Left":0.5,
        ".*Connector_Servo_thumb_Finger_Thumb":0.5
	    }, 	
	    close_command_expr={
		  ".*Core_Bottom_Box_Right":0.0,
        ".*Core_Bottom_Umdrehung_104":0.0,
        ".*Core_compact_Box_Thumb":0.0,
        ".*Motorbox_D5021_right_Connector_Right":0.0,
        ".*Motorbox_D5021_left_Connector_Left":0.0,
        ".*Motorbox_D5021_thumb_Connector_Thumb":0.0,
        ".*Connector_Servo_Right_Finger_Right":0.0,
        ".*Connector_Servo_left_Finger_Left":0.0,
        ".*Connector_Servo_thumb_Finger_Thumb":0.0
	    }
	)
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_link8"

        self.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link8",       # or ".../panda_link8/Core_Bottom"
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),           # tweak as you like
            ),
        ],
    )



        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link8",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
