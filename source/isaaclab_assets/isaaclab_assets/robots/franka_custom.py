# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np


def convert_angle(angle_deg):
    angle_rad = angle_deg / 180  * np.pi
    return angle_rad


##
# Configuration
##

# ... (previous imports and docstring remain the same)

FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        #usd_path="/home/nicolas/isaacsim/IsaacLab/My_Gripper/franka_mod_flat.usd",
        usd_path="/home/nicolas/isaacsim/IsaacLab/My_Gripper/Franka_mod_v2.usd",  # 1. Select your Gripper USD File
        
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,

            # 2. Add Initial States (just to not crash the script on init, will be set later)
            'Core_Bottom_Box_Right':convert_angle(15), 
            'Core_Bottom_Umdrehung_104':convert_angle(45),
            'Core_compact_Box_Thumb':convert_angle(0),
            'Motorbox_D5021_right_Connector_Right':convert_angle(39),
            'Motorbox_D5021_left_Connector_Left':convert_angle(1),
            'Motorbox_D5021_thumb_Connector_Thumb':convert_angle(19),
            #'Connector_Servo_Right_Finger_Right':convert_angle(-15),
            #'Connector_Servo_left_Finger_Left':convert_angle(30),
            #'Connector_Servo_thumb_Finger_Thumb':convert_angle(20)
                },
            ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0
        ),

        
        #"panda_link8": ImplicitActuatorCfg(
        # 3. Add Actuation for new joints
        "UDRF_Hand_Assembly": ImplicitActuatorCfg(
            joint_names_expr=[
                    ".*Core_Bottom_Box_Right",
                    ".*Core_Bottom_Umdrehung_104",
                    ".*Core_compact_Box_Thumb",
                    ".*Motorbox_D5021_right_Connector_Right",
                    ".*Motorbox_D5021_left_Connector_Left",
                    ".*Motorbox_D5021_thumb_Connector_Thumb",
                    #".*Connector_Servo_Right_Finger_Right",
                    #".*Connector_Servo_left_Finger_Left",
                    #".*Connector_Servo_thumb_Finger_Thumb"
            ],
            effort_limit_sim=20,  #<-- 4.
            stiffness=50,         #<-- 5.
            damping=10            #<-- 6.
        ),
        
    },
    
    soft_joint_pos_limit_factor=1.0,
)

"""Configuration of Franka Emika Panda robot with custom 3-finger gripper."""

FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

