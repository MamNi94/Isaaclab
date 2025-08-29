from  isaaclab.envs import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
 # your config with custom gripper

# Spawn env
cfg = FrankaCubeLiftEnvCfg()
env = ManagerBasedRLEnv(cfg)
env.reset()

robot = env.scene["robot"]

# Command dicts (use DOF names from step 2)
open_command_expr = {...}
close_command_expr = {...}

# Prepare target arrays
open_cmd  = torch.zeros_like(robot.articulation_view.get_dof_targets())
close_cmd = torch.zeros_like(open_cmd)

name_to_id = {n: i for i, n in enumerate(robot.articulation_view.dof_names)}
for n, v in open_command_expr.items():
    open_cmd[name_to_id[n]] = v
for n, v in close_command_expr.items():
    close_cmd[name_to_id[n]] = v

# Open
for _ in range(300):
    robot.set_joint_position_targets(open_cmd)
    env.step(torch.zeros(env.action_space.shape))  # no arm motion

# Close
for _ in range(300):
    robot.set_joint_position_targets(close_cmd)
    env.step(torch.zeros(env.action_space.shape))



