import time
import numpy as np
from isaacsim import SimulationApp

# Set up SimulationApp first
simulation_app = SimulationApp({"headless": False, "physics": True})





from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid
from isaacsim.core.utils.prims import get_prim_at_path, create_prim, is_prim_path_valid
from isaacsim.core.utils.bounds import compute_aabb
from pxr import Usd, UsdPhysics, UsdGeom

import gymnasium as gym
from gymnasium import spaces





class GraspEnv:
    def __init__(self):
        # Load scene
        open_stage("/home/nicolas/Downloads/UDRF_Hand_Assembly/UDRF_Hand_Assembly/test.usd")

        # Set up sim
        self.sim = SimulationContext()
        self.sim.initialize_physics()
        self.stage = get_current_stage()

        # Gripper + object
        self.gripper = ArticulationView(prim_paths_expr="/World/UDRF_Hand_Assembly", name="gripper")
        self.cube = DynamicCuboid(
            prim_path="/World/TargetCube",
            position=[0.02, -0.02, -0.10499933],
            size=0.05
        )
        self.floor = DynamicCuboid(
            prim_path="/World/Floor",
            position=[0, 0, -0.63],
            size=1
        )

        self.finger_paths = [
            "/World/UDRF_Hand_Assembly/Finger_left",
            "/World/UDRF_Hand_Assembly/Finger_Right",
            "/World/UDRF_Hand_Assembly/Finger_Thumb",
        ]

        # Fixed joints
        self._fix_prim("/World/UDRF_Hand_Assembly/Core_Bottom", "/World/fix_gripper")
        self._fix_prim("/World/Floor", "/World/fix_floor")

        self.bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_],
            useExtentsHint=False
        )
        joint_limits_deg = [[-20,70 ], [0,90 ], [-35,35] , [-50,40] , [0, 90], [-60,20], [-75,15], [-10,80] , [-30,60]]
        self.joint_limits_rad = np.array(joint_limits_deg) / 180.0 * np.pi

     

        self.sim.reset()
        for _ in range(10):  # settle
            self.sim.step()

    def _fix_prim(self, body1_path, joint_path):
        create_prim(joint_path, "PhysicsFixedJoint")
        joint_prim = get_prim_at_path(joint_path)
        UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
        UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets([body1_path])

  

    def reset(self):
        # Reset object and gripper
        self.cube.set_world_pose(position=np.array([0.02, -0.02, -0.10499933]))
        self.gripper.set_joint_positions(np.zeros(self.gripper.num_joints))
        self.sim.reset()
        for _ in range(10):
            self.sim.step()
        return self.get_observation()

    def step(self, action):
        self.gripper.set_joint_positions(action)
        self.sim.step()

        obs = self.get_observation()
        #reward = self.compute_reward()
        #done = reward >= 1.0
        info = {}

        return obs,  info

    def get_observation(self):
        cube_pos, _ = self.cube.get_world_pose()
        joint_positions = self.gripper.get_joint_positions()
        print("joint_postitions",joint_positions)
        if joint_positions is None:
            print("[ERROR] Gripper joint positions not available!")
            joint_positions = np.zeros(self.gripper.num_joints)
        
        # If joint_positions is 2D like (n, 1), flatten it:
        joint_positions = joint_positions.flatten()

        # Or, if it's (1, n), also flatten
        joint_positions = joint_positions.reshape(-1)

        # Then concatenate:
        obs = np.concatenate([cube_pos, joint_positions])
        return obs
        

    
class GymGraspEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = GraspEnv()
        
        obs = self.env.get_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=0.8, shape=(self.env.gripper.num_joints,), dtype=np.float32)

        joint_limits_deg = [[-20,70 ], [0,90 ], [-35,35] , [-50,40] , [0, 90], [-60,20], [-75,15], [-10,80] , [-30,60]]
        joint_limits_rad = np.array(joint_limits_deg) / 180.0 * np.pi

        self.joint_lower_limits = joint_limits_rad[:, 0]
        self.joint_upper_limits = joint_limits_rad[:, 1]

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs.astype(np.float32), {}

    def step_(self, action):
        obs, info = self.env.step(action)
        return obs.astype(np.float32), info
    
    def step(self, action):
        
        
        # Limit how big the action steps can be (in radians)
        max_delta = 0.05  # e.g., ~3 degrees
        action = np.clip(action, -1.0, 1.0)  # Assuming policy outputs in [-1, 1]
        delta_action = action * max_delta

        # Get current joint positions
        current_positions = self.env.gripper.get_joint_positions()[0]

        # Compute the new target joint positions
        new_positions = current_positions + delta_action
        new_positions = np.clip(new_positions, self.joint_lower_limits, self.joint_upper_limits)
        
        
        
        #obs, reward, done, info = self.env.step(action)
        obs, info = self.env.step(new_positions)

   

        return obs.astype(np.float32),info

    def render(self):
        pass

    def close(self):
        pass

def evaluate():
    # Load trained model and environment
    env = GymGraspEnv()
    env.reset()
    model = PPO.load("My_Scripts/grasp_sac_model")
    print("model loaded")
    # Run one episode
    obs, _ = env.reset()
    t = time.time()
    iterations = 0
    action = env.env.gripper.get_joint_positions()[0]
    d_action = action- action
    for step in range(10000):

        #simulation_app.update()
        #env.env.sim.step()
       
        current_joints_position = env.env.gripper.get_joint_positions()
        dt = time.time()-t
     
        #if  dt > 1:
        #env.env.cube.set_world_pose(position=np.array([0.02, -0.02, -0.1]))
        #obs, _ =env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        try:
            if action == None:
                return
        except:
            print("action ",action)

        
        try:
            if current_joints_position == None:
                return
        except:
            print("current joint None")

        current_joints_position[0]
        
        delta = action - current_joints_position[0]
        d_action = delta/10

        print("d action", d_action)
        
        
        print("joints:", env.env.gripper.get_joint_positions())
        t = time.time()
        iterations = 0

        obs , info = env.step(action)


        #if iterations < 11:
        
         #   current_action = current_joints_position + d_action
          #  print("current action", current_action)
           # obs , info = env.step(action)
            #iterations +=1
            
        simulation_app.update()
       
       
       
        
        #if step == 3:
         #   break



    
    
    

if __name__ == "__main__":
    from stable_baselines3 import PPO

    #evaluate()
    while simulation_app.is_running():
        simulation_app.update()
        evaluate()
        
            
