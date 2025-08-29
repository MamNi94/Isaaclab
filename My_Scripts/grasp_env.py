

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "physics": True})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim, is_prim_path_valid,create_prim
from omni.isaac.core.utils.bounds import compute_aabb

from pxr import Usd, UsdPhysics, UsdGeom, Gf

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
            position=[0.02, -0.02, -0.105],
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

    def draw_debug_marker(self,position, name="/World/Debug/Marker", scale=0.012):
        position = Gf.Vec3d(*position)  # Ensure it's in the correct format

        if not is_prim_path_valid(name):
            create_prim(
                prim_path=name,
                prim_type="Sphere",
                translation=position,
                scale=(scale, scale, scale)
            )

        
        else:
            prim = get_prim_at_path(name)
            attr = prim.GetAttribute("xformOp:translate")
            if attr.IsValid():
                attr.Set(position)

    def _fix_prim(self, body1_path, joint_path):
        create_prim(joint_path, "PhysicsFixedJoint")
        joint_prim = get_prim_at_path(joint_path)
        UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
        UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets([body1_path])

    def reset(self):
        # Reset object and gripper
        print("env reset")
        self.sim.reset()
    
        self.cube.set_world_pose(position=np.array([0.02, -0.02, -0.105]))
        #self.gripper.set_joint_positions(np.zeros(self.gripper.num_joints))


        #initiate gripper ranom
        random_joint_values = np.random.uniform(
            low=self.joint_limits_rad[:, 0],  # lower bounds
            high=self.joint_limits_rad[:, 1]  # upper bounds
        )

        # Set joint positions
        self.gripper.set_joint_positions(random_joint_values)
        
        for _ in range(20):
            self.sim.step()
        return self.get_observation()

    def step(self, action):
        self.gripper.set_joint_positions(action)
        self.sim.step()


        obs = self.get_observation()
        
        reward = self.compute_reward()

        done = reward >= 20
        info = {}

        
        if abs(max(obs[:2]) ) > 1:

            obs[:2] =0
            done = True
            reward = -50
            print("obs mod ", obs)
  

        

        return obs, reward, done, info

    def get_observation(self):
        cube_pos, _ = self.cube.get_world_pose()
        joint_positions = self.gripper.get_joint_positions()
        #print("joint positions ", joint_positions)
        # If joint_positions is 2D like (n, 1), flatten it:
        joint_positions = joint_positions.flatten()

        # Or, if it's (1, n), also flatten
        joint_positions = joint_positions.reshape(-1)


        if abs(max(cube_pos)) > 1.0:
            print("cube out of bounds")
            

        # Then concatenate:
        obs = np.concatenate([cube_pos, joint_positions])
        

        

        return obs
        



    def compute_reward_(self):
        cube_pos, _ = self.cube.get_world_pose()
        reward = 0.0
        
        offsets = [[-0.005,0.045,-0.02], [0.02,0.045,-0.02], [0,005.045,-0.01]]

        distances = []

        for i, path in enumerate(self.finger_paths):
            tip_position = self.get_tip_position(path, offset= offsets[i])
            self.draw_debug_marker(tip_position, name=f"/World/Debug/{path.split('/')[-1]}_tip")
           

            distance = np.linalg.norm(cube_pos - tip_position)

            #reward += 1.0 - np.clip(distance / 0.2, 0.0, 1.0)
            #distance = max(0.001, distance)

            distances.append(distance)

        def closeness_cost(values):
            return max(values) - min(values)
            
        distance_reward = (1/sum(distances))**2 
        sync_reward = (1/closeness_cost(distances))**2
        reward = distance_reward+ distance_reward
        print("distance array",distances)
        print("distance reward",distance_reward, "sync_reward", sync_reward)
        return  reward
    
    def compute_reward(self):
        cube_pos, _ = self.cube.get_world_pose()
        #print("cube position", cube_pos)
        #cube_z_position = cube_pos[0]

        z = cube_pos[2]

        d_z = 0 - z
        #print("z",z)

        if z < -0.104:
            reward = 0
        else:
            reward = 0
            #reward = ((1/d_z)/1000)**2


        base_z = -0.105  # height of table
        reward = max(0, z - base_z) * 10


        #print(reward)

        return reward


    def get_tip_position(self, prim_path , offset = [0,0,0]):

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim not found: {prim_path}")

        xform = UsdGeom.Xformable(prim)
        local_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        #local to world
        prim_origin_local = Gf.Vec3d(0,0,0)
        prim_world_pos = local_to_world.Transform(prim_origin_local)

        #offset
        offset_local = Gf.Vec3d(*offset)
        offset_world = local_to_world.Transform(offset_local)

        #print("finger_position",prim_world_pos )
        #print("finger_position",offset_world )

        return  np.array([offset_world[0], offset_world[1], offset_world[2]])

    def get_finger_front_half_bounds(self, finger_path):
        aabb = compute_aabb(
            prim_path=finger_path,
            bbox_cache=self.bbox_cache,
            include_children=False
        )
        if aabb is None or len(aabb) != 6:
            return None
        min_corner = np.array(aabb[:3])
        max_corner = np.array(aabb[3:])
        center = (min_corner + max_corner) / 2
        front_min = np.array([center[0], min_corner[1], min_corner[2]])
        front_max = np.array([max_corner[0], max_corner[1], max_corner[2]])


        return front_min, front_max
    
class GymGraspEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = GraspEnv()
        
        self.max_episode_steps = 200
        self.current_step = 0

        obs = self.env.get_observation()
     
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

        joint_limits_deg = [[-20,70 ], [0,90 ], [-35,35] , [-50,40] , [0, 90], [-60,20], [-75,15], [-10,80] , [-30,60]]
        joint_limits_rad = np.array(joint_limits_deg) / 180.0 * np.pi

        self.joint_lower_limits = joint_limits_rad[:, 0]
        self.joint_upper_limits = joint_limits_rad[:, 1]

        self.action_space = spaces.Box(
            low=self.joint_lower_limits,
            high=self.joint_upper_limits,
            dtype=np.float32
        )
        #self.action_space = spaces.Box(low= -np.pi, high=np.pi, shape=(self.env.gripper.num_joints,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        print("Stage reset")
        obs = self.env.reset()
        self.current_step = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1
        
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
        obs, reward, done, info = self.env.step(new_positions)

        done = False
        if self.current_step >= self.max_episode_steps:
            done = True
            print(obs, reward, info)
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):  # Implement check_invalid_physics()
            print("Invalid physics detected: terminating episode early")
            print(obs)
            reward = -50
            done = True
            obs = np.array([0.02343354 ,-0.01901526, -0.118299 ,   0.72929555 , 0.59860486 ,-0.28831086, -0.68519914 , 1.1403425  , 0.04274729 ,-0.23389198,  0.05641067 ,-0.3293158 ])
                                

            
        return obs.astype(np.float32), reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass




'''
if __name__ == "__main__":
    env = GraspEnv()

    for episode in range(10):
        obs = env.reset()
        for step in range(50):
            action = np.random.uniform(0.0, 0.8, size=env.gripper.num_joints)
            obs, reward, done, info = env.step(action)
            if done:
                print(f"Episode {episode}: Success at step {step} with reward {reward}")
                break
'''

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import SAC
    #model = SAC("MlpPolicy", env, verbose=1, device="gpu")

    # Use wrapped Gym-style env
    env = DummyVecEnv([lambda: GymGraspEnv()])

    #model = PPO("MlpPolicy", env, verbose=1, device = "cpu", n_steps = 512)
    # Typical PPO config
    '''
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=2048,  # Recommended for most environments, tune based on yours
        batch_size=256,  # Should divide n_steps evenly
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=2.5e-3,
    )
    model.learn(total_timesteps=200000)
    '''

    model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=2,
    device="cuda",
    learning_rate=3e-4,
    buffer_size=1000000,
    batch_size=256,
    gamma=0.99,
    tau=0.0025,
    train_freq=1,
    gradient_steps=2,
    ent_coef="auto",
    )

    model.learn(total_timesteps=100000)

   

    # Optionally save
    model.save("My_Scripts/grasp_sac_model")



    # Clean shutdown
    simulation_app.close()
   





names = ['Core_Bottom_Box_Right', 
 'Core_Bottom_Umdrehung_104', 
 'Core_compact_Box_Thumb', 
 'Motorbox_D5021_right_Connector_Right', 
 'Motorbox_D5021_left_Connector_Left', 
 'Motorbox_D5021_thumb_Connector_Thumb', 
 'Connector_Servo_Right_Finger_Right', 
 'Connector_Servo_left_Finger_Left', 
 'Connector_Servo_thumb_Finger_Thumb']


