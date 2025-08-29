import numpy as np
from isaacsim import SimulationApp
# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": False,"physics": True})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid

from omni.isaac.core.utils.prims import get_prim_at_path,create_prim
from pxr import UsdPhysics,PhysxSchema


open_stage("/home/nicolas/Downloads/UDRF_Hand_Assembly/UDRF_Hand_Assembly/test.usd")

# Start the simulation
sim = SimulationContext()
sim.initialize_physics() 
stage = get_current_stage()

gripper = ArticulationView(prim_paths_expr="/World/UDRF_Hand_Assembly", name="gripper")


cube = DynamicCuboid(
    prim_path="/World/TargetCube",
    position=[0, -0.1, -0.05],
    size=0.05,  # single float, not a list
)


########FIX GRIPPER IN FRAME ###############
fixed_joint_path = "/World/fix_gripper"
create_prim(fixed_joint_path, "PhysicsFixedJoint")
joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/UDRF_Hand_Assembly/Core_Bottom"])
########END FIX GRIPPER IN FRAME  


########FIX Cube IN FRAME ###############
fixed_joint_path = "/World/fix_cube"
create_prim(fixed_joint_path, "PhysicsFixedJoint")
joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/TargetCube"])
########END FIX Cube IN FRAME  


sim.reset()


for _ in range(100):
    sim.step()


#target_positions = np.array([0.5] * gripper.num_joints)
#gripper.set_joint_positions(target_positions)
#print("Target Postions", target_positions)

#while simulation_app.is_running():
    #sim.step()


# Training loop
for episode in range(100):
    print(f"Episode {episode}")
    
    # Reset object position
    cube.set_world_pose(position=np.array([0, -0.02, -0.1]))

    # Reset gripper (if needed: custom method or hard-coded)
    gripper.set_joint_positions(np.zeros(gripper.num_joints))
    
    sim.step()
    
    for step in range(50):  # one episode max 50 steps
        # Random action for now
        action = np.random.uniform(0.0, 0.8, size=gripper.num_joints)
        gripper.set_joint_positions(action)
        
        sim.step()
        
        # Grasp success check (very basic)
        cube_pos, _ = cube.get_world_pose()
        if cube_pos[2] > 0.1:
            print("Grasp Success!")
            break