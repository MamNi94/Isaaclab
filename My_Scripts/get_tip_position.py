
import numpy as np
from isaacsim import SimulationApp
import time

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import SimulationContext
#from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Usd
from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.prims import create_prim, get_prim_at_path, is_prim_path_valid
from pxr import UsdGeom, UsdPhysics, Gf
import math
import omni.physx as physx



def draw_debug_marker(position, name="/World/Debug/Marker", scale=0.012):
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




def get_tip_position(stage, prim_path , offset = [0,0,0]):
    prim = stage.GetPrimAtPath(prim_path)
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

    print("finger_position",prim_world_pos )
    print("finger_position",offset_world )

    return np.array([prim_world_pos[0], prim_world_pos[1], prim_world_pos[2]]), np.array([offset_world[0], offset_world[1], offset_world[2]])


def get_joint_state(gripper, joint_name="Connector_Servo_Right_Finger_Right", offset_in_local_frame=(0, 0, 0)):
    # Step 1: Get joint index
    joint_index = gripper.get_dof_index(joint_name)
    if joint_index == -1:
        raise ValueError(f"Joint '{joint_name}' not found in articulation.")

    
    # Step 2: Get joint angle (position of DOF)
    joint_positions = gripper.get_joint_positions()
    try:
        if joint_positions == None:
            return
    except:
        print("not running)")
    #print(joint_positions)
    joint_angle = joint_positions[0][joint_index]

    

    print(f"[{joint_name}] → angle (rad): {joint_angle:.3f}, position:")



finger_paths = [
            "/World/UDRF_Hand_Assembly/Finger_left",
            "/World/UDRF_Hand_Assembly/Finger_Right",
            "/World/UDRF_Hand_Assembly/Finger_Thumb",
        ]

joint_paths = [
            "/World/UDRF_Hand_Assembly/joints/Connector_Servo_Right_Finger_Right"
]

# Open the USD file
open_stage("/home/nicolas/Downloads/UDRF_Hand_Assembly/UDRF_Hand_Assembly/test.usd")


# Start the simulation
sim = SimulationContext()
stage = get_current_stage()


#get Articulatio view
gripper = ArticulationView(prim_paths_expr="/World/UDRF_Hand_Assembly", name="gripper")




sim.reset()
# Run a few steps to initialize physics handles
for _ in range(100):
    simulation_app.update()
    sim.step()

#print(gripper.dof_names)
positions = gripper.get_joint_positions()



########FIX GRIPPER IN FRAME ###############

fixed_joint_path = "/World/fix_gripper"
create_prim(fixed_joint_path, "PhysicsFixedJoint")

joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/UDRF_Hand_Assembly/Core_Bottom", "/World/fix_gripper"])
########END FIX GRIPPER IN FRAME  

#Get DOF names to map joints (for debugging)
dof_names = gripper.dof_names
print("DOF names:", dof_names)

print(positions)

finger_path  = "/World/UDRF_Hand_Assembly/Finger_Thumb"

offset_right = [0.02,0.045,-0.02]
offset_left = [-0.005,0.045,-0.02]
offset_thumb = [0,005.045,-0.01]

offsets = [[-0.005,0.045,-0.02], [0.02,0.045,-0.02], [0,005.045,-0.01]]

sim.reset()
for _ in range(10):  # settle
    sim.step()
# Run the simulation (optional, for interactive viewing)
while simulation_app.is_running():
    simulation_app.update()
    sim.step()
    for i, finger_path in enumerate(finger_paths):
        finger_base, finger_tip = get_tip_position(stage, finger_path, offsets[i])

        draw_debug_marker(finger_base, name=f"/World/Debug/{finger_path.split('/')[-1]}_base")
        draw_debug_marker(finger_tip, name=f"/World/Debug/{finger_path.split('/')[-1]}_tip")
   
    #get_joint_state(gripper, offset_in_local_frame=(0, 0, 0))
    #pos1, pos2 = get_tip_position(stage, finger_path, offset_thumb)
   
    #draw_debug_marker(pos1, name=f"/World/Debug/{finger_path.split('/')[-1]}_center1")
    #draw_debug_marker(pos2, name=f"/World/Debug/{finger_path.split('/')[-1]}_center2")
    time.sleep(0.01)


 