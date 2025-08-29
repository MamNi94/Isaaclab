import numpy as np
from isaacsim import SimulationApp
# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": False,"physics": True})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid

from isaacsim.core.utils.prims import get_prim_at_path,create_prim, is_prim_path_valid, get_prim_children,get_prim_at_path
from pxr import UsdPhysics,UsdGeom, Usd, PhysxSchema

from isaacsim.core.utils.bounds import compute_aabb
from omni.usd import get_context


##Grasp Funcitons##
def get_finger_front_half_bounds(finger_path):

    visual_path = finger_path + "/visuals"

    if not is_prim_path_valid(finger_path):
        print(f"❌ Prim not found: {finger_path}")
        return None
    #prim = get_prim_at_path(finger_path)
    aabb =compute_aabb(prim_path = finger_path, bbox_cache = bbox_cache,include_children=False)
    # Assume finger points in +X; adjust for your model
    #print(aabb)
    if aabb is None or len(aabb) != 6:
        print(f"⚠️ Warning: Failed to compute AABB for {finger_path}")
        return None

    min_corner = np.array(aabb[:3])
    max_corner = np.array(aabb[3:])
    center = (min_corner + max_corner) / 2

    # Adjust axis if needed! Assuming +X is "forward"
    front_min = np.array([
        center[0],          # Front half starts at center X
        min_corner[1],
        min_corner[2],
    ])
    front_max = np.array([
        max_corner[0],      # Front half ends at max X
        max_corner[1],
        max_corner[2],
    ])

    return np.array([front_min, front_max])



def check_grasp_success(cube, finger_paths):
    #bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    cube_pos, _ = cube.get_world_pose()
    

    contact_count = 0
    for finger_path in finger_paths:
        bounds = get_finger_front_half_bounds(finger_path)
        if bounds is None:
            continue

        min_bound, max_bound = bounds

        threshhold = 0.01

        if (
            min_bound[0] -threshhold<= cube_pos[0] <= max_bound[0] +threshhold and
            min_bound[1] -threshhold<= cube_pos[1] <= max_bound[1] +threshhold
        ):
            contact_count += 1

    return contact_count >= 2 

###END GRASP FUNCTIONS####


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


floor = DynamicCuboid(
    prim_path="/World/Floor",
    position=[0, 0, -0.63],
    size=1,  # single f, not a list
)


########FIX GRIPPER IN FRAME ###############
fixed_joint_path = "/World/fix_gripper"
create_prim(fixed_joint_path, "PhysicsFixedJoint")
joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/UDRF_Hand_Assembly/Core_Bottom"])
########END FIX GRIPPER IN FRAME  


########FIX Cube IN FRAME ###############
fixed_joint_path = "/World/fix_floor"
create_prim(fixed_joint_path, "PhysicsFixedJoint")
joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/Floor"])
########END FIX Cube IN FRAME  

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    includedPurposes=[UsdGeom.Tokens.default_],
    useExtentsHint=False
)

for finger_path in [
    "/World/UDRF_Hand_Assembly/Finger_left",
    "/World/UDRF_Hand_Assembly/Finger_Right",
    "/World/UDRF_Hand_Assembly/Finger_Thumb"
]:
    prim = get_prim_at_path(finger_path)
    if prim.IsValid():
        children = [child.GetPath().pathString for child in prim.GetChildren()]
        print(f"Children of {finger_path}: {children}")
    else:
        print(f"Prim not found or invalid: {finger_path}")

sim.reset()


for _ in range(100):
    sim.step()


#target_positions = np.array([0.5] * gripper.num_joints)
#gripper.set_joint_positions(target_positions)
#print("Target Postions", target_positions)

#while simulation_app.is_running():
    #sim.step()

finger_paths = [
    "/World/UDRF_Hand_Assembly/Finger_left",
    "/World/UDRF_Hand_Assembly/Finger_Right",
    "/World/UDRF_Hand_Assembly/Finger_Thumb",
]


def reset_episode():
    cube.set_world_pose(position=np.array([0, -0.02, -0.1]))
    gripper.set_joint_positions(np.zeros(gripper.num_joints))
    sim.step()

def random_policy():
    return np.random.uniform(0.0, 0.8, size=gripper.num_joints)

# Training loop
for episode in range(100):
    print(f"Episode {episode}")
    
    reset_episode()
    
    for step in range(50):  # one episode max 50 steps
        # Random action for now
        action = random_policy()
        gripper.set_joint_positions(action)
        
        sim.step()
        
        # Grasp success check (very basic)
        cube_pos, _ = cube.get_world_pose()
        #if cube_pos[2] > 0.1:
        if check_grasp_success(cube,finger_paths):
            print("Grasp Success!")
            successes +=1
            break