
import numpy as np
from isaacsim import SimulationApp
import time

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import SimulationContext
#from isaacsim.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Usd
from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from pxr import UsdGeom, UsdPhysics, PhysxSchema

def detect_joint_limits(gripper, sim, step=0.05, max_angle=np.pi/2):
    limits = []
    dof_names = gripper.dof_names
    n_dofs = len(dof_names)

    print(f"🔍 Detecting limits for {n_dofs} joints...")

    # Get current joint pose as neutral
    base_pose = gripper.get_joint_positions()
    print("BASE POSE", base_pose)
    base_pose = base_pose if base_pose.ndim == 2 else base_pose

    #for idx in range(n_dofs):
    for idx in range(9):
        idx = idx + 3
        joint_name = dof_names[idx]
        print(f"\n▶ Joint {idx}: {joint_name}")

        # Start with the neutral pose
        test_pose = base_pose.copy()
        lower, upper = 0.0, 0.0
        print("TEST POSE",test_pose)
        print("IDX", idx)
        # Negative direction

        
        
        angle = 0.0
        while angle > -max_angle:
            print("ANGLE", angle)
            sim.step()
            angle -= step
            test_pose[0][idx] = angle
            print("TEST POSE LOOP", test_pose)
            gripper.set_joint_positions(test_pose)  # IsaacSim needs shape (1, N)
             
            time.sleep(0.1)

            positions = gripper.get_joint_positions()
            if not np.all(np.isfinite(positions)):
                print(f"  ⚠️ Invalid state at {angle:.2f}, using {angle + step:.2f} as lower limit")
                lower = angle + step + 0.1
                break
            lower = angle

        # Reset to neutral
        sim.step()
        test_pose = base_pose.copy()
        gripper.set_joint_positions(test_pose)
        for _ in range(10): sim.step()

        # Positive direction
        angle = 0.0
        while angle < max_angle:
            sim.step()
            angle += step
            test_pose[0][idx] = angle
            gripper.set_joint_positions(test_pose)
            for _ in range(5): sim.step()

            positions = gripper.get_joint_positions()
            if not np.all(np.isfinite(positions)):
                print(f"  ⚠️ Invalid state at {angle:.2f}, using {angle - step:.2f} as upper limit")
                upper = angle - step -0.1
                break
            upper = angle
            time.sleep(0.1)

        sim.step()
        gripper.set_joint_positions(base_pose) 
        time.sleep(1)
        sim.step()

        print(f"  ✅ Detected limits: [{lower:.2f}, {upper:.2f}]")
        limits.append((lower, upper))

        # Reset pose for next joint
        gripper.set_joint_positions(base_pose[None])
        for _ in range(10): sim.step()

    return limits


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


#Get DOF names to map joints (for debugging)
dof_names = gripper.dof_names
print("DOF names:", dof_names)


########FIX GRIPPER IN FRAME ###############

fixed_joint_path = "/World/fix_gripper"
create_prim(fixed_joint_path, "PhysicsFixedJoint")

joint_prim = get_prim_at_path(fixed_joint_path)
UsdPhysics.FixedJoint(joint_prim).CreateBody0Rel().SetTargets(["/World"])
UsdPhysics.FixedJoint(joint_prim).CreateBody1Rel().SetTargets(["/World/UDRF_Hand_Assembly/Core_Bottom"])
########END FIX GRIPPER IN FRAME  



print(positions)

limits = detect_joint_limits(gripper, sim)
# Run the simulation (optional, for interactive viewing)
while simulation_app.is_running():
    simulation_app.update()

    sim.step()  # Step physics simulation
    print(limits)
    time.sleep(1)

 