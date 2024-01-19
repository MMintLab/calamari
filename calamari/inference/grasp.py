import numpy as np
import copy
from transforms3d.quaternions import axangle2quat, qmult
from transforms3d.euler import euler2quat
from l4c_rlbench.rlbench.environment import Environment
from calamari.inference.utils.utils import transform_mesh
from rlbench.backend.task import Task
def grasp_pcd(env: Environment, task):
    print("grasp_pcd")
    p =   env._prev_task.grasp_target.get_pose()
    r_euler =   env._prev_task.grasp_target.get_orientation()
    print("get pose", p, r_euler)

    pcd = transform_mesh(env._prev_task.handle_pcd, p[np.newaxis,:], device = 'cpu', from_euler=False, return_numpy=True)
    tool_center = np.mean(pcd[0], axis=0) # squeeze pcd
    print("tool center", tool_center)

    # r_quat = euler2quat(r_euler[0], r_euler[1] + np.pi/2 , r_euler[2], "rxyz")  # output = w, x, y, z
    r_quat = euler2quat(0, np.pi/2, r_euler[2] + np.pi/2, "sxyz")  # output = w, x, y, z

    # [0.17590057 0.18112557 0.87000866]

    grasp_point = np.array([tool_center[0],
                            tool_center[1],
                            tool_center[2],
                             r_quat[1], r_quat[2], r_quat[3], r_quat[0]])
    print(grasp_point)

    # ## grasp the tool - 1. approach
    offset = np.array([0., 0., 0.10, 0., 0., 0., 0.])
    action =   grasp_step(target=grasp_point + offset, gripper=1.0)
    print("action", action)

    obs, _, _ =  task.step(action)

    action =   grasp_step(target=grasp_point , gripper=1.0)
    obs, _, _ =  task.step(action)

    ## grasp the tool - 2. grasp
    action = grasp_step(target=grasp_point, gripper=0.0)
    obs, _, _ = task.step(action)

    print("Success grasping")
    return obs
def grasp(env: Environment, task):
    p =   env._prev_task.grasp_target.get_pose()
    r_euler =   env._prev_task.grasp_target.get_orientation()
    # r_quat = euler2quat(r_euler[0], r_euler[1] + np.pi/2 , r_euler[2], "rxyz")  # output = w, x, y, z
    r_quat = euler2quat(0, np.pi/2, r_euler[2] + np.pi/2, "sxyz")  # output = w, x, y, z

    grasp_point = np.array([p[0],
                            p[1],
                            p[2],
                             r_quat[1], r_quat[2], r_quat[3], r_quat[0]])

    ## grasp the tool - 1. approach
    offset = np.array([0., 0., 0.10, 0., 0., 0., 0.])
    action =   grasp_step(target=grasp_point + offset, gripper=1.0)
    print("action", action)
    # action[2] = 0.8
    obs, _, _ =  task.step(action)

    action =   grasp_step(target=grasp_point , gripper=1.0)
    obs, _, _ =  task.step(action)

    ## grasp the tool - 2. grasp
    action = grasp_step(target=grasp_point, gripper=0.0)
    obs, _, _ = task.step(action)

    print("Success grasping")
    return obs


def grasp_step( target, gripper):
    arm = copy.copy(target)
    q2 = axangle2quat([0, 1, 0], np.pi / 2)
    r = qmult(q2, target[3:7])
    arm[3] = r[0]
    arm[4] = r[1]
    arm[5] = r[2]
    arm[6] = r[3]

    gripper = [gripper]  # Always open
    ee_goal = np.concatenate([arm, gripper], axis=-1)
    return ee_goal