from scipy.spatial.transform import Rotation as R
import torch
import copy
import numpy as np
import open3d as o3d
from calamari.inference import grasp


# def reset_robot(env, obs, task, controller, reset_pose, task_config):
#     env._robot.gripper.release()
#     env._pyrep.step()
#
#     # Release Sponge.
#     release_pose = np.concatenate([obs.gripper_pose, [1.0]])
#     release_pose[3] = + 0.07
#     obs, _, _ = task.step(reset_pose)
#     env._robot.gripper.release()
#     env._pyrep.step()
#
#     # Reset Position.
#     obs, _, _ = task.step(reset_pose)
#
#     # Set Goal.
#     print("set new goal")
#     prev_cont_goal = copy.copy(controller.cnt_pts)
#     controller.generate_contact_goal(obs, l= task_config["l"])
#
#     # Re-grasp
#     obs = grasp_pcd(env, task)

def get_next_pose(cur_pose, action, return_euler = False):
    '''
    :param cur_pose: tensor [x,y,z,r,p,y]
    :param action: tensor or list or array [x,y,z,r,p,y]
    :return: next_pose : tensor [x,y,z,r,p,y]
    '''
    if len(action.shape) == 1 :
        state_next_r = R.from_euler('XYZ', [cur_pose[3] + action[3],
                                            cur_pose[4] + action[4],
                                            cur_pose[5] + action[5]]).as_quat()
        state_next = [cur_pose[0] + action[0],
                      cur_pose[1] + action[1],
                      cur_pose[2] + action[2],
                      state_next_r[0],
                      state_next_r[1],
                      state_next_r[2],
                      state_next_r[3], 0.0]
        state_next = torch.tensor(state_next)

    elif len(action.shape) == 2 :
        assert action.shape == cur_pose.shape
        # print("action", action)
        # breakpoint()
        state_euler = torch.stack((cur_pose[:, 3] + action[:, 3],
                                            cur_pose[:, 4] + action[:, 4],
                                            cur_pose[:, 5] + action[:, 5]), dim = 1)

        state_next_euler = torch.stack((cur_pose[:, 0] + action[:,0],
                      cur_pose[:, 1] + action[:,1],
                      cur_pose[:, 2] + action[:,2],
                      state_euler[:, 0],
                      state_euler[:, 1],
                      state_euler[:, 2]), dim = 1)

        if return_euler:
            return state_next_euler
        else:
            state_next_r = R.from_euler('XYZ', state_euler).as_quat()
            state_next_r = torch.tensor(state_next_r)
            state_next = torch.stack((cur_pose[:, 0] + action[:,0],
                          cur_pose[:, 1] + action[:,1],
                          cur_pose[:, 2] + action[:,2],
                          state_next_r[:, 0],
                          state_next_r[:, 1],
                          state_next_r[:, 2],
                          state_next_r[:, 3], torch.zeros_like(cur_pose[:,0])), dim = 1)
    else:
        print("action length", len(action.shape), action)
        raise "Raise exception: Wrong action shape"

    return state_next
