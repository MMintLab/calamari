import copy
import logging
import os.path
from argparse import ArgumentParser
import time
import yaml
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from rlbench.environment import Environment
from l4c_rlbench.rlbench.tasks.desk_wipe_h import WipeDesk, WipeDeskWb,WipeDeskHd
import calamari.inference.utils.utils as utils
from calamari.inference.utils.dynamics import get_next_pose
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from calamari.inference.grasp import grasp_pcd as grasp
from calamari.inference.rigid_dynamics import RigidDynamics
from calamari.cfg.config import MppiConfig
from calamari.cfg.task_mpc_configs import TaskMPCConfig

from calamari.pretrained import PretrainedPolicy
from controller import Controller
from rlbench.observation_config import ObservationConfig, CameraConfig

TASKMPCCONFIG = TaskMPCConfig()
logger = logging.getLogger(__name__)
cfg = MppiConfig()

parser = ArgumentParser()
parser.add_argument("--cfg", "-c", type=str, help="train_config")
parser.add_argument("--variation", "-v", type=int, default= 0, help="train_config")
parser.add_argument("--logdir", "-l", type=str, default= '', help="train_config")
parser.add_argument("--ttm_idx", "-t", type=int, default= -1, help="train_config")
parser.add_argument("--txt_idx", "-tx", type=int, default= -1, help="train_config")
parser.add_argument("--model_type", "-m", type=str, default= "", help="train_config")
parser.add_argument("--task", "-k", type=str, default= "", help="train_config")
parser.add_argument("-s", type=int, default= "", help="start config index (train object 125, test object 0)")

args = parser.parse_args()

# Configs.
if args.cfg is not None:
    with open(args.cfg) as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)

# numpy random seed.
# random.seed(TASKMPCCONFIG.mpc_seed)
# np.random.seed(TASKMPCCONFIG.mpc_seed)

# reset from log
live_demos = False
DATASET = '' if live_demos else 'dataset/'

if __name__ == '__main__':

    # Turn off camera.
    cam_config = CameraConfig()
    cam_config.set_all(False)


    # Policy.
    ttm_idx = args.ttm_idx if args.ttm_idx >-1 else train_config["ttm_idx"]
    logdir = args.logdir if len(args.logdir)>0 else train_config["logdir"]
    task = args.task if len(args.task)>0 else train_config["task"]
    d_idx = args.txt_idx if args.txt_idx >-1 else train_config["txt_idx"]

    task_config = TASKMPCCONFIG.task_mpc_configs[task]  # TODO: support multiple task

    policy = PretrainedPolicy(logdir, 'huy')
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(wrist_camera=cam_config,
                                     left_shoulder_camera=cam_config,
                                     right_shoulder_camera=cam_config,
                                     overhead_camera=cam_config),
        dataset_root=DATASET,
        headless=False)
    env.launch(task_config['tool_name'], task_config['target_name'])
    results = []


    for i in range(args.s, args.s + 25):

        st = time.time()
        records = []
        contact_records = []
        goal_steps = 1

        # Reset.
        task_env = env.get_task(task_config['tasks'][ttm_idx])
        task_env.set_variation(args.variation)
        d = task_env.get_demos(1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        descriptions, obs_init = task_env.reset_to_demo(d)
        task_env._task.tool_dynamics = RigidDynamics(env._prev_task.tool, env._prev_task.target, cfg, policy = policy, init_obs = obs_init)


        for r in task_env._task.render:
            r(True)


        # Controller.
        controller = Controller(env, policy, env._prev_task.tool_dynamics)
        print("Loaded Controller")

        obs = task_env.get_observation()
        records.append(obs)

        # Grasp.
        if not env._prev_task.grasp_target_name is None:
            obs = grasp(env, task_env)
            records.append(obs)
            controller.contact = True
        contact_records.append(obs)


        # Initial Goal.
        controller.generate_next_goal([obs], l=descriptions[d_idx], mode='soft')
        prev_cont_goal = copy.copy(controller.cnt_pts)


        # Initialize.
        total_reward = 0
        robot_initialize = False
        reached = False
        tick = 0
        tot_tick = 0
        cost = 1
        cnt_non_valid_goal = 0


        # MPC.
        obs.sim_cost = 0
        completed = False
        while not completed:
            # TODO: if no contact, then directly go to the goal

            # controller.cur_cnt_pts = obs.contact_info["points"]
            # Set a goal when (goal reached) or (takes too long) or (non valid goal)
            # Task Success

            if controller.contact:
                # make cost 0 when satisfied completed conditions.

                if env._prev_task.get_completed(tot_tick, goal_steps):
                    cost = 0

                # Subgoal success
                print(cost, env._prev_task.subgoal_success_threshold())
                if cost < env._prev_task.subgoal_success_threshold() or tick > 20:
                    tick = 0
                    cnt_non_valid_goal = 0
                    goal_steps += 1

                    # check if the task is completed
                    completed = env._prev_task.get_completed(tot_tick, goal_steps)

                    if completed:

                        # Cur state.
                        rob_state = obs.gripper_pose
                        rob_state_r = R.from_quat(rob_state[3:7]).as_euler('XYZ')
                        rob_state_ = [rob_state[0], rob_state[1], rob_state[2], rob_state_r[0], rob_state_r[1],
                                      rob_state_r[2]]

                        state = env._prev_task.tool.get_pose()  # (X,Y,Z,Qx,Qy,Qz,Qw)
                        state_r = R.from_quat(state[3:7]).as_euler('XYZ')
                        state_ = [state[0], state[1], state[2], state_r[0], state_r[1], state_r[2]]

                        try:
                            # Get action from mppi
                            action = env._prev_task.get_final_action(obs)

                            # Step.
                            state_next = get_next_pose(rob_state_, action)
                            obs, r, t = task_env.step(state_next.numpy())
                        except:
                            print("failed to get final action")

                        records[-1].sim_cost = env._prev_task.get_score(t)
                        results.append(records[-1].sim_cost)

                        print("save file with time", time.time() - st, "[s]")
                        np.save(f'out/demo_{task}_{i}.npy', records)

                        break


                    # If not completed, generate next goal
                    start = time.time()
                    # print("set new goal", cost, tick)
                    prev_cont_goal = copy.copy(controller.cnt_pts)
                    contact_records.append(obs)
                    controller.generate_next_goal(contact_records, l=descriptions[d_idx], mode='soft')
                    cnt_goal_delta = np.linalg.norm(np.mean(controller.cnt_pts, axis=0) - np.mean(prev_cont_goal, axis=0))
                    print("set new goal time", time.time() - start)

                tick += 1
                tot_tick += 1

                # Cur state.
                rob_state = obs.gripper_pose
                rob_state_r = R.from_quat(rob_state[3:7]).as_euler('XYZ')
                rob_state_ = [rob_state[0], rob_state[1], rob_state[2], np.pi, 0, rob_state_r[2]]


                state = env._prev_task.tool.get_pose() #(X,Y,Z,Qx,Qy,Qz,Qw)
                state_r = R.from_quat(state[3:7]).as_euler('XYZ')
                state_ = [state[0], state[1], state[2], state_r[0], state_r[1], state_r[2]]


                # Get action from mppi
                command_start = time.perf_counter()
                actions = controller.ctrl.command(state_)
                time.sleep(1)

                for action in actions:
                    try:
                        action[-1] = -action[-1]
                        action[-2] = action[-2]
                        state_next = get_next_pose(rob_state_, action)

                        state_next[:3] = np.clip(state_next[:3],
                                             [task_env._scene._workspace_minx+ 0.01, task_env._scene._workspace_miny+ 0.01,
                                              task_env._scene._workspace_minz + 0.01],
                                             [task_env._scene._workspace_maxx- 0.01, task_env._scene._workspace_maxy- 0.01,
                                              task_env._scene._workspace_maxz])

                        obs, r, t = task_env.step(state_next.numpy())
                        break
                    except:
                        print("failed to get action")
                        continue

                state = env._prev_task.tool.get_pose() #(X,Y,Z,Qx,Qy,Qz,Qw)
                state_r = R.from_quat(state[3:7]).as_euler('XYZ')
                state_ = [state[0], state[1], state[2], state_r[0], state_r[1], state_r[2]]
                cost = controller.running_cost(torch.tensor(state_).unsqueeze(0), torch.zeros_like(action).unsqueeze(0))

                obs.contact_goal = controller.cnt_img

                obs.cost = cost
                obs.sim_cost = env._prev_task.get_score(t)

                obs.goal_steps = goal_steps
                records.append(obs)


            # If target is close enough, then directly go to the target
            else:
                try:
                    rob_state = obs.gripper_pose
                    state_next = [rob_state[0],rob_state[1],rob_state[2], rob_state[3], rob_state[4], rob_state[5], rob_state[6], 0]
                    obs, r, t = task_env.step( state_next)
                    obs.sim_cost = env._prev_task.get_score(t)
                    obs.contact_goal = controller.cnt_img
                except:
                    print("failed to close gripper")


                # rob_state_r = R.from_quat(rob_state[3:7]).as_euler('XYZ')
                state_next = [controller.dynamics.contact_goal_center[0],
                              controller.dynamics.contact_goal_center[1],
                              controller.dynamics.contact_goal_center[2]-0.01,
                              rob_state[3], rob_state[4], rob_state[5], rob_state[6], 0]

                state_next[:3] = np.clip(state_next[:3],
                                     [task_env._scene._workspace_minx, task_env._scene._workspace_miny,
                                      0.77],
                                     [task_env._scene._workspace_maxx, task_env._scene._workspace_maxy,
                                      task_env._scene._workspace_maxz])

                try:
                    obs, r, t = task_env.step( state_next)
                    obs.sim_cost = env._prev_task.get_score(t)
                    obs.contact_goal = controller.cnt_img
                    records.append(obs)
                except:
                    print("failed to get final action")

                tick += 1
                tot_tick += 1
                completed = env._prev_task.get_completed(tot_tick, goal_steps)
                print("======================completed=====================")
                # print("target", state_next, t, completed)


                # completed = True
            if obs.sim_cost == 100 or completed:
                try:
                    # Get action from mppi
                    action = env._prev_task.get_final_action(obs)

                    # Step.
                    state_next = get_next_pose(rob_state_, action)
                    obs, r, t = task_env.step(state_next.numpy())
                    # visualize image with matplotlib
                except:
                    print("failed to get final action")

                records[-1].sim_cost = env._prev_task.get_score(t)
                results.append(records[-1].sim_cost)

                print(records[-1].sim_cost, ",save file with time", time.time() - st, "[s]")
                np.save(f'out/demo_{task}_{i}.npy', records)
                break




    print("result:", results, np.mean(results[:25]), np.std(results[:25]))
    import csv
    csv_name = 'eval_result_.csv'
    if os.path.exists(csv_name):
        mode = 'a'
    else:
        mode = 'w'

    with open(csv_name, mode, encoding='UTF8', newline='') as f:
        header = ['log', 'val mean', 'val std', 'test mean', 'test std', 'len']
        data = [args.logdir, np.mean(results[:25]), np.std(results[:25]), len(results)]

        writer = csv.writer(f)

        # write the header
        if mode == 'w':
            writer.writerow(header)

        # write the data
        writer.writerow(data)
