"""Ravens main training script."""

import os
import json
import hydra
from PIL import Image
import copy
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment

import logging
import os.path
from argparse import ArgumentParser
import time
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from l4c_rlbench.rlbench.environment import Environment
from calamari.cfg.task_mpc_configs import TaskMPCConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.const import *
from rlbench.backend import utils as rlbench_utils

from calamari.inference.grasp import grasp_pcd as grasp
from calamari.inference.rigid_dynamics import RigidDynamics
from calamari.cfg.config import MppiConfig



# from language4contact.pretrained import PretrainedPolicy

TASKMPCCONFIG = TaskMPCConfig()
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--cfg", "-c", type=str, help="train_config")
parser.add_argument("--variation", "-v", type=int, default= 0, help="train_config")
parser.add_argument("--logdir", "-l", type=str, default= '', help="train_config")

args = parser.parse_args()

# Configs.
with open(args.cfg) as f:
    train_config = yaml.load(f, Loader=yaml.SafeLoader)
task_config = TASKMPCCONFIG.task_mpc_configs[train_config["task"]]  # TODO: support multiple task

# numpy random seed.
# random.seed(TASKMPCCONFIG.mpc_seed)
# np.random.seed(TASKMPCCONFIG.mpc_seed)

# reset from log
live_demos = False
DATASET = '' if live_demos else 'dataset/'


# Turn off camera.
cam_config = CameraConfig()
cam_config.set_all(False)

# Policy.
logdir = args.logdir if len(args.logdir)>0 else train_config["logdir"]
d_idx = train_config["txt_idx"]
print("Loaded Policy")
mode = 'test'
ckpt = 'best.ckpt'
vcfg = {'agent': 'cliport',
        "mode": 'test',
        'n_demos':100,
        'test_n_demos':25,
        'checkpoint_type':'test_best',
        'data_dir':'dataset',
        'eval_task': 'sweep_to_dustpan1',  # no variation
        # 'eval_task': 'wipe_desk', # no variation
        'model_path': '../cliport/exps'}
N_goal = 3
offset = 0.05 #0.1
cfg = MppiConfig()

def main():
    # Load train cfg
    tcfg = utils.load_hydra_config('../cliport/cliport/cfg/train.yaml')
    # Initialize environment and task.
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(wrist_camera=cam_config,
                                     left_shoulder_camera=cam_config,
                                     right_shoulder_camera=cam_config,
                                     overhead_camera=cam_config),
        dataset_root=DATASET,
        headless=False) # TODO: change this to CoppeliaSimEnvironment

    results = []
    records = []


    for i in range(0, 25):
        # Simulation Setup.
        # if i not in [127, 141]:
        #     continue
        st = time.time()
        records = []
        contact_records = []
        goal_steps = 1

        # Reset.
        task_env = env.get_task(task_config['tasks'][train_config["ttm_idx"]])
        task_env.set_variation(args.variation)
        d = task_env.get_demos(1, live_demos=False, random_selection=False, from_episode_number=i)[0]
        descriptions, obs_init = task_env.reset_to_demo(d)
        for r in task_env._task.render:
            r(True)

        task_env._task.tool_dynamics = RigidDynamics(env._prev_task.tool, env._prev_task.target, cfg, policy = None, init_obs = obs_init)

        try:
            obs_init = grasp(env, task_env)
        except:
            pass
        records.append(obs_init)

        # Choose eval mode and task.
        eval_task = vcfg['eval_task']

        # Load eval dataset.
        # dataset_type = vcfg['type']
        print(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"))
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                       tcfg,
                                       n_demos=vcfg['test_n_demos'],
                                       augment=False)

        all_results = {}
        name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

        # Save path for results.
        # json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"



        model_file = os.path.join(vcfg['model_path'], name + '-train', 'checkpoints', ckpt)
        #
        # if not os.path.exists(model_file) or not os.path.isfile(model_file):
        #     print(f"Checkpoint not found: {model_file}")
        #     continue



        # Initialize agent.
        agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")

        # record = vcfg['record']['save_video']
        # n_demos = vcfg['n_demos']

        # Run testing and save total rewards with last transition info.
        episode, seed = ds.load(0)
        _, _, _, info = episode[0]

        # goal = episode[-1]
        # total_reward = 0
        np.random.seed(seed)

        # set task
        # task_name = vcfg['eval_task']
        # task = tasks.names[task_name]()

        dd = rlbench_utils.float_array_to_rgb_image(obs_init.front_depth, scale_factor=DEPTH_SCALE)
        front_depth = rlbench_utils.image_to_float_array(dd,DEPTH_SCALE)
        near = obs_init.misc['front_camera_near']
        far = obs_init.misc['front_camera_far']
        front_depth_m = near + front_depth * (far - near)

        obs = {'color': obs_init.front_rgb[np.newaxis, ...],
                'depth': front_depth_m[np.newaxis, ...],
        }


        # print(obs)
        height = obs_init.gripper_pose[2]

        # Start recording video (NOTE: super slow)
        result_i = 0
        for _ in range(N_goal):
            act = agent.act(obs, info, goal = None)
            lang_goal = descriptions[0]  # info['lang_goal']
            print("lang_goal", lang_goal)
            # print(obs_init.gripper_pose, act)

            for pp_idx in range(2):
                act_i = act[f'pose{pp_idx}']
                print(act_i)

                # Execute picking primitive.
                prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
                postpick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
                prepick_pose = utils.multiply(act_i, prepick_to_pick)
                postpick_pose = utils.multiply(act_i, postpick_to_pick)
                # print( prepick_pose, postpick_pose  )
                # breakpoint()


                act_i_rot = R.from_quat(act_i[1]).as_euler('XYZ')

                p1 = R.from_quat(act[f'pose{1}'][1])
                p2 = R.from_euler('x', 180, degrees=True)
                p2.as_quat()

                p3 = p1.as_matrix() * p2.as_matrix()
                p4 = R.from_matrix(p3).as_quat()


                # state_next = [act_i[0][0], act_i[0][1], act_i[0][2], act_i_rot[0], act_i_rot[1], act_i_rot[2]]
                try:
                    state_next = [act_i[0][0] + np.sign(env._prev_task.grasp_target.get_pose()[-1]) * offset,
                                  act_i[0][1],
                                  height,
                                  p4[0], p4[1], p4[2], p4[3] , 0]
                except:
                    state_next = [act_i[0][0],
                                  act_i[0][1],
                                  0.775,
                                  p4[0], p4[1], p4[2], p4[3], 0]
                print("state next", state_next)



                state_next[:3] = np.clip(state_next[:3],
                                     [task_env._scene._workspace_minx + 0.01, task_env._scene._workspace_miny + 0.01,
                                      task_env._scene._workspace_minz + 0.01],
                                     [task_env._scene._workspace_maxx - 0.01, task_env._scene._workspace_maxy - 0.01,
                                      task_env._scene._workspace_maxz])

                try:
                    if pp_idx == 0:
                        state_next_lift = copy.copy(state_next)
                        state_next_lift[2] += 0.1
                        task_env.step(state_next_lift)

                    obs_raw, r, t = task_env.step( np.array( state_next))
                except:
                    pass

                obs_raw.sim_cost = env._prev_task.get_score(t)

                dd = rlbench_utils.float_array_to_rgb_image(obs_raw.front_depth, scale_factor=DEPTH_SCALE)
                front_depth = rlbench_utils.image_to_float_array(dd, DEPTH_SCALE)
                front_depth_m = near + front_depth * (far - near)
                obs = {'color': obs_raw.front_rgb[np.newaxis, ...],
                       'depth': front_depth_m[np.newaxis, ...],
                       }
                result_i = obs_raw.sim_cost
                records.append(obs_raw)

                try:
                    if pp_idx == 1:
                        state_next_lift = copy.copy(state_next)
                        state_next_lift[2] += 0.1
                        task_env.step(state_next_lift)

                    if obs_raw.sim_cost == 100:
                        result_i = 100
                        break
                except:
                    pass

                obs_raw.sim_cost = env._prev_task.get_score(t)

                dd = rlbench_utils.float_array_to_rgb_image(obs_raw.front_depth, scale_factor=DEPTH_SCALE)
                front_depth = rlbench_utils.image_to_float_array(dd, DEPTH_SCALE)
                front_depth_m = near + front_depth * (far - near)
                obs = {'color': obs_raw.front_rgb[np.newaxis, ...],
                       'depth': front_depth_m[np.newaxis, ...],
                       }
                result_i = obs_raw.sim_cost
                records.append(obs_raw)


        np.save(f'out/demo_cliport_{train_config["task"]}_{i}.npy', records)






        # print(f'Lang Goal: {lang_goal}', act)
        # obs, reward, done, info = env.step(act)
        # total_reward += reward # update reward to 0-1 scale
        # print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
        # if done:
        #     break


        print("======================", result_i, "======================")
        results.append(result_i)

    print("result:", results, np.mean(results))


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == '__main__':
    main()
