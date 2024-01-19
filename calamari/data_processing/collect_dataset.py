from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition,ConditionSet
from rlbench.tasks.push_buttons import PushButtons

import numpy as np
import os
from argparse import ArgumentParser
import pickle

from rlbench.backend.const import *

from calamari.inference.rigid_dynamics import RigidDynamics

from calamari.cfg.task_mpc_configs import TaskMPCConfig

# Generate argparse with task flags


from l4c_rlbench.rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.const import *
import os
from calamari.data_processing.dataset_generator import save_demo, check_and_make
from l4c_rlbench.rlbench.tasks.desk_wipe_h import wipe_heuristic_demo
from calamari.cfg.config import MppiConfig

parser = ArgumentParser()
parser.add_argument("--task", "-t", type=str, help="wipe/sweep/push")
parser.add_argument("--end_idx", "-e", type=int, default= 105, help="train_config")
parser.add_argument("--start_idx", "-s", type=int, default = 0, help="train_config")
parser.add_argument("--variation", "-v", type=int, default = -1, help="train_config")
parser.add_argument("--tool_idx", "-o", type=int, default = -1, help="train_config")

args = parser.parse_args()

cfg = MppiConfig()
TASKCONFIG =  TaskMPCConfig()
TASK = TASKCONFIG.task_mpc_configs[args.task]["tasks"][args.tool_idx]

SAVE_PATH = os.path.abspath( os.path.join(os.path.dirname(__file__), '../../dataset/'))

if __name__ == '__main__':
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

    obs_config = ObservationConfig()
    obs_config.set_all(True)


    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=TASKCONFIG.task_mpc_configs[args.task]['demo_controller'],
            gripper_action_mode=Discrete()),
        dataset_root=DATASET,
        obs_config=ObservationConfig(),
        headless=False)

    env.launch(tool_name=TASKCONFIG.task_mpc_configs[args.task]['tool_name'],
               target_name=TASKCONFIG.task_mpc_configs[args.task]['target_name'])

    i = 0
    while i < args.end_idx:
        # This will keep the random seed the same for each episode

        task = env.get_task(TASK)
        possible_variations = task.variation_count()
        if args.variation == -1:
            variation = np.random.randint(possible_variations)
            variation_path = os.path.join(SAVE_PATH, task.get_name(), VARIATIONS_ALL_FOLDER)
        else:
            variation = args.variation
            variation_path = os.path.join(SAVE_PATH, task.get_name(), VARIATIONS_FOLDER % variation)

        check_and_make(variation_path)

        task = env.get_task(TASK)
        task.set_variation(variation)
        try:
            descriptions, obs = task.reset()
        except:
            continue

        if i < args.start_idx:
            i += 1
            continue
        task._task.tool_dynamics = RigidDynamics(env._prev_task.tool, env._prev_task.target, cfg)

        my_variation_count = variation

        # i += 1
        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)
        episode_path = os.path.join(episodes_path, EPISODE_FOLDER % i)


        if args.task != 'wipe':
            try:
                print("start demo")
                demos = task.get_demos(1, live_demos=live_demos) # from_episode_number=i # -> List[List[Observation]]
                success = True
            except:
                print("failed")
                success = False
                continue

        ## Demonstration we are using are heuristic.
        elif args.task == 'wipe':
            try:
                success, demos = wipe_heuristic_demo(i, env, task)
            except:
                success = False
                continue

        else:
            raise NotImplementedError("Task not implemented")


        if success:
            i += 1
            demos.variation_number = my_variation_count
            save_demo(demos, episode_path, my_variation_count)

            # save description
            front_mask_path = os.path.join(episode_path, 'description.txt')
            with open(front_mask_path, 'w') as f:
                f.write(descriptions[0])



    print('Done')
    env.shutdown()

