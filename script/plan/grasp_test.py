import logging
from argparse import ArgumentParser
import time
import yaml
import random
import numpy as np

from l4c_rlbench.rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
<<<<<<< HEAD
from calamari.calamari.inference.grasp import grasp
=======
from calamari.calamari.inference.grasp import grasp_pcd
>>>>>>> origin
from calamari.calamari.cfg.config import MppiConfig
from rlbench.observation_config import ObservationConfig, CameraConfig
from calamari.calamari.cfg.task_mpc_configs import TaskMPCConfig

<<<<<<< HEAD


TASKMPCCONFIG = TaskMPCConfig()

=======
TASKMPCCONFIG = TaskMPCConfig()


>>>>>>> origin
logger = logging.getLogger(__name__)
cfg = MppiConfig()

parser = ArgumentParser()
parser.add_argument("--cfg", "-c", type=str, help="train_config")
args = parser.parse_args()

# Configs.
with open(args.cfg) as f:
    train_config = yaml.load(f, Loader=yaml.SafeLoader)
task_config = TASKMPCCONFIG.task_mpc_configs[train_config["task"]]  # TODO: support multiple task

# numpy random seed.
random.seed(TASKMPCCONFIG.mpc_seed)
np.random.seed(TASKMPCCONFIG.mpc_seed)


if __name__ == '__main__':

    # Turn off camera.
    cam_config = CameraConfig()
    cam_config.set_all(False)

    # Simulation Setup.
    env = Environment(
        action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(wrist_camera=cam_config,
                                   left_shoulder_camera=cam_config,
                                   right_shoulder_camera=cam_config,
                                     overhead_camera=cam_config),
        headless=False)
    env.launch(task_config['tool_name'], task_config['target_name'])

<<<<<<< HEAD
    st = time.time()
    records = []
    contact_records = []
    goal_steps = 0

    # Reset.
    task_env = env.get_task(task_config['tasks'][train_config["ttm_idx"]])
    env._prev_task.init_task(grasp_target_name=task_config['grasp_target_name'][train_config["ttm_idx"]])

    # grasp
    obs = grasp(env, task_env)
    records.append(obs)
=======
    results = []

    contact_records = []
    goal_steps = 0

    for _ in range(5):
        # Reset.
        task_env = env.get_task(task_config['tasks'][train_config["ttm_idx"]])
        env._prev_task.init_task(tool_name =task_config['tool_name'],
                                 target_name=task_config['target_name'],
                                 grasp_target_name=task_config['grasp_target_name'][train_config["ttm_idx"]])
        descriptions, obs = task_env.reset()
        obs = grasp_pcd(env, task_env)

>>>>>>> origin
