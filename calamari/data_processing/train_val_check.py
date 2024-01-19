from argparse import ArgumentParser
import numpy as np
import os
from language4contact.config.task_policy_configs import TaskConfig
# argument
parser = ArgumentParser()
parser.add_argument("--path", "-p", type=str, help="path to a single task data folder")
args = parser.parse_args()

TC =  TaskConfig()
TASK = 'scoop'
# check if validation set is not in the training set
if __name__ == '__main__':
    cnt_lst = []
    cnt_force = []
    hull_len_hist = []
    train_rgb_lst = []

    # Read train
    train_idxs = TC.task_policy_configs[TASK]["train_idx"]
    for j in train_idxs:
        # folder = 'datacollection/data_h_desk_wipe__'#'sweep_dustpan'
        folder = TC.task_policy_configs[TASK]["data_dir"]
        taskname =f"demo_{TASK}"
        demo = np.load(f'out/{TASK}/{taskname}_{j}.npy', allow_pickle=True)

        # Make directory.
        i = j


        cnt_traj = []
        hull_traj = []
        front_rgb = []
        ee_history2d = []
        ee_historyx = []
        hull_hist = {'idx' : [], 'hull_topdown' : [], 'hull_front' : []}

        # Start reading.
        for i_traj, traj in enumerate(demo):
            rgb = traj.front_rgb
            train_rgb_lst.append(rgb)
    test_rgb = []

    # Read train
    test_idxs = TC.task_policy_configs[TASK]["test_idx"]
    for j in test_idxs:
        # folder = 'datacollection/data_h_desk_wipe__'#'sweep_dustpan'
        folder = TC.task_policy_configs[TASK]["data_dir"]
        taskname = f"demo_{TASK}"
        demo = np.load(f'out/{TASK}/{taskname}_{j}.npy', allow_pickle=True)

        # Make directory.
        # i = j

        cnt_traj = []
        hull_traj = []
        front_rgb = []
        ee_history2d = []
        ee_historyx = []
        hull_hist = {'idx': [], 'hull_topdown': [], 'hull_front': []}

        # Start reading.
        for i_traj, traj in enumerate(demo):
            rgb = traj.front_rgb
            for train_rgb in train_rgb_lst:
                if np.all(rgb == train_rgb):
                    print("same rgb for ", j, i_traj)
                    # break
            # else:
            #     print("different rgb")