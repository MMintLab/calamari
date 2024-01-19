# Later integrate this with the main data-collection pipeline
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from rlbench.backend.const import *
from rlbench.backend.utils import *
from rlbench.utils import _resize_if_needed
import copy

# from .read_npy import process_demo, check_and_make

def folder2filelist(traj_gt_path, sort = True):
    fn_lst = []
    for (dirpath, dirnames, filenames) in os.walk(traj_gt_path):
        for f in filenames:
            if f.endswith('.png'):
                fn_lst.append( os.path.abspath(os.path.join(dirpath, f)))

    if sort:
        fn_lst.sort()
    return fn_lst


DATASET_NAME = 'push_buttons'
LANG = ('push the red button')

for i in range(150):
    # train/val/test
    if i < 100:
        mode = 'train'
    elif i < 125:
        mode = 'val'
        i = i - 100
    else:
        mode = 'test'
        i = i - 125

    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../dataset/',
                                             DATASET_NAME , 'all_variations', 'episodes'))
    SAVE_DATA_PATH =  os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../dataset/',
                                             DATASET_NAME + f'-{mode}'))


    # read all the files with the name "low_dim_obs.pkl"

    # make path
    action_path = os.path.join(SAVE_DATA_PATH, 'action')
    color_path = os.path.join(SAVE_DATA_PATH, 'color')
    depth_path = os.path.join(SAVE_DATA_PATH, 'depth')
    reward_path = os.path.join(SAVE_DATA_PATH, 'reward')
    info_path = os.path.join(SAVE_DATA_PATH, 'info')
    try:
        os.mkdir(SAVE_DATA_PATH)
        os.mkdir(action_path)
        os.mkdir(color_path)
        os.mkdir(depth_path)
        os.mkdir(reward_path)
        os.mkdir(info_path)
    except:
        pass


    low_dim_obs_path = f'{DATA_PATH}/episode{i}/low_dim_obs.pkl'
    # read the file
    with open(low_dim_obs_path, 'rb') as f:
        low_dim_obs = pickle.load(f)

    rgb_folder_path = f'{DATA_PATH}/episode{i}/rgb'
    depth_folder_path = f'{DATA_PATH}/episode{i}/front_depth'

    traj_rgb_fn = folder2filelist(rgb_folder_path)
    traj_rgb_fn.sort()

    pkl_path = f'{i:06d}' + '-' + str(2 * i) + '.pkl'

    rot_hist = []
    depth_hist = []
    info_hist =[]
    rgb_hist = []
    reward = []

    if len(traj_rgb_fn) > 2:
        traj_rgb_fn = traj_rgb_fn[1:] # pop reset rgb & last obs

    for local_idx, rgb_file_path in enumerate(traj_rgb_fn):
        num = int( rgb_file_path.split('_')[-1][:3] )
        low_obs = low_dim_obs._observations[num]


        ## 1. rgb
        rgb_hist.append( np.array(Image.open(rgb_file_path))[np.newaxis,np.newaxis, ...]) # TODO- check if it is 255 or 1 scale

        ## 2. reward
        if local_idx == len(traj_rgb_fn) -1:
            reward.append(1.)
        else:
            reward.append(0.)

        # ## 4. depth
        # depth_img_path = os.path.join(f'{DATA_PATH}/episode{i}', 'depth.pkl')
        # with open(depth_img_path, 'rb') as f:
        #     depth_img = pickle.load(f)
        # depth_hist.append(depth_img[np.newaxis, np.newaxis, ...])
        depth_img_path = os.path.join(depth_folder_path, f'{num}.png')
        front_depth = image_to_float_array(Image.open(depth_img_path),DEPTH_SCALE)
        near = low_obs.misc['front_camera_near']
        far = low_obs.misc['front_camera_far']
        front_depth_m = near + front_depth * (far - near)
        depth_hist.append(front_depth_m[np.newaxis, np.newaxis, ...])
        # d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth

        ## 3. info
        info_hist.append( {'lang_goal': LANG, 'camera_info': [{
        'image_size': (256,256),
        'intrinsics':  low_obs.misc['front_camera_intrinsics'],
        'position': low_obs.misc['front_camera_extrinsics'][:3,3],
        'rotation': R.from_matrix(low_obs.misc['front_camera_extrinsics'][:3,:3]).as_euler('xyz'),
        'zrange': (near, far),
        'noise': False
        }]} )



        ## 5. action
        if local_idx != 0:
            pose_goal = low_obs.gripper_pose
            rot_goal = pose_goal[-4:]

            p1 = R.from_quat(rot_goal)
            p2 = R.from_euler('x', 180, degrees=True)
            p2.as_quat()

            p3 =  p1.as_matrix() * p2.as_matrix()
            p4 = R.from_matrix(p3)

            # get ee orientation and disdcretize it in 10 deg wise.
            p4_euler = p4.as_euler('zxy', degrees=True)[0] // 10 * 10
            p4_ = R.from_euler('xyz', [0,0, p4_euler], degrees=True)


            rot_goal_ = p4_.as_quat()

            rot_hist.append ({'pose0': (np.array([prev_pose[0], prev_pose[1], 0.75]), np.array([0, 0, 0, 1])),
                            'pose1': (np.array([pose_goal[0], pose_goal[1], 0.75]),
                                      np.array(p4_.as_quat()))} )
            prev_pose = copy.copy(low_obs.gripper_pose)
        else:
            prev_pose = copy.copy(low_obs.gripper_pose)


    rot_hist.append (None)


    # 1. write a pkl file with the name "rot_hist.pkl"
    with open(f'{action_path}/{pkl_path}', 'wb') as f:
        pickle.dump(rot_hist, f)

    with open(f'{color_path}/{pkl_path}', 'wb') as f:
        pickle.dump(np.concatenate(rgb_hist), f)

    with open(f'{depth_path}/{pkl_path}', 'wb') as f:
        pickle.dump(np.concatenate(depth_hist), f)

    with open(f'{reward_path}/{pkl_path}', 'wb') as f:
        pickle.dump(reward, f)

    with open(f'{info_path}/{pkl_path}', 'wb') as f:
        pickle.dump(info_hist, f)