import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R
from pyrep.objects.object import Object
from calamari.cfg.config import MppiConfig
import l4c_rlbench as utils
import matplotlib.pyplot as plt

# TODO : numpy array function -> tensor function
class RigidDynamics:
    def __init__(self, tool : Object, target: Object, cfg : MppiConfig, policy = None, init_obs = None):
        # Nominal Geometry
        self.geom = tool
        self.target = target

        self.geom.is_collidable = True
        self.target.is_collidable = True

        self.contact_goal = None
        self.contact_goal_center = None
        self.cfg = cfg
        self.policy = policy
        self.device = cfg.device
        self.init_obs = init_obs





    def estimate_binary_contact_wo_table(self, pcd: torch.tensor, front_pxls, thres_p = 0.05) -> (np.ndarray, np.ndarray):
        # TODO : estimate contact with general objects ?
        '''
        pcd: pointcloud B X N X 3
        self.init_obs.front_depth : 2D front depthmap
        front_pxls : 2D pixel coordinates of tools
        bottom_env_depth : dynamics's 2D mask

        '''

        # Pointcloud size = B x N x 3
        if len(pcd.shape) == 2:
            pcd = pcd.unsqueeze(0) #[np.newaxis, :, :]
        assert len(pcd.shape) == 3

        # detect contact using z offset
        # TODO : step 1- extract the bottom 1cm of the tool pcd
        # get min_z.
        bottom_thres = torch.amin(pcd[...,2]).repeat(pcd.shape[0], pcd.shape[1])+ 0.01
        bottom_binary = torch.where(pcd[..., 2] < bottom_thres, torch.ones_like(pcd[..., 2]), torch.zeros_like(pcd[..., 2]))
        bottom_binary = bottom_binary.unsqueeze(2).repeat((1, 1, 2)) #[..., np.newaxis] # B x N X 1
        # bottom_binary = bottom_binary.reshape(front_pxls.shape[0], front_pxls.shape[1])

        # step 2 - Project it back to the camera plane. 2D mask of the bottom of the tool.
        bottom_obj_2d_mask, bottom_obj_2d_depth, bottom_obj_2d_pcdidx  = self.policy.Config.pxl_labels_to_image(front_pxls,
                                                                    depth = pcd[...,2],
                                                                    mask=bottom_binary,
                                                                    img_size=(bottom_binary.shape[0], self.init_obs.front_depth.shape[0], self.init_obs.front_depth.shape[1]))

        bottom_env_2d_depth = bottom_obj_2d_mask * self.init_obs.front_depth[np.newaxis, ...]

        # contact == 1.
        contact_2d_mask = np.where( bottom_obj_2d_depth < bottom_env_2d_depth,
                               np.ones_like(bottom_obj_2d_depth),
                               np.zeros_like(bottom_obj_2d_depth))

        # Penetration == 2.
        contact_2d_mask = np.where( bottom_obj_2d_depth < bottom_env_2d_depth - 0.01,
                               np.ones_like(bottom_obj_2d_depth)*2,
                               contact_2d_mask)
        # print(contact_2d_mask.shape, front_pxls.shape, bottom_obj_2d_mask.shape, pcd.shape, self.init_obs.front_depth.shape)

        return contact_2d_mask, bottom_obj_2d_pcdidx
        # print(contact_2d_mask.shape)
        # breakpoint
        # contact_pxls = front_pxls * bottom_binary # Contact candidates in pixel location. (0,0,0) is null.



        # step 3 - overlay with the depthmap & get the corresponding depth
        # env_depth_at_contact = np.zeros_like(bottom_binary)
        # for b in range(front_pxls):
        #     env_depth_at_contact[] = self.init_obs.front_depth[]
        # torch.tensor(self.init_obs.front_depth[]
        # env_depth_pcd = torch.tensor(self.init_obs.front_depth[np.newaxis, ...] * bottom_mask)
        # print( self.init_obs.front_depth.shape, bottom_mask.shape, env_depth.shape)


        # compare step 3 and the depthmap
        # z_thes = self.cfg.table_height + self.cfg.cnt_eps
        # contact_binary = torch.where(pcd[..., 2] < env_depth, torch.ones_like(pcd[..., 2]), torch.zeros_like(pcd[..., 2]))
        # contact_binary = torch.where(pcd[..., 2] < bottom_env_depth, torch.ones_like(pcd[..., 2]), torch.zeros_like(pcd[..., 2]))
        #
        # return contact_binary.unsqueeze(2) #[..., np.newaxis] # B x N X 1


    def estimate_binary_contact(self, pcd: torch.tensor) -> np.ndarray:
        # TODO : estimate contact with general objects ?

        # Pointcloud size = B x N x 3
        if len(pcd.shape) == 2:
            pcd = pcd.unsqueeze(0) #[np.newaxis, :, :]
        assert len(pcd.shape) == 3

        z_thes = self.cfg.table_height + self.cfg.cnt_eps
        contact_binary = torch.where(pcd[..., 2] < z_thes, torch.ones_like(pcd[..., 2]), torch.zeros_like(pcd[..., 2]))
        return contact_binary.unsqueeze(2) #[..., np.newaxis] # B x N X 1

    def _set_contact_goal(self, img :np.ndarray, pcd :np.ndarray):
        self.contact_img = img
        self.contact_goal = pcd
        self.contact_goal_center = np.mean(pcd, axis=0)



    def angle_normalize(self, x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)
    def rigid_dynamics(self, state, perturbed_action):
        '''
        Input:
            state: tool state [x, y, z, r, p, y]
            perturbed_action: tool action [x, y, z, r, p, y]
        Output:
            next_state: tool state [x, y, z, r, p, y]
        '''
        # Get next state
        state_next = state + perturbed_action
        return state_next
