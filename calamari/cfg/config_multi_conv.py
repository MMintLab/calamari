import itertools
import json
import os.path
from pathlib import Path

import numpy as np
import torch
# from .task_policy_configs import TaskConfig
from .task_policy_configs_v2 import TaskConfig

class Config:
    def __init__(self, camera_config):

        # self.options = self.get_contact_options(self.W)
        self.W = 6 # window size
        self.N = 50 # counter example numbers
        self.epoch = 2000
        self.gamma = 0.96

        self.B = 300
        self.device = 'cuda'
        self.dim_emb = 512
        self.dim_ft = 32 # 32
        self.seed = 42

        self.train_s = 10

        self.g_mat = self.get_gamma_mat()
        self.tab_scale = (110,110)
        self.tab_offset = (0.4, 0.6)
        self.table_h = 0.5 #496

        # Data dir 
        # TODO: Update dataset_temporal to use the data config dictionary 
        self.contact_seq_l = 4
        self.max_sentence_l = 16
        self.task_confg = TaskConfig()
        self.dataset_config = self.task_confg.task_policy_configs

        self.heatmap_size = (256, 256) #224,224) # Resize the heatmap by this size
        self.heatmap_type = 'huy' # 'chefer'
        self.contact_folder = 'contact_front'
        self.contact_seq_l = 4
        self.aug_num = 20


        ## Different Frames ##
        if camera_config is None:
            json_path = os.path.join(Path( os.path.dirname(__file__)).parents[1],
                                    'calamari/cfg/camera_config.json')
            self.camera_config = json.load(open(json_path, 'r'))
            self.camera_proj = self.get_camera_proj(self.camera_config)
        else:
            print("loaded Real camera configs")
            self.camera_config = camera_config
            # self.camera_proj = camera_config["P"]
            self.camera_proj = self.get_camera_proj(self.camera_config)

    def get_camera_proj(self, camera_config):
        R = np.array(camera_config["RT"])[:3,:3]
        
        intrinsics = np.array(camera_config["K"])
        extrinsics = np.array(camera_config["RT"])
        C = np.expand_dims(extrinsics[:3, 3], 0).T

        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)

        return np.matmul(intrinsics, extrinsics)





    def world_to_camera(self, cnt_pts):
        cnt_pts_ = np.ones((4, cnt_pts.shape[0]))
        cnt_pts_[:3, :] = cnt_pts.T

        front_img = self.camera_proj @ cnt_pts_
        front_img =front_img[:2, :] / front_img[2, :]
        front_img = np.round(front_img).astype(int)
        front_img = np.clip(front_img, 0, 255)
        front_img = front_img.transpose(1, 0)
        return front_img

    def pcd_to_camera(self, cnt_pts):
        assert len(cnt_pts.shape) == 3
        B, N, _ = cnt_pts.shape
        cnt_pts_ = torch.ones((B, 4, N))
        cnt_pts_[:, :3, :] = cnt_pts.transpose(2, 1)

        front_pxls = torch.tensor(self.camera_proj).to(self.device).float() @ cnt_pts_.clone().detach().to(
            self.device).float()
        front_pxls = front_pxls[:, :2, :] / front_pxls[:, 2, :].unsqueeze(1)
        front_pxls = torch.round(front_pxls)
        front_pxls = torch.clip(front_pxls, 0, 255)
        front_pxls = front_pxls.transpose(2, 1).long()
        front_pxls = front_pxls[:,:,[1,0]]

        return front_pxls

    def pxl_labels_to_image(self, front_pxls: torch.tensor, depth: torch.tensor, mask: torch.tensor ,img_size: list):
        '''
        :param cnt_pts: Padded pointcloud (B, N, 3). Zero rgb value for null points
        '''
        # front_pxls = self.pcd_to_camera(cnt_pts)
        # front_pxls = front_pxls.detach().cpu().numpy()

        imgs = np.zeros(img_size)#.to(self.device)
        imgs_b = np.zeros(img_size) #.to(self.device)
        contact_pcd_idx = []
        # imgs_idx = np.ones(img_size) #.to(self.device) # corresponding image index for each pixel

        for b in range(mask.shape[0]):
            iidx, _ = np.where( mask[b, :, :] == torch.ones_like(mask[b, :, :]))
            imgs[ b, front_pxls[b,iidx,0], front_pxls[b,iidx,1]] = np.ones_like(front_pxls[b,iidx,0]) #.to(self.device).float()
            imgs_b[ b, front_pxls[b,iidx,0], front_pxls[b,iidx,1]] = depth[b, iidx] #.to(self.device).float()
            contact_pcd_idx.append(iidx)
            # imgs_idx [b, front_pxls[b,iidx,0], front_pxls[b,iidx,1]] = iidx

        # return imgs, imgs_b, contact_pcd_idx




    def contact_frame_to_world(self, cnt_pxls):
        ## cnt_pxls = N x 2 where N  = contact points, 2 = i , j
        cnt_pts = np.zeros((cnt_pxls.shape[0] , 3))
        # cnt_pts[:,0] = (cnt_pxls[:,0] - 30) / 150
        # cnt_pts[:,1] = (cnt_pxls[:,1] - 70) / 150
        cnt_pts[:,0] = cnt_pxls[:,0]/self.tab_scale[0] - self.tab_offset[0]
        cnt_pts[:,1] = cnt_pxls[:,1]/self.tab_scale[1] - self.tab_offset[1]

        cnt_pts[:,2] = self.table_h
        return cnt_pts # 2 X N
        
    def get_negative_idxs(self):
        n_idx = list (itertools.permutations(np.arange(self.W)))
        n_idx.pop(0)
        return n_idx


    def set_gamma_mat(self, W = None, mode = 'exp'):
        if W is None:
            W = self.W

        g_mat = torch.ones((W , 1))
        
        if mode == 'exp':
            for i in range(W):
                g_mat[i,:] = self.gamma ** i
        elif mode == 'linear':
            g_mat = torch.linspace(1, 0, W+1).view(-1,1)
        self.g_mat = g_mat.to(self.device)

    def get_gamma_mat(self, W = None):
        if W is None:
            W = self.W

        g_mat = torch.ones((W , 1))
        for i in range(W):
            g_mat[i,:] = self.gamma ** i
        return g_mat.to(self.device)
        

    def get_contact_options(self, W):
        options = np.zeros((W ** 6, W))
        for i in range(W):
            if i == 0:
                options[:, i] = np.arange(0, W ** 6) // (W ** 5)
            else:
                options[:, i] = (np.arange(0, W ** 6) % (W ** (6 - i - 1))) // (W ** (6 - i - 2))

        return options