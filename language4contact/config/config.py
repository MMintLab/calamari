import itertools
import json
import os.path
from pathlib import Path

import numpy as np
import torch


class Config:
    def __init__(self):

        self.motion = np.array([[[1, 0, 1], [0, 1, 0], [0, 0, 1]],
                          [[1, 0, -1], [0, 1, 0], [0, 0, 1]],
                          [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
                          [[1, 0, 0], [0, 1, -1], [0, 0, 1]],
                          [[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                           [0, 0, 1]],
                          [[np.cos(np.pi / 4), np.sin(np.pi / 4), 0], [-np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                           [0, 0, 1]]])

        # self.options = self.get_contact_options(self.W)
        self.W = 6 # window size
        self.N = 50 # counter example numbers
        self.epoch = 30000
        self.gamma = 0.96

        self.B = 30
        self.device = 'cuda'
        self.dim_ft = 512 # 32
        self.seed = 42

        self.train_idx = np.concatenate([np.arange(0,40), np.arange(50,200)])
        # self.train_idx = np.arange(0,45)
        self.test_idx = np.arange(45,50)
        self.train_s = 10

        # self.n_idx = self.get_negative_idxs()
        self.g_mat = self.get_gamma_mat()

        ## Different Frames ##
        json_path = os.path.join(Path( os.path.dirname(__file__)).parents[1],
                                  'dataset/config/camera_config.json')
        self.camera_config = json.load(open(json_path, 'r'))
        R = np.array(self.camera_config["RT"])[:3,:3]
        
        intrinsics = np.array(self.camera_config["K"])
        extrinsics = np.array(self.camera_config["RT"])
        C = np.expand_dims(extrinsics[:3, 3], 0).T

        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        self.camera_proj =  np.matmul(intrinsics, extrinsics)
        self.contact_seq_l = 4
        self.txt_cmd = "Use the sponge to clean up the dirt."
        self.data_dir = 'dataset/heuristics_0228'
        self.contact_folder = 'contact_front'
        self.heatmap_type = 'huy' #chefer
        self.heatmap_size = (224, 224)





    def world_to_camera(self, cnt_pts):
        cnt_pts_ = np.ones((4, cnt_pts.shape[0]))
        cnt_pts_[:3, :] = cnt_pts.T

        front_img = self.camera_proj @ cnt_pts_
        front_img =front_img[:2, :] / front_img[2, :]
        front_img = np.round(front_img).astype(int)
        front_img = np.clip(front_img, 0, 255)
        front_img = front_img.transpose(1, 0)
        return front_img

    def world_to_camera_batch(self, cnt_pts: torch.tensor, mask: torch.tensor ,img_size: list) -> np.ndarray:
        '''
        :param cnt_pts: Padded pointcloud (B, N, 3). Zero rgb value for null points
        '''

        assert len(cnt_pts.shape) == 3
        B, N, _ = cnt_pts.shape
        cnt_pts_ = torch.ones((B, 4, N))
        cnt_pts_[:, :3, :] = cnt_pts.transpose(2, 1)

        front_pxls = torch.tensor(self.camera_proj).to(self.device).float() @ cnt_pts_.clone().detach().to(self.device).float()
        front_pxls = front_pxls[:, :2, :] / front_pxls[:, 2, :].unsqueeze(1)
        front_pxls = torch.round(front_pxls)
        front_pxls = torch.clip(front_pxls, 0, 255)
        front_pxls = front_pxls.transpose(2, 1).long()
        # front_pxls = front_pxls.detach().cpu().numpy()

        imgs = torch.zeros(img_size).to(self.device)
        for b in range(B):
            iidx, _ = torch.where( mask[b, :, :] == torch.ones_like(mask[b, :, :]))
            torch.ones_like(front_pxls[b, iidx, 0]).to(self.device)
            imgs[ b, front_pxls[b,iidx,0], front_pxls[b,iidx,1]] = torch.ones_like(front_pxls[b,iidx,0]).to(self.device).float()
        return imgs.detach().cpu().numpy()




    # def _transform(self, coords, trans):
    #     h, w = coords.shape[:2]
    #     coords = np.reshape(coords, (h * w, -1))
    #     coords = np.transpose(coords, (1, 0))
    #     transformed_coords_vector = np.matmul(trans, coords)
    #     transformed_coords_vector = np.transpose(
    #         transformed_coords_vector, (1, 0))
    #     return np.reshape(transformed_coords_vector,
    #                       (h, w, -1))
    # def _pixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
    #     h, w = pixel_coords.shape[:2]
    #     pixel_coords = np.concatenate(
    #         [pixel_coords, np.ones((h, w, 1))], -1)
    #     world_coords = self._transform(pixel_coords, cam_proj_mat_inv)
    #     world_coords_homo = np.concatenate(
    #         [world_coords, np.ones((h, w, 1))], axis=-1)
    #     return world_coords_homo
    # def camera_to_world(self, img_cnt_pts):
    #
    #     cnt_pts_ = np.ones((3, img_cnt_pts.shape[0]))
    #     cnt_pts_[:2, :] = img_cnt_pts.T
    #
    #
    #     print(cnt_pts_.shape)
    #     cam_proj_mat_homo = np.concatenate(
    #         [self.camera_proj, [np.array([0, 0, 0, 1])]])
    #     cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    #     world_coords_homo = np.expand_dims(self._pixel_to_world_coords(
    #         pc, cam_proj_mat_inv), 0)
    #
    #     world_coords = world_coords_homo[..., :-1][0]
    #
    #     world_pts = np.linalg.pinv( self.camera_proj) @ cnt_pts_
    #     world_pts = world_pts / np.mean(world_pts[3,:])
    #     world_pts = world_pts.T
    #     print("world pts", world_pts)
    #     return world_pts[:,:3]


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