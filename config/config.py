import itertools
import json

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

        self.B = 16
        self.device = 'cuda'
        self.dim_ft = 64 # 32
        self.seed = 42
        self.len = 70 # number of training data

        self.train_idx = np.arange(0,50)
        self.test_idx = np.arange(140,150)
        self.train_s = 10

        # self.n_idx = self.get_negative_idxs()
        self.g_mat = self.get_gamma_mat()

        ## Different Frames ##
        self.camera_config = json.load(open('dataset/config/camera_config.json', 'r'))
        R = np.array(self.camera_config["RT"])[:3,:3]
        
        intrinsics = np.array(self.camera_config["K"])
        extrinsics = np.array(self.camera_config["RT"])
        C = np.expand_dims(extrinsics[:3, 3], 0).T

        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        self.camera_proj = np.matmul(intrinsics, extrinsics)
        
        self.tab_scale = (110,110)
        self.tab_offset = (0.4, 0.6)
        self.table_h = 0.75 #496

    def world_to_camera(self, cnt_pts):
        cnt_pts_ = np.ones((4, cnt_pts.shape[0]))
        cnt_pts_[:3, :] = cnt_pts.T

        front_img = self.camera_proj @ cnt_pts_
        return front_img

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