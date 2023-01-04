import os 

import math
import torch
import torch.utils.data as data_utils

from utils import *
class DatasetSeq_front_gt_feedback(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train'):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = 'contact_front'
        self.mode = mode 
        
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.data_summary = self.get_data_summary()
        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()
    
    def get_data_summary(self):
        data_summary = {'tot_data_flat': [], 'tot_data_index':{}}
        tot_length = 0
        for i in self.folder_idx:
            folder_path = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'

            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(traj_cnt_path)
            traj_cnt_fn.sort()

            local_idx = range(len(traj_cnt_fn))
            data_summary['tot_data_flat'].append(traj_cnt_fn)
            data_summary['tot_data_index'] = { tot_length + local_idx : (i, local_idx) }

            tot_length += len(traj_cnt_fn)
        return traj_cnt_fn_dic

        def __len__(self):
            return len(data_summary['tot_data_flat'])


    def get_cnt_img_dim(self):
        folder_path1 = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h
    
    
    def _get_gt_cost(self, traj_cnt_lst):

        center_lst = []
        cnt_img_lst = []

        for i in range( len(traj_cnt_lst)):
            cnt_img = traj_cnt_lst[i]
            cnt_img_lst.append(cnt_img)
            
            ## extract center pixel of a mask 
            iidx, jidx = torch.where(cnt_img != 0)
            i_center = math.floor(torch.mean(iidx * 1.0)) * 1.0
            j_center = math.floor(torch.mean(jidx * 1.0)) * 1.0
            
            ## append center list 
            center_lst.append([i_center, j_center])

        ## ground truth cost
        center_lst = torch.tensor(center_lst)
        l2 = torch.norm( center_lst[1:] - center_lst[:-1], dim = -1) 
        
        ## append 0 to the end of tensor
        vel = torch.cat((l2, torch.tensor([0])), dim = 0)
        
        ## cost of each step based on the distance to the goal
        cost = []
        for i in range(len(l2)):
            cost_i = torch.sum(l2[i:])
            cost.append(cost_i)
        cost.append( torch.tensor(0))

        assert len(cnt_img_lst) == len(cost)

        cost_map = []
        for i in range( len(cnt_img_lst)):
            cost_map_i = cost[i] * cnt_img_lst[i]
            cost_map.append(cost_map_i)
        cost_map = torch.amax( torch.stack(cost_map), dim = 0)
        cost_map = cost_map / torch.max(cost_map)

        ## assign high value to the pixels that are not in the contact area
        cost_map = torch.where(cost_map == 0, torch.ones_like(cost_map) * 5, cost_map)


        ## velocity map
        vel_map = []
        for i in range( len(cnt_img_lst)):
            vel_map_i = vel[i] * cnt_img_lst[i]
            vel_map.append(vel_map_i)
        vel_map = torch.amax( torch.stack(vel_map), dim = 0) / self.Config.tab_scale[0]


        return cost_map, vel_map
        

    def __getitem__(self, idx):
        demo_idx, local_idx = self.data_summary['tot_data_index'][idx] # convert to actual index by train/test
        folder_path  = self.data_summary['tot_data_flat'][idx]

        traj_cnt_path = os.path.join(folder_path, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        traj_cnt_fn = traj_cnt_fn[local_idx:]
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()
        traj_rgb_lst = traj_rgb_lst[ local_idx:]


        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)
        cost_map, vel = self._get_gt_cost(traj_cnt_lst)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "cost_map": cost_map,
                    "vel_map": vel,
                    "idx": idx}



class DatasetSeq_front_gt(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train'):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = 'contact_front'
        self.file_list = self.get_file_list()
        self.mode = mode 

        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()
    
    def get_file_list(self):
        traj_cnt_fn_list = []
        for i in self.folder_idx:
            folder_path = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'

            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(traj_cnt_path)
            traj_cnt_fn.sort()
            traj_cnt_fn_list.append(traj_cnt_fn)

        return traj_cnt_fn_list






    def get_cnt_img_dim(self):
        folder_path1 = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h
    
    
    def __len__(self):
        return len(self.folder_idx)

    def _get_gt_cost(self, traj_cnt_lst):

        center_lst = []
        cnt_img_lst = []
        for i in range( len(traj_cnt_lst)):
            cnt_img = traj_cnt_lst[i]
            cnt_img_lst.append(cnt_img)
            
            ## extract center pixel of a mask 
            iidx, jidx = torch.where(cnt_img != 0)
            i_center = math.floor(torch.mean(iidx * 1.0)) * 1.0
            j_center = math.floor(torch.mean(jidx * 1.0)) * 1.0
            
            ## append center list 
            center_lst.append([i_center, j_center])

        ## ground truth cost
        center_lst = torch.tensor(center_lst)
        l2 = torch.norm( center_lst[1:] - center_lst[:-1], dim = -1) 
        
        ## append 0 to the end of tensor
        vel = torch.cat((l2, torch.tensor([0])), dim = 0)
        
        ## cost of each step based on the distance to the goal
        cost = []
        for i in range(len(l2)):
            cost_i = torch.sum(l2[i:])
            cost.append(cost_i)
        cost.append( torch.tensor(0))

        assert len(cnt_img_lst) == len(cost)

        cost_map = []
        for i in range( len(cnt_img_lst)):
            cost_map_i = cost[i] * cnt_img_lst[i]
            cost_map.append(cost_map_i)
        cost_map = torch.amax( torch.stack(cost_map), dim = 0)
        cost_map = cost_map / torch.max(cost_map)

        ## assign high value to the pixels that are not in the contact area
        cost_map = torch.where(cost_map == 0, torch.ones_like(cost_map) * 5, cost_map)


        ## velocity map
        vel_map = []
        for i in range( len(cnt_img_lst)):
            vel_map_i = vel[i] * cnt_img_lst[i]
            vel_map.append(vel_map_i)
        vel_map = torch.amax( torch.stack(vel_map), dim = 0) / self.Config.tab_scale[0]


        return cost_map, vel_map
        

    def __getitem__(self, idx):
        idx = self.folder_idx[idx] # convert to actual index by train/test
        folder_path1 = f'dataset/keyframes/t_{idx:02d}'
        folder_path2 = f'dataset/keyframes/t_{idx:02d}'

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()


        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)
        cost_map, vel = self._get_gt_cost( traj_cnt_lst)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "cost_map": cost_map,
                    "vel_map": vel,
                    "idx": idx}


class DatasetSeq_front(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train'):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = 'contact_key_front'
        self.mode = mode 
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()

    def get_cnt_img_dim(self):
        folder_path1 = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h
    
    
    def __len__(self):
        return len(self.folder_idx)

    def __getitem__(self, idx):
        idx = self.folder_idx[idx] # convert to actual index by train/test
        folder_path1 = f'dataset/keyframes/t_{idx:02d}'
        folder_path2 = f'dataset/keyframes/t_{idx:02d}'

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()


        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "idx": idx}



class DatasetSeq_front(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train'):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = 'contact_key_front'
        self.mode = mode 
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()

    def get_cnt_img_dim(self):
        folder_path1 = f'dataset/keyframes/t_{self.folder_idx[0]:02d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h
    
    
    def __len__(self):
        return len(self.folder_idx)

    def __getitem__(self, idx):
        idx = self.folder_idx[idx] # convert to actual index by train/test
        folder_path1 = f'dataset/keyframes/t_{idx:02d}'
        folder_path2 = f'dataset/keyframes/t_{idx:02d}'

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()


        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "idx": idx}

class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train'):
        self.Config = Config
        self.len = self.Config.len

        self.mode = mode 
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

    def __len__(self):
        return len(self.folder_idx)

    def __getitem__(self, idx):
        idx = self.folder_idx[idx] # convert to actual index by train/test
        folder_path1 = f'dataset/keyframes/t_{idx:02d}'
        folder_path2 = f'dataset/keyframes/t_{idx:02d}'

        traj_cnt_path = os.path.join(folder_path1, 'contact_key')
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        # traj_cnt_fn = [ traj_cnt_fn[i] for i in [0,2,5,8]]
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()


        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "idx": idx}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, Config):
        self.Config = Config
        self.len = self.Config.len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_path = f'dataset/keyframes/t_{idx:02d}'

        traj_rgb_path = os.path.join(folder_path, 'rgb')
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_path = os.path.join(folder_path, 'contact')
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_cnt_lst = fn2img(traj_cnt_fn, d = 1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim = 0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)


        return   {"traj_rgb": traj_rgb_lst[0], 
                    "traj_cnt_lst": traj_cnt_lst, 
                    "mask_t": mask_t, 
                    "traj_len" : seq_l,
                    "idx": idx}