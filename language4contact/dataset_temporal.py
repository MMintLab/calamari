import os 
from pathlib import Path

import math
import torch
import torch.utils.data as data_utils

from language4contact.utils import *

class DatasetTemporal(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train', seq_l = -1):
        self.Config = Config
        self.contact_seq_l = self.Config.contact_seq_l
        self.mode = mode 
        self.return_seq_l = seq_l
        self.txt_cmd = self.Config.txt_cmd
        
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx


        self.contact_folder = self.Config.contact_folder  #'contact_key_front'
        self.data_dir = self.Config.data_dir
        self.data_summary = self.get_data_summary()
        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()


    def _get_idx_from_fn(self, fn):
        return int(fn.split('.')[0][-3:])

    def _get_idxs_from_fns(self, fns):
        idxs = []
        for fn in fns:
            idxs.append(self._get_idx_from_fn(fn))
        return idxs

    def _get_valid_keyc_idx(self, option, cand):
        valid_idx = []
        for cidx, c in enumerate(cand):
            if c in option:
                valid_idx.append(cidx)
        return valid_idx


    def get_data_summary(self):
        data_summary = {'tot_rgb_flat': [], 'tot_rgb_index':{}, 
                        'tot_cnt_flat': [], 'tot_cnt_index':{}}
        tot_length = 0
        for i in self.folder_idx:
            cnt_folder_path = f'{self.data_dir}/t_{i:03d}/contact_front'
            rgb_folder_path = f'{self.data_dir}/t_{i:03d}/rgb'

            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(cnt_folder_path)
            traj_cnt_fn.sort()

            # get the list of trajectories within the folder
            traj_rgb_fn = folder2filelist(rgb_folder_path)
            traj_rgb_fn.sort()
            # traj_rgb_fn = [traj_cnt_fn_i.replace('contact_front', 'rgb') for traj_cnt_fn_i in traj_cnt_fn]
            # traj_rgb_fn = [traj_rgb_fn_i.replace('contact', 'rgb') for traj_rgb_fn_i in traj_rgb_fn]

            
            # traj_cnt_fn = traj_cnt_fn # pop first cnt 
            traj_rgb_fn = traj_rgb_fn[:-1] # pop last obs 

            # Add dataset.
            local_idx = range(len(traj_cnt_fn))
            data_summary['tot_cnt_flat'].extend(traj_cnt_fn)
            data_summary['tot_rgb_flat'].extend(traj_rgb_fn)

            # Index.
            for local_idx_i in local_idx:
                # (tot_idx, local_idx, min local idx)
                data_summary['tot_rgb_index'][tot_length + local_idx_i] = (i, local_idx_i, tot_length)
                data_summary['tot_cnt_index'][tot_length + local_idx_i] = (i, local_idx_i, tot_length)
            
            
            tot_length += len(traj_cnt_fn)
        self.tot_length = tot_length

        return data_summary

    def __len__(self):
        return self.tot_length
        # return len(self.data_summary['tot_rgb_flat'])

    def get_cnt_img_dim(self):
        # Read random contact img.
        folder_path1 = f'{self.data_dir}/t_{self.folder_idx[0]:03d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        # Get size.
        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def __getitem__(self, idx):
        txt  = self.txt_cmd 
        # txt  = "sponge sponge dirt dirt dirt."

        # local index follows rgb.
        demo_idx, local_idx, min_local_idx = self.data_summary['tot_rgb_index'][idx] # convert to actual index by train/test
        folder_path  =  Path(self.data_summary['tot_rgb_flat'][idx]).parent.parent
        
        # t-3, t-2, t-1, t RGB
        if local_idx < self.contact_seq_l:
            # traj_rgb_lst = [-1, -1, -1, -1]
            traj_rgb_lst = ['','','','']

            traj_rgb_lst[:local_idx+1] = self.data_summary['tot_rgb_flat'][min_local_idx:idx+1]
            traj_cnt_lst = [self.data_summary['tot_cnt_flat'][idx]]
        else:
            traj_rgb_lst = self.data_summary['tot_rgb_flat'][idx-3:idx+1]
            traj_cnt_lst = [self.data_summary['tot_cnt_flat'][idx]]

        traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
        mask_ = get_traj_mask(traj_cnt_lst)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        # print(traj_rgb_lst,traj_cnt_lst )
        return   {"traj_rgb_paths": traj_rgb_lst, 
                "traj_cnt_paths": traj_cnt_lst, 
                "traj_cnt_img": traj_cnt_img, 
                "mask_t": mask_t, 
                "idx": idx, 
                "local_idx": local_idx, 
                # "heatmap_folder": heatmap_folders, 
                "txt": txt}

