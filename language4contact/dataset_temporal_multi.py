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
        # self.txt_cmd = self.Config.txt_cmd
        
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
        data_summary = {}
        '''
        {idx :{ rgb history path list: List[str], txt command: str, contact goal path: str}}
        '''
        
        tot_length = 0
        for task_n, task in self.Config.dataset_config.items():
            txt_cmd = task["txt_cmd"]
            for i in task[self.mode+"_idx"]:
                cnt_folder_path = f'{task["data_dir"]}/t_{i:03d}/contact_front'
                rgb_folder_path = f'{task["data_dir"]}/t_{i:03d}/rgb'

                traj_cnt_fn = folder2filelist(cnt_folder_path)
                traj_cnt_fn.sort()

                traj_rgb_fn = folder2filelist(rgb_folder_path)
                traj_rgb_fn.sort()

                traj_rgb_fn = traj_rgb_fn[:-1] # pop last obs

                for local_idx in range(len(traj_rgb_fn)):

                    # t-3, t-2, t-1, t RGB
                    if local_idx < self.contact_seq_l:
                        traj_rgb_lst = ['','','','']
                        traj_rgb_lst[:local_idx+1] = traj_rgb_fn[:local_idx+1]
                        traj_cnt_lst = [traj_cnt_fn[local_idx]]
                    else:
                        traj_rgb_lst = traj_rgb_fn[local_idx-3:local_idx+1]
                        traj_cnt_lst = [traj_cnt_fn[local_idx]]

                    # Save.
                    data_summary[tot_length] = {
                                                "traj_rgb_paths": traj_rgb_lst,
                                                "traj_cnt_paths": traj_cnt_lst, 
                                                "txt": txt_cmd,
                                                "task": task_n,
                                                }

                    # Increase idx.
                    tot_length += 1
        return data_summary

    def __len__(self):
        return len(self.data_summary)
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
        flip = np.random.randint(0, 2)
        traj_rgb_lst = self.data_summary[idx]["traj_rgb_paths"]
        traj_cnt_lst = self.data_summary[idx]["traj_cnt_paths"]
        txt = self.data_summary[idx]["txt"]
        task = self.data_summary[idx]["task"]


        traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
        mask_ = get_traj_mask(traj_cnt_lst)
        mask_t = torch.tensor(mask_).to(self.Config.device)
        
        if flip:
            traj_cnt_img =  torch.flip( torch.stack(traj_cnt_img), (-1,))
            traj_cnt_img = [traj_cnt_img[0]]
            mask_t =  torch.flip(mask_t, (-1,))
        
        # print(torch.stack(trajcnt_img).shape)


        # print(traj_rgb_lst,traj_cnt_lst )
        return   {
                "flip": torch.tensor(flip),
                "traj_rgb_paths": traj_rgb_lst, 
                "traj_cnt_paths": traj_cnt_lst, 
                "traj_cnt_img": traj_cnt_img, 
                "mask_t": mask_t, 
                "idx": idx, 
                "txt": txt, "task": task}

