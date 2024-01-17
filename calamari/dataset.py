import os
from pathlib import Path

import math
import torch
import torch.utils.data as data_utils

from calamari.utils import *


class DatasetSeq_front_feedback(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train", seq_l=-1):
        self.Config = Config
        self.len = self.Config.len
        self.mode = mode
        self.return_seq_l = seq_l
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.contact_folder = self.Config.contact_folder  #'contact_key_front'
        self.data_dir = self.Config.data_dir
        self.data_summary = self.get_data_summary()
        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()

    def _get_idx_from_fn(self, fn):
        return int(fn.split(".")[0][-3:])

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
        data_summary = {"tot_rgb_flat": [], "tot_rgb_index": {}}
        tot_length = 0
        for i in self.folder_idx:
            if i < 100:
                folder_path = f"{self.data_dir}/t_{i:02d}/rgb"
            else:
                folder_path = f"{self.data_dir}/t_{i}/rgb"
            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(folder_path)
            traj_cnt_fn.sort()
            traj_cnt_fn.pop()

            local_idx = range(len(traj_cnt_fn))
            data_summary["tot_rgb_flat"].extend(traj_cnt_fn)

            for local_idx_i in local_idx:
                data_summary["tot_rgb_index"][tot_length + local_idx_i] = (
                    i,
                    local_idx_i,
                )
            tot_length += len(traj_cnt_fn)
        return data_summary

    def __len__(self):
        return len(self.data_summary["tot_rgb_flat"])

    def get_cnt_img_dim(self):
        folder_path1 = f"{self.data_dir}/t_{self.folder_idx[0]:02d}"
        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def __getitem__(self, idx):
        txt = "Use the sponge to clean up the dirt."
        # txt  = "sponge sponge dirt dirt dirt."

        demo_idx, local_idx = self.data_summary["tot_rgb_index"][
            idx
        ]  # convert to actual index by train/test
        folder_path = Path(self.data_summary["tot_rgb_flat"][idx]).parent.parent

        ## Get RGB list
        traj_rgb_path = os.path.join(os.path.join(folder_path), "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        # get RGB image path for batch
        options = self._get_idxs_from_fns(traj_rgb_lst[local_idx:])
        traj_rgb_lst = traj_rgb_lst[local_idx]

        # Get heatmaps.
        heatmap_folders = traj_rgb_lst.replace("rgb/", "heatmap_cont/").replace(
            ".png", ""
        )

        traj_cnt_path = os.path.join(os.path.join(folder_path), self.contact_folder)
        traj_cnt_fn_all = folder2filelist(traj_cnt_path)
        cnt_idx_cands = self._get_idxs_from_fns(traj_cnt_fn_all)

        traj_cnt_fn_idx = self._get_valid_keyc_idx(options, cnt_idx_cands)
        traj_cnt_fn = [traj_cnt_fn_all[i] for i in traj_cnt_fn_idx]
        if self.return_seq_l > 0:
            traj_cnt_fn = traj_cnt_fn[: self.return_seq_l]

        if len(traj_cnt_fn) > self.Config.contact_seq_l:
            traj_cnt_fn = traj_cnt_fn[: self.Config.contact_seq_l]

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)

        if len(traj_cnt_fn) < self.Config.contact_seq_l:
            # Fill with padding.
            filler = torch.zeros(
                (
                    self.Config.contact_seq_l - len(traj_cnt_fn),
                    traj_cnt_lst.shape[1],
                    traj_cnt_lst.shape[2],
                )
            )
            traj_cnt_lst = torch.cat([traj_cnt_lst, filler], dim=0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        # print(len(traj_cnt_lst), len(traj_rgb_lst))
        return {
            "traj_rgb": traj_rgb_lst,
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "idx": idx,
            "heatmap_folder": heatmap_folders,
            "txt": txt,
        }


class Dataset_front_gt_feedback(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train"):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = "contact_front"
        self.mode = mode

        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.data_summary = self.get_data_summary()
        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()
        self.l = self.__len__()

    def get_data_summary(self):
        data_summary = {"tot_data_flat": [], "tot_data_index": {}}
        tot_length = 0
        for i in self.folder_idx:
            folder_path = f"dataset/keyframes/t_{i:02d}/{self.contact_folder}"
            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(folder_path)
            traj_cnt_fn.sort()
            traj_cnt_fn.pop()

            local_idx = range(len(traj_cnt_fn))
            data_summary["tot_data_flat"].extend(traj_cnt_fn)

            for local_idx_i in local_idx:
                data_summary["tot_data_index"][tot_length + local_idx_i] = (
                    i,
                    local_idx_i,
                )

            tot_length += len(traj_cnt_fn)

        return data_summary

    def __len__(self):
        return len(self.data_summary["tot_data_flat"])

    def get_cnt_img_dim(self):
        folder_path1 = f"dataset/keyframes/t_{self.folder_idx[0]:02d}"
        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def _get_gt_cost(self, traj_cnt_lst, mask_t):
        center_lst = []
        cnt_img_lst = []

        for i in range(len(traj_cnt_lst)):
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
        l2 = torch.norm(center_lst[1:] - center_lst[:-1], dim=-1) / 160

        ## append 0 to the end of tensor
        vel = torch.cat((l2, torch.tensor([0])), dim=0)

        ## cost of each step based on the distance to the goal
        cost = []
        for i in range(len(l2)):
            cost_i = torch.sum(l2[i:])
            cost.append(cost_i)
        cost.append(torch.tensor(0))

        assert len(cnt_img_lst) == len(cost)

        cost_map = []
        for i in range(len(cnt_img_lst)):
            cost_map_i = cost[i] * cnt_img_lst[i]
            cost_map.append(cost_map_i)
        cost_map = torch.amax(torch.stack(cost_map), dim=0)
        # cost_map = cost_map / torch.max(cost_map)

        ## assign high value to the pixels that are not in the contact area
        cost_map = torch.where(
            torch.tensor(mask_t) == 0, torch.ones_like(cost_map) * 10, cost_map
        )

        ## velocity map
        vel_map = []
        for i in range(len(cnt_img_lst)):
            vel_map_i = vel[i] * cnt_img_lst[i]
            vel_map.append(vel_map_i)
        vel_map = torch.amax(torch.stack(vel_map), dim=0) / self.Config.tab_scale[0]

        return cost_map, vel_map

    def __getitem__(self, idx):
        demo_idx, local_idx = self.data_summary["tot_data_index"][
            idx
        ]  # convert to actual index by train/test
        folder_path = os.path.dirname(self.data_summary["tot_data_flat"][idx])

        traj_cnt_path = os.path.join(folder_path)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        traj_cnt_fn = traj_cnt_fn[local_idx:]
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = folder_path.replace("contact_front", "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()
        traj_rgb_lst = traj_rgb_lst[local_idx]

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)
        cost_map, vel = self._get_gt_cost(traj_cnt_lst, mask_)

        # print( torch.amax( cost_map * torch.tensor(mask_)))

        return {
            "traj_rgb": traj_rgb_lst,
            # "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "cost_map": cost_map,
            "vel_map": vel,
            "idx": idx,
        }


class DatasetSeq_front_gt(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train"):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = "contact_front"
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
            folder_path = f"dataset/keyframes/t_{self.folder_idx[0]:02d}"

            # get the list of trajectories within the folder
            traj_cnt_fn = folder2filelist(traj_cnt_path)
            traj_cnt_fn.sort()
            traj_cnt_fn_list.append(traj_cnt_fn)

        return traj_cnt_fn_list

    def get_cnt_img_dim(self):
        folder_path1 = f"dataset/keyframes/t_{self.folder_idx[0]:02d}"
        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def __len__(self):
        return len(self.folder_idx)

    def _get_gt_cost(self, traj_cnt_lst):
        center_lst = []
        cnt_img_lst = []
        for i in range(len(traj_cnt_lst)):
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
        l2 = torch.norm(center_lst[1:] - center_lst[:-1], dim=-1)

        ## append 0 to the end of tensor
        vel = torch.cat((l2, torch.tensor([0])), dim=0)

        ## cost of each step based on the distance to the goal
        cost = []
        for i in range(len(l2)):
            cost_i = torch.sum(l2[i:])
            cost.append(cost_i)
        cost.append(torch.tensor(0))

        assert len(cnt_img_lst) == len(cost)

        cost_map = []
        for i in range(len(cnt_img_lst)):
            cost_map_i = cost[i] * cnt_img_lst[i]
            cost_map.append(cost_map_i)
        cost_map = torch.amax(torch.stack(cost_map), dim=0)
        cost_map = cost_map / torch.max(cost_map)

        ## assign high value to the pixels that are not in the contact area
        cost_map = torch.where(cost_map == 0, torch.ones_like(cost_map) * 5, cost_map)

        ## velocity map
        vel_map = []
        for i in range(len(cnt_img_lst)):
            vel_map_i = vel[i] * cnt_img_lst[i]
            vel_map.append(vel_map_i)
        vel_map = torch.amax(torch.stack(vel_map), dim=0) / self.Config.tab_scale[0]

        return cost_map, vel_map

    def __getitem__(self, idx):
        idx = self.folder_idx[idx]  # convert to actual index by train/test
        folder_path1 = f"dataset/keyframes/t_{idx:02d}"
        folder_path2 = f"dataset/keyframes/t_{idx:02d}"

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)
        cost_map, vel = self._get_gt_cost(traj_cnt_lst)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        return {
            "traj_rgb": traj_rgb_lst[0],
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "cost_map": cost_map,
            "vel_map": vel,
            "idx": idx,
        }


class DatasetSeq_front(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train"):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = "contact_key_front"
        self.mode = mode
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()

    def get_cnt_img_dim(self):
        folder_path1 = f"dataset/keyframes/t_{self.folder_idx[0]:02d}"
        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def __len__(self):
        return len(self.folder_idx)

    def __getitem__(self, idx):
        idx = self.folder_idx[idx]  # convert to actual index by train/test
        folder_path1 = f"dataset/keyframes/t_{idx:02d}"
        folder_path2 = f"dataset/keyframes/t_{idx:02d}"

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        return {
            "traj_rgb": traj_rgb_lst[0],
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "idx": idx,
        }


class DatasetSeq_front(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train"):
        self.Config = Config
        self.len = self.Config.len
        self.contact_folder = "contact_key_front"
        self.mode = mode
        if self.mode == "train":
            self.folder_idx = self.Config.train_idx
        if self.mode == "test":
            self.folder_idx = self.Config.test_idx

        self.cnt_w, self.cnt_h = self.get_cnt_img_dim()

    def get_cnt_img_dim(self):
        folder_path1 = f"dataset/keyframes/t_{self.folder_idx[0]:02d}"
        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h

    def __len__(self):
        return len(self.folder_idx)

    def __getitem__(self, idx):
        idx = self.folder_idx[idx]  # convert to actual index by train/test
        folder_path1 = f"dataset/keyframes/t_{idx:02d}"
        folder_path2 = f"dataset/keyframes/t_{idx:02d}"

        traj_cnt_path = os.path.join(folder_path1, self.contact_folder)
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        return {
            "traj_rgb": traj_rgb_lst[0],
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "idx": idx,
        }


class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, Config, mode="train"):
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
        idx = self.folder_idx[idx]  # convert to actual index by train/test
        folder_path1 = f"dataset/keyframes/t_{idx:02d}"
        folder_path2 = f"dataset/keyframes/t_{idx:02d}"

        traj_cnt_path = os.path.join(folder_path1, "contact_key")
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        # traj_cnt_fn = [ traj_cnt_fn[i] for i in [0,2,5,8]]
        seq_l = len(traj_cnt_fn)

        traj_rgb_path = os.path.join(folder_path2, "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        return {
            "traj_rgb": traj_rgb_lst[0],
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "idx": idx,
        }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, Config):
        self.Config = Config
        self.len = self.Config.len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_path = f"dataset/keyframes/t_{idx:02d}"

        traj_rgb_path = os.path.join(folder_path, "rgb")
        traj_rgb_lst = folder2filelist(traj_rgb_path)
        traj_rgb_lst.sort()

        traj_cnt_path = os.path.join(folder_path, "contact")
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()
        seq_l = len(traj_cnt_fn)

        traj_cnt_lst = fn2img(traj_cnt_fn, d=1)
        traj_cnt_lst = torch.stack(traj_cnt_lst, dim=0)

        mask_ = get_traj_mask(traj_cnt_fn)
        mask_t = torch.tensor(mask_).to(self.Config.device)

        return {
            "traj_rgb": traj_rgb_lst[0],
            "traj_cnt_lst": traj_cnt_lst,
            "mask_t": mask_t,
            "traj_len": seq_l,
            "idx": idx,
        }
