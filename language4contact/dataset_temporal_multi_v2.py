import os 
from pathlib import Path

import math
import torch
import torch.utils.data as data_utils
import pickle
from language4contact.modules_shared import *
from language4contact.utils import *
from .Transformer_MM_Explainability.CLIP import clip 
from tqdm._tqdm import trange
import torchvision.transforms.functional as F
import torchvision.transforms as T



class DatasetTemporal(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train', seq_l = -1, image_encoder = None):
        self.Config = Config
        self.aug_num = self.Config.aug_num
        self.contact_seq_l = self.Config.contact_seq_l
        self.mode = mode 
        self.return_seq_l = seq_l
        # self.txt_cmd = self.Config.txt_cmd
        

        self.contact_folder = self.Config.contact_folder  #'contact_key_front'
        self.data_summary = self.get_data_summary()

        # pretrained model.
        self.randomaffine = T.RandomAffine(degrees = 10, 
                                           translate = (0.15, 0.15), 
                                           scale = (0.9, 1.1), 
                                           shear = (5, 5))
        self._image_encoder = image_encoder 
        self.explainability = ClipExplainability(self.Config.device)

        # get language embeddings
        self.preprocess_all()
        



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
            # read pickle


            txt_cmd = task["txt_cmd"]
            heatmap_folder = task["heatmap_folder"]



            for i in task[self.mode+"_idx"]:
                
                des = f'{task["data_dir"]}/episode{i}/description.txt'

                with open(des) as f:
                    txt_cmd = f.readlines()[0]



                cnt_folder_path = f'{task["data_dir"]}/episode{i}/contact_front'
                rgb_folder_path = f'{task["data_dir"]}/episode{i}/rgb'

                traj_cnt_fn = folder2filelist(cnt_folder_path)
                traj_cnt_fn.sort()

                traj_rgb_fn = folder2filelist(rgb_folder_path)
                traj_rgb_fn.sort()

                traj_rgb_fn = traj_rgb_fn[:-1] # pop last obs

                for local_idx in range(len(traj_rgb_fn)):

                    # t-3, t-2, t-1, t RGB
                    if local_idx < self.contact_seq_l:
                        traj_rgb_lst = ['','','','']
                        traj_rgb_lst[ - (local_idx+1):] = traj_rgb_fn[:local_idx+1]
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
                                                "heatmap_folder":heatmap_folder,
                                                }

                    # Increase idx.
                    tot_length += 1
        return data_summary


    def read_heatmap_temporal(self, img_pths, texts, task_i, aug_idx_i, heatmap_folder = None):

        words_i = sentence2words(texts)
        if self.Config.heatmap_type == 'chefer':
            cnt_dir = 'heatmap/'
            for row_idx, row in enumerate(words_i):
                row.insert(0, texts[row_idx])

        elif self.Config.heatmap_type == 'huy':
            if heatmap_folder is None:
                cnt_dir = 'heatmap_huy_center_/'
            else: 
                cnt_dir = heatmap_folder

        ## image processing
        txt_emb_batch = []
        heatmaps_batch = []
        img_batch = []
        tp_masks = []
        vl_masks = torch.ones((self.Config.contact_seq_l, self.Config.max_sentence_l))
        tp_mask = torch.ones(self.Config.contact_seq_l)
        heatmaps_batch = torch.zeros((self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.heatmap_size[0], self.Config.heatmap_size[1])).to(self.Config.device)
        txt_batch =  torch.zeros((self.Config.contact_seq_l, self.Config.max_sentence_l, 512)).to(self.Config.device)


        



        # Image seq
        for l, img_pth in enumerate(img_pths):
            
            if len(img_pth) > 0:
                heatmaps = []
                txts = []

                for i in range(self.Config.max_sentence_l):

                    # Read word-wise heatmaps.
                    if i < len(words_i):
                        # Real VL-transformer Inputs.

                        # i-th word.
                        wd = words_i[i].replace('.', '')
                        wd = wd.replace(',', '')

                        # Load word-wise heatmaps.
                        hm_pth = img_pth.replace('rgb/', cnt_dir).split('.')[0]
                        hm_pth = os.path.join(hm_pth, wd + '.png')
                        heatmap = Image.open(hm_pth)
                        # heatmap = Image.fromarray(augmentation(aug_idx_i, heatmap, rgb= False).numpy())
                        
                        # heatmap = heatmap.resize((self.Config.heatmap_size[0], self.Config.heatmap_size[1]))
                        heatmap = torch.tensor( np.array(heatmap)).to(self.Config.device)
                        # print(heatmap.shape)
                        # breakpoint()
                        # heatmap = augmentation(aug_idx_i, heatmap, rgb= False)
                        heatmaps.append(heatmap)


                        # read txt emb.
                        with open(hm_pth.replace('png', 'npy'), 'rb') as f:
                            txt_i = torch.tensor(np.load(f))
                            txts.append( txt_i)

                        # vl transformer mask = 0 : Real Input.
                        vl_masks[l, i] = 0
                    
                    else:
                        # Fake (Padding) VL-transformer Inputs.
                        padding = torch.zeros_like(heatmaps[-1]).to(self.Config.device)
                        heatmaps.append(padding)
                        vl_masks[l, len(words_i):] = 1

                # Real Temporal-transformer Inputs.   
                # txt_emb_batch.insert(0, txt_emb_i)
                # txt_emb_batch[b,-(l+1),:] = txt_emb_i
                heatmaps_batch[l,:] = torch.stack(heatmaps)
                txt_batch[l, :len(words_i)] = torch.cat(txts, dim=0)
                # heatmaps_batch.insert( 0,torch.stack(heatmaps))
                tp_mask[l] = 0

            elif len(img_pth) == 0: 
                 tp_mask[l] = 1

        # print(vl_masks, tp_mask, torch.sum(heatmaps_batch, dim = (2,3)), torch.sum(txt_batch, dim = -1))
        # breakpoint()
        # Formatting.
        # heatmap_batch_ = torch.stack(heatmaps_batch)
        # heatmaps_batch = torch.flatten(heatmaps_batch, 0, 1)
        # txt_emb_batch = torch.flatten(txt_emb_batch, 0, 1)
        vl_masks = vl_masks.to(self.Config.device) # Match heatmap size.
        # vl_masks = torch.flatten(vl_masks, 0, 1).to(self.Config.device) # Match heatmap size.
        tp_masks = torch.tensor(tp_mask).to(self.Config.device)
        return txt_batch, heatmaps_batch, vl_masks.bool(), tp_masks.bool()
    


    def heatmap_augmentation(self, heatmaps_batch, aug_param):


        # Image seq
        for l in range(heatmaps_batch.shape[0]): # per batch
            heatmaps = []

            heatmaps_batch[-(l+1),:] = F.affine( heatmaps_batch[-(l+1),:].unsqueeze(1), *aug_param, interpolation=self.randomaffine.interpolation, fill=self.randomaffine.fill).squeeze()
            # heatmaps_batch[-(l+1),:] = torch.stack(heatmaps)

        return heatmaps_batch
    


    
    def input_processing(self, fused_x, heatmaps, vl_mask, tp_mask, aug_param):
        """
        mode = 'train' : load heatmap
        mode = 'test' : generate heatmap every time step
        """
        # txt_emb, heatmaps, padding_mask = self.explainability.get_heatmap_temporal(img, texts)
        self.B = len(heatmaps)
        # if aug_idx is None:
        #     aug_idx = torch.zeros((self.B, 1))

        # fused_x, heatmaps, vl_mask, tp_mask = self.read_heatmap_temporal(img, texts, tasks, heatmap_folder = heatmap_folder) 
        """
        heatmaps : (B*l_temp) X l_lang x 225x225
        """

        if aug_param is not None:
            heatmaps = self.heatmap_augmentation(heatmaps, aug_param)
        
        hm_emb = []

        ## Get clip attention
        # heatmaps_ = heatmaps[ ~vl_mask] # zero image to padding
        heatmaps_ = heatmaps[ ~vl_mask[:,0]] # zero image to padding


        # print("hm", heatmaps_.shape)
        img_enc_inp = torch.flatten(heatmaps_, 0, 1).unsqueeze(1).float()
        img_enc_inp = torch.cat([img_enc_inp,
                                  img_enc_inp,
                                  img_enc_inp] , dim = 1)
        # img_enc_inp = heatmaps_.unsqueeze(1).float()
        # print(heatmaps_.shape)

        self._image_encoder["model"](img_enc_inp)
        out = self._image_encoder["activation"]['avgpool'].squeeze()
        # out = self._image_encoder["model"](img_enc_inp)



        # inp = torch.zeros((heatmaps.shape[0], out.shape[-1])).to(self.Config.device)
        visual_sentence = torch.zeros((heatmaps.shape[0], heatmaps.shape[1], out.shape[-1])).to(self.Config.device)
        # print(inp.shape, heatmaps_.shape, out.shape)
        # inp[ ~vl_mask] = out #.reshape((self.Config.contact_seq_l, self.Config.max_sentence_l, -1))
        visual_sentence[ ~vl_mask[:,0]] = out.reshape((heatmaps_.shape[0], heatmaps.shape[1], out.shape[-1]))
        


        # inp = inp.reshape(( heatmaps.shape[0], heatmaps.shape[1], out.shape[-1])) # [batch size x seq x feat_dim]
        return visual_sentence, fused_x, vl_mask, tp_mask
        # return inp, txt_emb, vl_mask, tp_mask

    
    def preprocess_all(self):
        self.tot_dataset = {"visual_sentence": [],
                        "fused_x" : [],
                        "vl_mask": [],
                        "tp_mask":[],
                        "aug_idx": [],
                        "traj_rgb_paths": [], 
                        "traj_cnt_paths": [], 
                        "traj_cnt_img": [], 
                        "idx": [], 
                        "txt": [], 
                        "task": [],
                        "heatmap_folder": [],
                        "binary_contact": [],
                        "heatmaps": []}
        
        
        for idx_raw in trange( self.__len__()):

            if self.mode == 'train':   
                idx = idx_raw // self.aug_num
            elif self.mode == 'test':
                idx = idx_raw

            # idx_flip, idx_vtc, idx_hr  = self.idx_lst[idx_raw % 18]


            traj_rgb_lst = self.data_summary[idx]["traj_rgb_paths"]
            traj_cnt_lst = self.data_summary[idx]["traj_cnt_paths"]
            heatmap_folder = self.data_summary[idx]["heatmap_folder"]
            txt = self.data_summary[idx]["txt"]
            task = self.data_summary[idx]["task"]

            traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
            # mask_ = get_traj_mask(traj_cnt_lst)


            self.cnt_w, self.cnt_h = traj_cnt_img[0].shape[-2:]


            binary_contact = 0 if torch.sum(traj_cnt_img[0]) == 0 else 1

            fused_x, heatmaps, vl_mask, tp_mask = self.read_heatmap_temporal(traj_rgb_lst, 
                                                                    txt, 
                                                                    task, 
                                                                    heatmap_folder) 
    


            self.tot_dataset["traj_rgb_paths"].append( traj_rgb_lst)
            self.tot_dataset["traj_cnt_paths"].append( traj_cnt_lst)
            self.tot_dataset["traj_cnt_img"].append( traj_cnt_img)
            self.tot_dataset["traj_cnt_paths"].append( traj_cnt_lst)
            self.tot_dataset["txt"].append( txt)
            self.tot_dataset["task"].append( task)
            self.tot_dataset["heatmap_folder"].append( heatmap_folder)
            self.tot_dataset["binary_contact"].append( binary_contact)
            
            self.tot_dataset["fused_x"].append( fused_x)
            self.tot_dataset["heatmaps"].append( heatmaps)
            self.tot_dataset["vl_mask"].append( vl_mask)
            self.tot_dataset["tp_mask"].append( tp_mask)



    def __len__(self):
        if self.mode == 'train':
            return len(self.data_summary) * self.aug_num
        
        elif self.mode == 'test':
            return len(self.data_summary)
        # return len(self.data_summary['tot_rgb_flat'])


    def __getitem__(self, idx_raw_):
        if self.mode == 'train':
            idx_raw = idx_raw_//  self.aug_num
        else:
            idx_raw = idx_raw_
        


        traj_cnt_img = self.tot_dataset["traj_cnt_img"][idx_raw]
        w, h = traj_cnt_img[0].shape[-2:]
        if self.mode == 'train': 
            # Param for data augmentation. 
            aug_param = self.randomaffine.get_params(self.randomaffine.degrees, self.randomaffine.translate, 
                                                    self.randomaffine.scale, self.randomaffine.shear, (w, h))
            
            # traj_cnt_img =  augmentation(aug_idx, torch.stack(traj_cnt_img), rgb = False)
            traj_cnt_img = [F.affine(traj_cnt_img[0].unsqueeze(0), *aug_param, interpolation=self.randomaffine.interpolation, fill=self.randomaffine.fill).squeeze()]


        elif self.mode == 'test':
            traj_cnt_img = [traj_cnt_img[0]]
            aug_param = None
        
        # fused_x, heatmaps, vl_mask, tp_mask, *aug_param
        visual_sentence, fused_x, vl_mask, tp_mask =  self.input_processing(self.tot_dataset["fused_x"][idx_raw], 
                                                                            self.tot_dataset["heatmaps"][idx_raw], 
                                                                            self.tot_dataset["vl_mask"][idx_raw], 
                                                                            self.tot_dataset["tp_mask"][idx_raw], 
                                                                            aug_param)
    
        aug_param = [-999] if aug_param is None else aug_param

        # self.tot_dataset["visual_sentence"].append( visual_sentence.detach().cpu())
        # self.tot_dataset["fused_x"].append( fused_x.detach().cpu())
        # self.tot_dataset["vl_mask"].append( vl_mask.detach().cpu())
        # self.tot_dataset["tp_mask"].append( tp_mask.detach().cpu())
        # self.tot_dataset["aug_idx"].append( aug_param)



        return   {
                "visual_sentence": visual_sentence.detach().to(self.Config.device) ,
                "fused_x" : fused_x.detach().to(self.Config.device) ,
                "vl_mask": vl_mask.detach().to(self.Config.device) ,
                "tp_mask": tp_mask.detach().to(self.Config.device),
                "aug_idx": aug_param,
                "traj_rgb_paths": self.tot_dataset["traj_rgb_paths"][idx_raw]  , 
                "traj_cnt_paths": self.tot_dataset["traj_cnt_paths"][idx_raw]  , 
                "traj_cnt_img": traj_cnt_img, 
                # "mask_t": mask_t, 
                "idx": idx_raw_, 
                "txt": self.tot_dataset["txt"][idx_raw], 
                "task": self.tot_dataset["task"][idx_raw],
                "binary_contact": self.tot_dataset["binary_contact"][idx_raw]
                }




