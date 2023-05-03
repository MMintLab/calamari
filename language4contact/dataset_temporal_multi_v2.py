import os 
from pathlib import Path

import math
import torch
import torch.utils.data as data_utils

from language4contact.modules_shared import *
from language4contact.utils import *
from .Transformer_MM_Explainability.CLIP import clip 
from tqdm._tqdm import trange



class DatasetTemporal(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train', seq_l = -1, image_encoder = None):
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

        # pretrained model.

        self._image_encoder = image_encoder 
        self.explainability = ClipExplainability(self.Config.device)

        # get language embeddings
        self.text_embs = self._get_text_emb()
        self.idx_lst = [[0, -1, -1], [0, -1, 0], [0, -1, 1],
                    [0, 0, -1], [0, 0, 0], [0, 0, 1],
                    [0, 1, -1], [0, 1, 0], [0, 1, 1],
                   [1, -1, -1], [1, -1, 0], [1, -1, 1],
                    [1, 0, -1], [1, 0, 0], [1, 0, 1],
                    [1, 1, -1], [1, 1, 0], [1, 1, 1]]
        self.preprocess_all()


    def _get_text_emb(self):
        txt_emb = {}
        for task_n, task in self.Config.dataset_config.items():
            texts = task["txt_cmd"]
            words = sentence2words(texts)
            words.insert(0, texts)
            text = clip.tokenize(words).to(self.Config.device)
            txt_emb_i = self.explainability.model.encode_text(text)
            txt_emb[task_n] = txt_emb_i.detach()

        return txt_emb


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
                import pickle
                des = f'{task["data_dir"]}/episode{i}/description.txt'
                with open(des, 'rb') as f:
                    txt_cmd = pickle.load(f)


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
        # text = clip.tokenize(words).to(self.device)

        ## image processing
        txt_emb_batch = []
        heatmaps_batch = []
        img_batch = []
        tp_masks = []
        vl_masks = torch.ones((self.Config.contact_seq_l, self.Config.max_sentence_l))
        # txt_emb_batch = torch.zeros((self.Config.contact_seq_l, self.Config.max_sentence_l, 512)).to(self.Config.device)
        heatmaps_batch = torch.zeros((self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.heatmap_size[0], self.Config.heatmap_size[1])).to(self.Config.device)

        # Batch
        # for b, img_pths in enumerate(img_batch_pths):
        tp_mask = []
        #     words_i = words[b]
        #     task_i = tasks[b]
        #     flip_i = flip[b]

        # Read Texts as a token.
        txt_emb_i = torch.zeros((self.Config.max_sentence_l, 512)).to(self.Config.device)
        txt_emb_i[:len(self.text_embs[task_i])] = self.text_embs[task_i]
        txt_emb_batch = txt_emb_i.unsqueeze(0).repeat((self.Config.contact_seq_l, 1, 1)) 

        # Image seq
        for l, img_pth in enumerate(img_pths):
            
            if len(img_pth) > 0:
                heatmaps = []

                for i in range(self.Config.max_sentence_l):

                    # Read word-wise heatmaps.
                    if i < len(words_i):
                        # Real VL-transformer Inputs.

                        # i-th word.
                        wd = words_i[i].replace('.', '')

                        # Load word-wise heatmaps.
                        hm_pth = img_pth.replace('rgb/', cnt_dir).split('.')[0]
                        hm_pth = os.path.join(hm_pth, wd + '.png')
                        heatmap = Image.open(hm_pth)
                        # heatmap = Image.fromarray(augmentation(aug_idx_i, heatmap, rgb= False).numpy())
                        
                        # heatmap = heatmap.resize((self.Config.heatmap_size[0], self.Config.heatmap_size[1]))
                        heatmap = torch.tensor( np.array(heatmap)).to(self.Config.device)
                        # print(heatmap.shape)
                        # breakpoint()
                        heatmap = augmentation(aug_idx_i, heatmap, rgb= False)
                        heatmaps.append(heatmap)

                        # vl transformer mask = 0 : Real Input.
                        vl_masks[-(l+1),i] = 0
                    
                    else:
                        # Fake (Padding) VL-transformer Inputs.
                        padding = torch.zeros_like(heatmaps[-1]).to(self.Config.device)
                        heatmaps.append(padding)
                        vl_masks[ -(l+1),i] = 1

                # Real Temporal-transformer Inputs.   
                # txt_emb_batch.insert(0, txt_emb_i)
                # txt_emb_batch[b,-(l+1),:] = txt_emb_i
                heatmaps_batch[-(l+1),:] = torch.stack(heatmaps)
                # heatmaps_batch.insert( 0,torch.stack(heatmaps))
                tp_mask.insert(0, 0)

            elif len(img_pth) == 0: 
                tp_mask.insert(0,1)


        # Formatting.
        # heatmap_batch_ = torch.stack(heatmaps_batch)
        # heatmaps_batch = torch.flatten(heatmaps_batch, 0, 1)
        # txt_emb_batch = torch.flatten(txt_emb_batch, 0, 1)
        vl_masks = vl_masks.to(self.Config.device) # Match heatmap size.
        # vl_masks = torch.flatten(vl_masks, 0, 1).to(self.Config.device) # Match heatmap size.
        tp_masks = torch.tensor(tp_mask).to(self.Config.device)
        return txt_emb_batch, heatmaps_batch, vl_masks.bool(), tp_masks.bool()
    
    def input_processing(self, img, texts, tasks, mode = 'train', aug_idx = None, heatmap_folder = None):
        """
        mode = 'train' : load heatmap
        mode = 'test' : generate heatmap every time step
        """
        # txt_emb, heatmaps, padding_mask = self.explainability.get_heatmap_temporal(img, texts)
        self.B = len(img)
        if aug_idx is None:
            aug_idx = torch.zeros((self.B, 1))

        fused_x, heatmaps, vl_mask, tp_mask = self.read_heatmap_temporal(img, texts, tasks, aug_idx, heatmap_folder = heatmap_folder) 
        """
        heatmaps : (B*l_temp) X l_lang x 225x225
        """
        
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


    def get_cnt_img_dim(self):
        # Read random contact img.
        folder_path1 = f'{self.data_dir}/t_{self.folder_idx[0]:03d}'
        traj_cnt_path = os.path.join(folder_path1,self.contact_folder )
        traj_cnt_fn = folder2filelist(traj_cnt_path)
        traj_cnt_fn.sort()

        # Get size.
        w, h, _ = torch.tensor(cv2.imread(traj_cnt_fn[0])).shape
        return w, h
    
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
                        "task": []}
        
        
        for idx_raw in trange(len(self.data_summary) * len(self.idx_lst)):
                   
            idx = idx_raw // 18
            idx_flip, idx_vtc, idx_hr  = self.idx_lst[idx_raw % 18]


            traj_rgb_lst = self.data_summary[idx]["traj_rgb_paths"]
            traj_cnt_lst = self.data_summary[idx]["traj_cnt_paths"]
            heatmap_folder = self.data_summary[idx]["heatmap_folder"]
            txt = self.data_summary[idx]["txt"]
            task = self.data_summary[idx]["task"]


            traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
            mask_ = get_traj_mask(traj_cnt_lst)
            mask_t = torch.tensor(mask_).to(self.Config.device)
            
            aug_idx_1 = np.array(idx_flip).reshape((1,1)) #np.random.randint(0, 2, (1,1))
            aug_idx_2 = np.array(idx_vtc).reshape((1,1))
            aug_idx_3 = np.array(idx_hr).reshape((1,1))

            aug_idx = np.concatenate([aug_idx_1, aug_idx_2, aug_idx_3], axis = -1)


            traj_cnt_img =  augmentation(aug_idx, torch.stack(traj_cnt_img), rgb = False)
            traj_cnt_img = [traj_cnt_img[0]]

            visual_sentence, fused_x, vl_mask, tp_mask =  self.input_processing(traj_rgb_lst, txt, task, aug_idx = aug_idx, heatmap_folder = heatmap_folder)
        
            self.tot_dataset["visual_sentence"].append( visual_sentence.detach().cpu())
            self.tot_dataset["fused_x"].append( fused_x.detach().cpu())
            self.tot_dataset["vl_mask"].append( vl_mask.detach().cpu())
            self.tot_dataset["aug_idx"].append( aug_idx)
            self.tot_dataset["traj_rgb_paths"].append( traj_rgb_lst)
            self.tot_dataset["traj_cnt_paths"].append( traj_cnt_lst)
            self.tot_dataset["traj_cnt_img"].append( traj_cnt_img)
            self.tot_dataset["traj_cnt_paths"].append( traj_cnt_lst)
            self.tot_dataset["txt"].append( txt)
            self.tot_dataset["task"].append( task)



    def __len__(self):
        return len(self.data_summary) * 18
        # return len(self.data_summary['tot_rgb_flat'])

    def __getitem__(self, idx_raw):
        # idx_lst = [[0, -1, -1], [0, -1, 0], [0, -1, 1],
        #             [0, 0, -1], [0, 0, 0], [0, 0, 1],
        #             [0, 1, -1], [0, 1, 0], [0, 1, 1],
        #            [1, -1, -1], [1, -1, 0], [1, -1, 1],
        #             [1, 0, -1], [1, 0, 0], [1, 0, 1],
        #             [1, 1, -1], [1, 1, 0], [1, 1, 1]]

        # idx = idx_raw // 18
        # idx_flip, idx_vtc, idx_hr  = idx_lst[idx_raw % 18]


        # traj_rgb_lst = self.data_summary[idx]["traj_rgb_paths"]
        # traj_cnt_lst = self.data_summary[idx]["traj_cnt_paths"]
        # heatmap_folder = self.data_summary[idx]["heatmap_folder"]
        # txt = self.data_summary[idx]["txt"]
        # task = self.data_summary[idx]["task"]


        # traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
        # mask_ = get_traj_mask(traj_cnt_lst)
        # mask_t = torch.tensor(mask_).to(self.Config.device)
        
        # aug_idx_1 = np.array(idx_flip).reshape((1,1)) #np.random.randint(0, 2, (1,1))
        # aug_idx_2 = np.array(idx_vtc).reshape((1,1))
        # aug_idx_3 = np.array(idx_hr).reshape((1,1))
        # # aug_idx_2 = np.random.randint(-4, 4, (1,1))
        # # aug_idx_3 = np.random.randint(-2, 2, (1,1))
        # aug_idx = np.concatenate([aug_idx_1, aug_idx_2, aug_idx_3], axis = -1)


        # traj_cnt_img =  augmentation(aug_idx, torch.stack(traj_cnt_img), rgb = False)
        # traj_cnt_img = [traj_cnt_img[0]]
        # # mask_t = augmentation(aug_idx, mask_t, rgb = False) 

        # # print(traj_cnt_img[0].shape, mask_t.shape)
        
        # # print(torch.stack(trajcnt_img).shape)

        #  # inp, txt_emb, vl_mask, tp_mask
        # visual_sentence, fused_x, vl_mask, tp_mask =  self.input_processing(traj_rgb_lst, txt, task, aug_idx = aug_idx, heatmap_folder = heatmap_folder)
        # # print(visual_sentence.shape, fused_x.shape, vl_mask.shape, tp_mask.shape)


        # print(traj_rgb_lst,traj_cnt_lst )
        return   {
                "visual_sentence": torch.tensor(self.tot_dataset["visual_sentence"][idx_raw]) ,
                "fused_x" : torch.tensor(self.tot_dataset["fused_x"][idx_raw]) ,
                "vl_mask": torch.tensor(self.tot_dataset["vl_mask"][idx_raw]) ,
                "tp_mask": torch.tensor(self.tot_dataset["tp_mask"][idx_raw]) ,
                "aug_idx": self.tot_dataset["aug_idx"][idx_raw],
                "traj_rgb_paths": self.tot_dataset["traj_rgb_lst"][idx_raw]  , 
                "traj_cnt_paths": self.tot_dataset["traj_cnt_lst"][idx_raw]  , 
                "traj_cnt_img": self.tot_dataset["traj_cnt_img"][idx_raw], 
                # "mask_t": mask_t, 
                "idx": self.tot_dataset["idx"][idx_raw], 
                "txt": self.tot_dataset["txt"][idx_raw], 
                "task": self.tot_dataset["task"][idx_raw]}


def augmentation( aug_idx, img, rgb = False):
    if aug_idx[0,0] == 1:
        if rgb:
            img = torch.flip(img, (-2,))
        else:
            img = torch.flip(img, (-1,))

    delta = 20 * abs(aug_idx[0,1]) #np.random.randint(20,40)
    
    if aug_idx[0,1] > 0:
        img_ = torch.zeros_like(img)

        if rgb:
            img_[..., :img.shape[-2]-delta,:] = img[..., delta:,:]
        else:
            img_[..., :img.shape[-1]-delta] = img[..., delta:]
        img = img_

    elif aug_idx[0,1] < 0 :
        img_ = torch.zeros_like(img)
        if rgb:
            img_[..., delta:,:] = img[..., :img.shape[-2]-delta,:]
        else:
            img_[..., delta:] = img[..., :img.shape[-1]-delta]
        img = copy.copy(img_)


    delta = 20 * abs(aug_idx[0,2]) #np.random.randint(20,40)
    

    ori_shape = img.shape
    img = img.squeeze()
    if aug_idx[0,2] > 0:
        img_ = torch.zeros_like(img)
        if rgb:
            img_[..., :img.shape[-3]-delta, :,:] = img[..., delta:,:, :]
        else:
            img_[..., :img.shape[-2]-delta,:] = img[..., delta:,:]
        img = img_

    elif aug_idx[0,2] < 0 :
        img_ = torch.zeros_like(img)
        if rgb:
            img_[ delta:, :, :] = img[ :img.shape[-3]-delta,:,:]
        else:
            img_[..., delta:, :] = img[..., :img.shape[-2]-delta, :]
        img = img_
    img = img.reshape(ori_shape)
    return img




