import os 
from pathlib import Path

import math
import torch
import torch.utils.data as data_utils
import torchvision.transforms as T

from language4contact.modules_shared import *
from language4contact.utils import *
from .Transformer_MM_Explainability.CLIP import clip 

from tqdm import tqdm


class DatasetTemporal(torch.utils.data.Dataset):
    def __init__(self, Config, mode = 'train', seq_l = -1, image_encoder = None):
        self.Config = Config
        self.contact_seq_l = self.Config.contact_seq_l
        self.mode = mode 
        self.return_seq_l = seq_l
        self.N_noise_aug = 5
        # get language embeddings
        if mode == 'train':
            self.idx_lst = [[0, -1, -1], [0, -1, 0], [0, -1, 1],
                        [0, 0, -1], [0, 0, 0], [0, 0, 1],
                        [0, 1, -1], [0, 1, 0], [0, 1, 1],
                    [1, -1, -1], [1, -1, 0], [1, -1, 1],
                        [1, 0, -1], [1, 0, 0], [1, 0, 1],
                        [1, 1, -1], [1, 1, 0], [1, 1, 1],
                        [0, -1, -1], [0, -1, 0], [0, -1, 1],
                        [0, 0, -1], [0, 0, 1],
                        [0, 1, -1], [0, 1, 0], [0, 1, 1],
                    [1, -1, -1], [1, -1, 0], [1, -1, 1],
                        [1, 0, -1], [1, 0, 0], [1, 0, 1],
                        [1, 1, -1], [1, 1, 0], [1, 1, 1]]
            self.idx_lst =  self.idx_lst * self.N_noise_aug
        elif mode == 'test':
            self.idx_lst = [[0, 0, 0]]
        
        # self.idx_lst = [[0, 0, 0] for _ in range(2)]

        # self.txt_cmd = self.Config.txt_cmd
        
        # if self.mode == "train":
        #     self.folder_idx = self.Config.train_idx
        # if self.mode == "test":
        #     self.folder_idx = self.Config.test_idx

        # pretrained model.

        self._image_encoder = image_encoder 
        self.explainability = ClipExplainability(self.Config.device)

        self.text_embs = self._get_text_emb()
        self.contact_folder = self.Config.contact_folder  #'contact_key_front'
        # self.data_dir = self.Config.data_dir
        self.data_summary, self.aug_summary = self.get_data_summary()
        # self.cnt_w, self.cnt_h = self.get_cnt_img_dim()





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
        aug_summary = {}
        '''
        {idx :{ rgb history path list: List[str], txt command: str, contact goal path: str}}
        '''
        
        tot_length = 0
        for task_n, task in tqdm(self.Config.dataset_config.items()):
            txt_cmd = task["txt_cmd"]
            heatmap_folder = task["heatmap_folder"]
            for i in tqdm(task[self.mode+"_idx"]):
                # cnt_folder_path = f'{task["data_dir"]}/t_{i:03d}/contact_front'
                # rgb_folder_path = f'{task["data_dir"]}/t_{i:03d}/rgb'
                cnt_folder_path = f'{task["data_dir"]}/episode{i}/contact_front'
                rgb_folder_path = f'{task["data_dir"]}/episode{i}/rgb'

                traj_cnt_fn = folder2filelist(cnt_folder_path)
                traj_cnt_fn.sort()

                traj_rgb_fn = folder2filelist(rgb_folder_path)
                traj_rgb_fn.sort()

                ## TODO: remove it 
                # traj_cnt_fn = traj_cnt_fn[1:]
                # traj_rgb_fn = traj_rgb_fn[1:]

                traj_rgb_fn = traj_rgb_fn[:-1] # pop last obs

                for local_idx in range(len(traj_rgb_fn)):

                    # t-3, t-2, t-1, t RGB
                    if local_idx < self.contact_seq_l:
                        traj_rgb_lst = ['','','','']
                        # traj_rgb_lst[:local_idx+1] = list( reversed(traj_rgb_fn[:local_idx+1]))
                        # traj_rgb_lst[-(local_idx+1):] = list( reversed(traj_rgb_fn[:local_idx+1]))
                        traj_rgb_lst[-(local_idx+1):] =  traj_rgb_fn[:local_idx+1]

                        traj_cnt_lst = [traj_cnt_fn[local_idx]]
                    else:
                        traj_rgb_lst = list( reversed(traj_rgb_fn[local_idx-3:local_idx+1]))
                        traj_cnt_lst = [traj_cnt_fn[local_idx]]

                    # print(traj_rgb_lst, traj_cnt_lst )
                    # breakpoint()
                    # Save.
                    data_summary[tot_length] = {
                                                "traj_rgb_paths": traj_rgb_lst,
                                                "traj_cnt_paths": traj_cnt_lst, 
                                                "txt": txt_cmd,
                                                "task": task_n,
                                                "heatmap_folder":heatmap_folder,
                                                }
                    for d_idx, aug_idx in enumerate(self.idx_lst):
                        aug_length = len(self.idx_lst) * tot_length + d_idx
                        
                        txt = txt_cmd

                        traj_cnt_img = fn2img(traj_cnt_lst, d = 1)
                        mask_ = get_traj_mask(traj_cnt_lst)
                        self.cnt_w, self.cnt_h = mask_.shape
                        mask_t = torch.tensor(mask_).to(self.Config.device)
                        
                        aug_idx= np.array(aug_idx)
                        aug_idx[1] = aug_idx[1] * np.random.randint(10,30)
                        aug_idx[2] = aug_idx[2] * np.random.randint(10,30)
                        # aug_idx = aug_idx[np.newaxis,:]


                        # scale augmentation
                        # scale = np.random.random() * 0.3 + 0.85
                        scale = 1
                        aug_idx = np.append(aug_idx, scale)

                        if self.mode == 'train':
                            traj_cnt_img =  augmentation(aug_idx, torch.stack(traj_cnt_img), rgb = False)   
                        traj_cnt_img = [traj_cnt_img[0].detach().cpu()]

                        noise_idx = d_idx % self.N_noise_aug
                        heatmap_ft, text_emb, vl_mask, tp_mask =  self.input_processing(traj_rgb_lst,
                                                                                         txt, task_n, 
                                                                                         aug_idx = aug_idx, 
                                                                                         heatmap_folder = heatmap_folder,
                                                                                         noise_idx = noise_idx)
                        
                        
                        # print(traj_rgb_lst, vl_mask, tp_mask, torch.sum(visual_sentence, dim = -1) , torch.sum(fused_x, dim = -1) )

                        # breakpoint()
                        aug_summary[aug_length] = {
                            "query": text_emb.detach().cpu(),
                            "key" : heatmap_ft.detach().cpu(),
                            "vl_mask": vl_mask.detach().cpu(),
                            "tp_mask":tp_mask.detach().cpu(),
                            "traj_cnt_img": traj_cnt_img, 
                            "aug_idx": torch.tensor(aug_idx),

                        }



                    # Increase idx.
                    tot_length += 1
        # print(data_summary)
        # breakpoint()
        # print(aug_summary)
        # breakpoint()
        return data_summary, aug_summary


    def read_heatmap_temporal(self, img_pths, texts, task_i, aug_idx_i, heatmap_folder = None, noise_idx = None):
        # print(self.mode, img_pths, heatmap_folder, aug_idx_i, noise_idx)

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
                        hm_pth = os.path.join(hm_pth, wd + '.png') if noise_idx is None else os.path.join(hm_pth, wd + f'_{noise_idx}.png')
                        heatmap = Image.open(hm_pth)
                        heatmap = torch.tensor(np.array(heatmap))
                        # heatmap = Image.fromarray(augmentation(aug_idx_i, heatmap, rgb= False).numpy())
                        
                        # heatmap = hehttps://studio.youtube.com/channel/UCLNmkcUHL0f9nGZ4YoV-4Yw/videos/upload?filter=%5B%5D&sort=%7B%22columnType%22%3A%22date%22%2C%22sortOrder%22%3A%22DESCENDING%22%7Dh.tensor( np.array(heatmap)).to(self.Config.device)


                        # print(heatmap.shape)
                        # breakpoint()
                        if self.mode == 'train':
                            heatmap = augmentation(aug_idx_i, heatmap, rgb= False, device=self.Config.device)
                        heatmaps.append(heatmap.to(self.Config.device))

                        # vl transformer mask = 0 : Real Input.
                        # vl_masks[-(l+1),i] = 0
                        vl_masks[l,i] = 0

                    
                    else:
                        # Fake (Padding) VL-transformer Inputs.
                        padding = torch.zeros_like(heatmaps[-1]).to(self.Config.device)
                        heatmaps.append(padding)
                        # vl_masks[ -(l+1),i] = 1
                        vl_masks[l,i] = 1


                # Real Temporal-transformer Inputs.   
                heatmaps_batch[l,:] = torch.stack(heatmaps)
                tp_mask.append(0)

            elif len(img_pth) == 0: 
                tp_mask.append(1)


        # Formatting.
        # heatmap_batch_ = torch.stack(heatmaps_batch)
        # heatmaps_batch = torch.flatten(heatmaps_batch, 0, 1)
        # txt_emb_batch = torch.flatten(txt_emb_batch, 0, 1)
        vl_masks = vl_masks.to(self.Config.device) # Match heatmap size.
        # vl_masks = torch.flatten(vl_masks, 0, 1).to(self.Config.device) # Match heatmap size.
        tp_masks = torch.tensor(tp_mask).to(self.Config.device)
        return txt_emb_batch, heatmaps_batch, vl_masks.bool(), tp_masks.bool()
    
    def input_processing(self, img, texts, tasks, mode = 'train', aug_idx = None, heatmap_folder = None, noise_idx = None, scale = 1):
        """
        mode = 'train' : load heatmap
        mode = 'test' : generate heatmap every time step
        """
        # txt_emb, heatmaps, padding_mask = self.explainability.get_heatmap_temporal(img, texts)
        self.B = len(img)
        # if aug_idx is None:
        #     aug_idx = torch.zeros((self.B, 1))

       
        text_emb, heatmaps, vl_mask, tp_mask = self.read_heatmap_temporal(img, texts, tasks, aug_idx, heatmap_folder = heatmap_folder,  noise_idx = noise_idx) 
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
        heatmap_ft = torch.zeros((heatmaps.shape[0], heatmaps.shape[1], out.shape[-1])).to(self.Config.device)
        # print(inp.shape, heatmaps_.shape, out.shape)
        # inp[ ~vl_mask] = out #.reshape((self.Config.contact_seq_l, self.Config.max_sentence_l, -1))
        heatmap_ft[ ~vl_mask[:,0]] = out.reshape((heatmaps_.shape[0], heatmaps.shape[1], out.shape[-1]))
        


        # inp = inp.reshape(( heatmaps.shape[0], heatmaps.shape[1], out.shape[-1])) # [batch size x seq x feat_dim]
        return heatmap_ft, text_emb, vl_mask, tp_mask
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

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_summary) * len(self.idx_lst)
        if self.mode =='test':
            return len(self.data_summary)
        # return len(self.data_summary['tot_rgb_flat'])

    def __getitem__(self, idx_raw):
        if self.mode == 'train':
            idx_lst = self.idx_lst
            idx = idx_raw // len(self.idx_lst)
        # idx_flip, idx_vtc, idx_hr  = idx_lst[idx_raw % len(self.idx_lst)]

        if self.mode == 'test':
            idx = idx_raw


        traj_rgb_lst = self.data_summary[idx]["traj_rgb_paths"]
        traj_cnt_lst = self.data_summary[idx]["traj_cnt_paths"]
        heatmap_folder = self.data_summary[idx]["heatmap_folder"]
        txt = self.data_summary[idx]["txt"]
        task = self.data_summary[idx]["task"]

        print(self.Config.device)
        return   {
                "query": self.aug_summary[idx_raw]['query'].to(self.Config.device),
                "key" : self.aug_summary[idx_raw]['key'].to(self.Config.device),
                "vl_mask": self.aug_summary[idx_raw]['vl_mask'].to(self.Config.device),
                "tp_mask":self.aug_summary[idx_raw]['tp_mask'].to(self.Config.device),
                "traj_cnt_img": [i.to(self.Config.device) for i in self.aug_summary[idx_raw]['traj_cnt_img']], 
                "aug_idx": self.aug_summary[idx_raw]['aug_idx'],

                # "aug_idx": torch.tensor(self.aug_summary[idx_raw]['aug_idx']),
                "traj_rgb_paths": traj_rgb_lst, 
                "traj_cnt_paths": traj_cnt_lst, 
                # "mask_t": mask_t, 
                "idx": idx_raw, 
                "txt": txt, "task": task}
    

def augmentation( aug_idx, img, rgb = False, device = 'cpu'):
    flip = aug_idx[0]
    delta_x  = int(aug_idx[1])
    delta_y = int(aug_idx[2])

    rgb = True if img.shape[-1] == 3 else False

    if np.sum(aug_idx) != 0 :
        img_shape_ori = img.shape
        w = img.shape[-2]
        s = aug_idx[-1]

        transform = T.Resize(size = (int(w*s), int(w*s)))
        # print("cpt0", img.squeeze().unsqueeze(0).shape)
        if rgb:
            img = transform(img.permute((2, 0, 1))) 
            img = img.permute((1,2,0))

        else:
            img = transform(img.squeeze().unsqueeze(0))
        if s>1:
            if rgb:
                img = img[-w:, (img.shape[-2] - w)//2: (img.shape[-2] - w)//2 + w, :]
                img = img.reshape(img_shape_ori)
            else:
                img = img[..., -w:, (img.shape[-2] - w)//2: (img.shape[-2] - w)//2 + w]
                img = img.reshape(img_shape_ori)
        else:
            if rgb:
                img_ = torch.zeros(img_shape_ori).to(device)
                img_[-img.shape[-2]:, 
                    -(img.shape[-2] - w)//2: -(img.shape[-2] - w)//2 + img.shape[-2], :]=img
                img = img_.reshape(img_shape_ori)
            else:
                img_ = torch.zeros(img_shape_ori).to(device)
                img_[...,-img.shape[-2]:, 
                    -(img.shape[-2] - w)//2: -(img.shape[-2] - w)//2 + img.shape[-1]]=img.squeeze()
                img = img_.reshape(img_shape_ori)

        # print(img.shape, w, h)
        # img = cv2.resize(img.detach().cpu().numpy(), (w//2, h), fx=s, fy=s)
            # print(img)
            # img = torch.tensor(img).to('cuda') #TODO
        # print(img.shape)

    if flip == 1:
        if rgb:
            img = torch.flip(img, (-2,))
        else:
            img = torch.flip(img, (-1,))

    
    if delta_x > 0:
        img_ = torch.zeros_like(img)

        if rgb:
            img_[..., :img.shape[-2]-delta_x,:] = img[..., delta_x:,:]
        else:
            img_[..., :img.shape[-1]-delta_x] = img[..., delta_x:]
        img = img_

    elif delta_x < 0 :
        img_ = torch.zeros_like(img)
        if rgb:
            img_[..., abs(delta_x):,:] = img[..., :img.shape[-2]-abs(delta_x),:]
        else:
            img_[..., abs(delta_x):] = img[..., :img.shape[-1]-abs(delta_x)]
        img = copy.copy(img_)

    

    ori_shape = img.shape
    img = img.squeeze()
    if delta_y > 0:
        img_ = torch.zeros_like(img)
        if rgb:
            img_[..., :img.shape[-3]-delta_y, :,:] = img[..., delta_y:,:, :]
        else:
            img_[..., :img.shape[-2]-delta_y,:] = img[..., delta_y:,:]
        img = img_

    elif delta_y < 0 :
        img_ = torch.zeros_like(img)
        if rgb:
            img_[ abs(delta_y):, :, :] = img[ :img.shape[-3]- abs(delta_y),:,:]
        else:
            img_[...,  abs(delta_y):, :] = img[..., :img.shape[-2]- abs(delta_y), :]
        img = img_
    img = img.reshape(ori_shape)

    # TODO: consider formatting
    # Add Noise to the scale
    # if not rgb:
    #     if delta_x ==0 and delta_y ==0:
    #         pass
    #     else:
    #         rand_scale = np.random.rand(1)[0] * 0.4 + 0.8 # 0.7 ~ 1.3
    #         img = img * rand_scale
    return img.to(torch.uint8) 


# def augmentation( aug_idx, img, rgb = False):
#     if aug_idx[0,0] == 1:
#         if rgb:
#             img = torch.flip(img, (-2,))
#         else:
#             img = torch.flip(img, (-1,))

#     delta = 20 * abs(aug_idx[0,1]) #np.random.randint(20,40)
    
#     if aug_idx[0,1] > 0:
#         img_ = torch.zeros_like(img)

#         if rgb:
#             img_[..., :img.shape[-2]-delta,:] = img[..., delta:,:]
#         else:
#             img_[..., :img.shape[-1]-delta] = img[..., delta:]
#         img = img_

#     elif aug_idx[0,1] < 0 :
#         img_ = torch.zeros_like(img)
#         if rgb:
#             img_[..., delta:,:] = img[..., :img.shape[-2]-delta,:]
#         else:
#             img_[..., delta:] = img[..., :img.shape[-1]-delta]
#         img = copy.copy(img_)


#     delta = 20 * abs(aug_idx[0,2]) #np.random.randint(20,40)
    

#     ori_shape = img.shape
#     img = img.squeeze()
#     if aug_idx[0,2] > 0:
#         img_ = torch.zeros_like(img)
#         if rgb:
#             img_[..., :img.shape[-3]-delta, :,:] = img[..., delta:,:, :]
#         else:
#             img_[..., :img.shape[-2]-delta,:] = img[..., delta:,:]
#         img = img_

#     elif aug_idx[0,2] < 0 :
#         img_ = torch.zeros_like(img)
#         if rgb:
#             img_[ delta:, :, :] = img[ :img.shape[-3]-delta,:,:]
#         else:
#             img_[..., delta:, :] = img[..., :img.shape[-2]-delta, :]
#         img = img_
#     img = img.reshape(ori_shape)
#     return img




