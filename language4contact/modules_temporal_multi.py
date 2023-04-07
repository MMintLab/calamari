import os
import torch

from torch import nn
import numpy as np
from PIL import Image

from .Transformer_MM_Explainability.CLIP import clip 
from language4contact.modules_shared import *
from language4contact.temporal_transformer import TemporalTransformer, PrenormPixelLangEncoder
import language4contact.utils as utils
from typing import Any, Callable, List, Optional, Type, Union
from language4contact.config.config_multi import Config

class policy(nn.Module):
    def __init__(self,  dim_in, dim_out, image_size = 255, Config:Config =None):
        super(policy, self).__init__()
        self.Config = Config
        self.dim_ft = Config.dim_ft
        self.device = Config.device
        self.B = None # Batch size.
        self.L = self.Config.contact_seq_l # Sequence Length.

        ## Input processing for Transformer
        self.explainability = ClipExplainability(self.device)
        self._image_encoder = image_encoder(self.device, dim_in = 1 , dim_out = int(Config.dim_emb))

        ## TODO: Put it temporally
        self.lang_dim_reducer = nn.Linear(Config.dim_emb, 32)
        self.vis_dim_reducer = nn.Linear(Config.dim_emb, 32)
        self.lang_layer_norm = nn.LayerNorm([Config.dim_emb])
        self.vis_layer_norm = nn.LayerNorm([Config.dim_emb])

        nn.init.uniform_(self.lang_dim_reducer.weight, 0, 0.05)
        nn.init.uniform_(self.lang_dim_reducer.bias, 0, 0.05)
        nn.init.uniform_(self.vis_dim_reducer.weight, 0, 0.05)
        nn.init.uniform_(self.vis_dim_reducer.bias, 0, 0.05)




        ## Transformer Encoder
        self.pos_enc = PositionalEncoding(self.dim_ft, dropout=0.1, max_len=30).to(self.device)

        # # vision language transformer.
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_ft, dim_feedforward=128, nhead=1) #default: nhead = 8
        # self.vl_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.vl_transformer_encoder.to(self.device)

        self.vl_transformer_encoder = PrenormPixelLangEncoder(num_layers=2,num_heads=2, dropout_rate=0.1, mha_dropout_rate=0.0,
          dff=self.dim_ft, device= self.Config.device)
        self.vl_transformer_encoder = self.vl_transformer_encoder.to(torch.float)


        self.transformer_decoder = contact_mlp_decoder(self.dim_ft, dim_out = dim_out).to(self.device)

        # temporal transformer.
        self.tp_transformer = TemporalTransformer(dim_in=self.dim_ft, d_model = self.dim_ft, sequence_length = self.L, device= self.device)
        # self.model_grd = [self._image_encoder, self.vl_transformer_encoder,  self.transformer_decoder, self.tp_transformer]

        self.text_embs = self._get_text_emb()

    def _get_text_emb(self):
        txt_emb = {}
        for task_n, task in self.Config.dataset_config.items():
            texts = task["txt_cmd"]
            words = sentence2words(texts)
            words.insert(0, texts)
            text = clip.tokenize(words).to(self.device)
            txt_emb_i = self.explainability.model.encode_text(text)
            txt_emb[task_n] = txt_emb_i.detach()

        return txt_emb


    def image_segment(self, pil_img, patch_size = 32):
        patches = pil_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.reshape( -1, patch_size, patch_size)
        return patches


    def read_heatmap_temporal(self, img_batch_pths, texts, tasks):

        words = sentence2words(texts)
        if self.Config.heatmap_type == 'chefer':
            cnt_dir = 'heatmap/'
            for row_idx, row in enumerate(words):
                row.insert(0, texts[row_idx])

        elif self.Config.heatmap_type == 'huy':
            cnt_dir = 'heatmap_huy/'
        
        # text = clip.tokenize(words).to(self.device)

        ## image processing
        txt_emb_batch = []
        heatmaps_batch = []
        img_batch = []
        tp_masks = []
        vl_masks = torch.ones((self.B, self.L, self.Config.max_sentence_l))
        txt_emb_batch = torch.zeros((self.B, self.L, self.Config.max_sentence_l, 512)).to(self.device)
        heatmaps_batch = torch.zeros((self.B, self.L, self.Config.max_sentence_l, self.Config.heatmap_size[0], self.Config.heatmap_size[1])).to(self.device)

        # print(img_batch_pths)
        for b, img_pths in enumerate(img_batch_pths):
            tp_mask = []
            words_i = words[b]
            task_i = tasks[b]

            # Read Texts as a token.
            txt_emb_i = torch.zeros((self.Config.max_sentence_l, 512)).to(self.device)
            txt_emb_i[:len(self.text_embs[task_i])] = self.text_embs[task_i]

            txt_emb_batch[b] = txt_emb_i.unsqueeze(0).repeat((txt_emb_batch.shape[1], 1, 1)) 


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
                            heatmap = Image.open(hm_pth).resize((self.Config.heatmap_size[0], self.Config.heatmap_size[1]))
                            heatmap = torch.tensor( np.array(heatmap)).to(self.device)
                            heatmaps.append(heatmap)

                            # vl transformer mask = 0 : Real Input.
                            vl_masks[b, -(l+1),i] = 0
                        
                        else:
                            # Fake (Padding) VL-transformer Inputs.
                            padding = torch.zeros_like(heatmaps[-1]).to(self.device)
                            heatmaps.append(padding)
                            vl_masks[b, -(l+1),i] = 1

                    # Real Temporal-transformer Inputs.   
                    # txt_emb_batch.insert(0, txt_emb_i)
                    # txt_emb_batch[b,-(l+1),:] = txt_emb_i
                    heatmaps_batch[b,-(l+1),:] = torch.stack(heatmaps)
                    # heatmaps_batch.insert( 0,torch.stack(heatmaps))
                    tp_mask.insert(0, 0)

                elif len(img_pth) == 0: 
                    # Fake (Padding) Temporal-transformer Inputs.
                    # txt_emb_i = torch.zeros_like(txt_emb_batch[0,0,...]).to(self.device)
                    # heatmaps = torch.zeros_like(heatmaps_batch[0,0,...]).to(self.device)
                    # txt_emb_batch.insert(0,txt_emb_i)
                    # heatmaps_batch.insert(0,heatmaps)
                    tp_mask.insert(0,1)

            tp_masks.append(tp_mask)

        # Formatting.
        # heatmap_batch_ = torch.stack(heatmaps_batch)
        heatmap_batch_ = torch.flatten(heatmaps_batch, 0, 1)
        vl_masks = torch.flatten(vl_masks, 0, 1).to(self.device) # Match heatmap size.
        tp_masks = torch.tensor(tp_masks).to(self.device)
        return txt_emb_batch, heatmap_batch_, vl_masks.bool(), tp_masks.bool()
    
    def input_processing(self, img, texts, tasks, mode = 'train'):
        """
        mode = 'train' : load heatmap
        mode = 'test' : generate heatmap every time step
        """
        # txt_emb, heatmaps, padding_mask = self.explainability.get_heatmap_temporal(img, texts)
        self.B = len(img)
        txt_emb, heatmaps, vl_mask, tp_mask = self.read_heatmap_temporal(img, texts, tasks) 

        """
        heatmaps : (B*l_temp) X l_lang x 225x225
        """
        
        hm_emb = []

        ## Get clip attention
        heatmaps_ = heatmaps[ ~vl_mask[:,0]]
        img_enc_inp = torch.flatten(heatmaps_, 0, 1).unsqueeze(1).float()
        out = self._image_encoder(img_enc_inp)

        inp = torch.zeros((heatmaps.shape[0], heatmaps.shape[1], out.shape[-1])).to(self.Config.device)
        inp[ ~vl_mask[:,0]] = out.reshape((heatmaps_.shape[0], heatmaps.shape[1], out.shape[-1]))


        # breakpoint()
        # print(img_enc_inp)

        # # inp = torch.where(torch.flatten(vl_mask, 0, 1).repeat((1, inp.shape[1])), inp, torch.zeros_like(inp).to(self.Config.device))
        # vl_mask_= vl_mask.unsqueeze(1).repeat((1, inp.shape[1])) # 1 padding
        # inp = torch.where(vl_mask_, torch.ones_like(inp).to(self.Config.device),  inp)

        # print(vl_mask.tolist(), np.sum(heatmaps.detach().cpu().numpy(), axis = (-1, -2)).tolist())


        inp = inp.reshape(( heatmaps.shape[0], heatmaps.shape[1], inp.shape[-1])) # [batch size x seq x feat_dim]
        
        # print(inp.detach().cpu().numpy()[:,:,0].tolist())

        # print(inp)
        # print("lang", txt_emb)
        return inp, txt_emb, vl_mask, tp_mask


    def forward_lava(self, visual_sentence, fused_x, vl_mask, tp_mask):
        # visual_sentence = visual_sentence.permute((1,0,2)) # pytorch transformer takes L X (B * l_contact_seq) X ft
        # fused_x = fused_x.permute((1,0,2)) # pytorch transformer takes L X (B * l_contact_seq) X ft
        
        # pos_enc_simple = position_encoding_simple(K= feat.size()[0], M=self.dim_ft, device=self.device) 
        # feat = pos_enc_simple(feat)
        # print(vl_mask.shape)

        # visual_sentence = visual_sentence[bidx, lidx].reshape(vl_mask.shape[0], -1)
        # fused_x = fused_x[bidx, lidx].reshape(vl_mask.shape[0], -1)
        # img_enc_inp = torch.flatten(heatmaps, 0, 1)[bidx].unsqueeze(1).float()
        # inp_[bidx] = inp

        visual_sentence = visual_sentence[ ~vl_mask[:,0]]
        fused_x = fused_x[ ~vl_mask[:,0]]
        vl_mask_ = vl_mask[ ~vl_mask[:,0]]

        visual_sentence = self.vis_layer_norm(visual_sentence)
        visual_sentence = self.vis_dim_reducer(visual_sentence)
        
        fused_x = self.lang_layer_norm(fused_x)
        fused_x = self.lang_dim_reducer(fused_x)

        # out = self.vl_transformer_encoder(visual_sentence, fused_x, padding_mask = vl_mask_) # L x (B * l_contact_seq) X ft
        # out = torch.mean(out, axis = 1) # TODO: remove?
        
        out = torch.zeros((self.B * self.L, visual_sentence.shape[-1])).to(self.Config.device)
        out[~vl_mask[:,0]] = torch.mean(self.vl_transformer_encoder(visual_sentence, fused_x, padding_mask = vl_mask_), axis = 1)
        out = out.reshape((self.B , self.L, visual_sentence.shape[-1]))

        tp_mask_tmp= tp_mask.unsqueeze(2).repeat((1, 1, out.shape[-1])) # 1 padding
        out = torch.where(tp_mask_tmp, torch.zeros_like(out).to(self.Config.device),  out)

        tp_output = self.tp_transformer(out, padding_mask = tp_mask)
        contact = self.transformer_decoder(tp_output) # (B * seq_l) X w x h

        w = int(np.sqrt(contact.shape[-1]))
        contact = contact.reshape((self.B, w, w ))
        return contact




class contact_mlp_decoder(nn.Module):
    def __init__(self, dim_ft, last_activation = False, dim_out = 150**2):
        super(contact_mlp_decoder, self).__init__()
        self.last_activation = last_activation

        self.l1 = nn.Linear(dim_ft, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 2048)
        # self.l4 = nn.Linear(2048, 4096)
        self.l4 = nn.Linear(2048, dim_out)

        # nn.init.uniform_(self.l3.weight, 0, 0.05)
        # nn.init.uniform_(self.l3.bias, 0, 0.05)


        # nn.init.kaiming_uniform_(self.l1.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # nn.init.kaiming_uniform_(self.l2.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # nn.init.kaiming_uniform_(self.l3.weight, a=0.0, nonlinearity='relu', mode='fan_in')

        # kaiming normal best compatible with tanh
        nn.init.kaiming_normal_(self.l1.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(self.l2.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(self.l3.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(self.l4.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # nn.init.kaiming_normal_(self.l5.weight, a=0.0, nonlinearity='relu', mode='fan_in')


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        # x = self.relu(self.l4(x))

        x_cnt = self.sigmoid(self.l4(x))


        return x_cnt

