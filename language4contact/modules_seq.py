import os
import torch
import math
import time

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import visualization
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from .Transformer_MM_Explainability.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .Transformer_MM_Explainability.CLIP import clip 
from language4contact.modules_shared import *
import language4contact.utils as utils
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from .resnet import ResNet, ResidualBlock

class policy(nn.Module):
    def __init__(self, device, dim_ft, dim_in, dim_out, image_size = 255):
        super(policy, self).__init__()
        self.dim_ft = dim_ft
        self.device = device
        self.input_length = (1, 20, 20) # (sentence, words, heatmap)

        ## Input processing for Transformer
        self.explainability = ClipExplainability(self.device)
        # self._image_encoder = image_encoder(self.device, dim_in = dim_in, dim_out = self.dim_ft)
        # self.dim_in = 56
        # self._image_encoder = image_encoder_mlp(self.device, dim_in = self.dim_in **2 , dim_out = self.dim_ft)
        self._image_encoder = image_encoder(self.device, dim_in = 1 , dim_out = self.dim_ft)


        ## Transformer Encoder
        self.pos_enc = PositionalEncoding(self.dim_ft, dropout=0.1, max_len=50).to(self.device)
        self.segment_emb = nn.Embedding(3, self.dim_ft).to(self.device)
        
        # initialize embedding with normal distribution
        nn.init.normal_(self.segment_emb.weight,mean = 0. , std=0.01)
        self.segment_emb.requires_grad_() #.retain_grad()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_ft, dim_feedforward=128, nhead=1) #default: nhead = 8
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_encoder.to(self.device)

        ## Transformer Decoder
        self.transformer_decoder = contact_mlp_decoder(self.dim_ft, dim_out = dim_out).to(self.device)

    def image_segment(self, pil_img, patch_size = 32):
        patches = pil_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.reshape( -1, patch_size, patch_size)
        return patches



    def input_processing_from_heatmap(self, heatmap_folder:str, sentence: List[str]):

        heatmaps_batch = []
        img_batch = []

        ## Encode image and texts with CLIP
        text_token = clip.tokenize(sentence).to(self.device)
        text_emb = self.explainability.model.encode_text(text_token)
        texts = utils.sentence2words(sentence)

        heat_map_batch = []
        src_padding_mask_batch = []
        for hm_folder_i in heatmap_folder:
            heatmaps = []
            src_padding_masks = [0]
            for idx, txt in enumerate(texts):
                pilimage = torch.tensor(np.array(Image.open(os.path.join(hm_folder_i, txt + ".png")).resize((224, 224))))
                # patches = self.image_segment(pilimage, patch_size=self.dim_in)
                # hm = torch.tensor(patches)
                heatmaps.append(pilimage) 

                # # All zero image = Padding
                if torch.sum(pilimage) ==0 :
                    src_padding_masks.append(1)
                else:
                    src_padding_masks.append(0)
            heat_map_batch.append(torch.stack(heatmaps, dim = 0))
            # heat_map_batch.append(torch.cat(heatmaps, dim = 0))
            src_padding_mask_batch.append(src_padding_masks)


        # print("loading heatmap takes:", time.time() - start, "[s]") -> 5ms per 1 scene
            # print(torch.amax(heat_map_batch[-1]), torch.amin(heat_map_batch[-1]))
        heat_map_batch = torch.stack(heat_map_batch).to(self.device)
        heat_map_batch = utils.image_reg_255(heat_map_batch)
        # heat_map_batch  = torch.flatten(heat_map_batch, 1, 2).float()
        
        # heat_map_batch  = torch.flatten(heat_map_batch, 1, 2).unsqueeze(1).float()

        seg_idx = [0]
        hm_emb = []

        ## Get clip attention
        img_enc_inp_2 = torch.flatten(heat_map_batch, 0, 1).unsqueeze(1).float() #[320, 1, 256, 256]
        # img_enc_inp_1 = torch.flatten(heat_map_batch, 0, 1).float() #[320, 1, 256, 256]
        # img_enc_inp_2 = torch.flatten(img_enc_inp_1, 1, 2).float() #[320, 1, 256, 256]

        
        inp = self._image_encoder(img_enc_inp_2)
        inp = inp.reshape(( heat_map_batch.shape[0], heat_map_batch.shape[1], inp.shape[-1])) # [batch size x seq x feat_dim]
        
        text_emb = text_emb.unsqueeze(0).repeat((heat_map_batch.shape[0], 1, 1))
        inp = torch.cat([text_emb, inp], dim = 1)
        
        seg_idx += [1] * heat_map_batch.shape[1]
        seg_idx = torch.tensor(seg_idx).repeat(heat_map_batch.shape[0]).to(self.device)

        return inp, seg_idx, text_token, torch.tensor(src_padding_mask_batch).bool().to(self.device)

    def input_processing(self, img, texts):
        ## Encode image and texts with CLIP
        # torch.Size([9, 512]), ([40, 9, 224, 224]) 
        txt_emb, heatmaps = self.explainability.get_heatmap(img, texts)
        seg_idx = [0]
        hm_emb = []

        ## Get clip attention
        img_enc_inp = torch.flatten(heatmaps, 0, 1).unsqueeze(1).float()
        inp = self._image_encoder(img_enc_inp)
        inp = inp.reshape(( heatmaps.shape[0], heatmaps.shape[1], inp.shape[-1])) # [batch size x seq x feat_dim]
        seg_idx += [1] * inp.shape[1]
        seg_idx = torch.tensor(seg_idx).repeat(inp.shape[0]).to(self.device)

        return inp, seg_idx


    def forward(self, feat, seg_idx = None, txt_token = None, src_key_padding_mask = None):

        feat = feat.permute((1,0,2)) # pytorch transformer takes L X B X ft
        pos_enc_simple = position_encoding_simple(feat.size()[0], self.dim_ft, self.device) 
        feat = pos_enc_simple(feat)


        # print(feat.shape, src_key_padding_mask)

        # transformer src = (S, N, E)
        if src_key_padding_mask is not None:
            out = self.transformer_encoder(feat, src_key_padding_mask = src_key_padding_mask)
            # out = self.transformer_encoder(hidden_states  = img, attention_mask = src_key_padding_mask) # L x dim_ft
            # out = self.transformer_encoder(input_ids = txt_token, pixel_values  = img, attention_mask = src_key_padding_mask) # L x dim_ft
        else:
            out = self.transformer_encoder(feat)
        contact_seq = self.transformer_decoder(out)
        return contact_seq



class contact_mlp_decoder(nn.Module):
    def __init__(self, dim_ft, last_activation = False, dim_out = 150**2):
        super(contact_mlp_decoder, self).__init__()
        self.last_activation = last_activation

        self.l1 = nn.Linear(dim_ft, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, dim_out)

        nn.init.kaiming_normal_(self.l1.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(self.l2.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(self.l3.weight, a=0.0, nonlinearity='relu', mode='fan_in')

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x_cnt = self.sigmoid(self.l3(x))

        return x_cnt

