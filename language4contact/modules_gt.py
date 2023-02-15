import os
import torch
import math

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import visualization
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from language4contact.modules_shared import *
from .resnet import ResNet, ResidualBlock
from .Transformer_MM_Explainability.CLIP.clip.simple_tokenizer import (
    SimpleTokenizer as _Tokenizer,
)
from .Transformer_MM_Explainability.CLIP import clip


class policy(nn.Module):
    def __init__(self, device, dim_ft, dim_out):
        super(policy, self).__init__()
        self.dim_ft = dim_ft
        self.device = device

        ## Input processing for Transformer
        self.explainability = ClipExplainability(self.device)
        self._image_encoder = image_encoder(self.device, dim_out=self.dim_ft)

        ## Transformer Encoder
        self.pos_enc = PositionalEncoding(self.dim_ft, dropout=0.1, max_len=50).to(
            self.device
        )
        self.segment_emb = nn.Embedding(3, self.dim_ft).to(self.device)
        # initialize embedding with normal distribution
        nn.init.normal_(self.segment_emb.weight, mean=0.0, std=0.01)
        self.segment_emb.requires_grad_()  # .retain_grad()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_ft, dim_feedforward=128, nhead=1
        )  # default: nhead = 8
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_encoder.to(self.device)

        ## Transformer Decoder
        self.transformer_decoder = contact_mlp_decoder(self.dim_ft, dim_out=dim_out).to(
            self.device
        )

    def input_processing(self, img, texts, return_heatmaps=False):
        ## Encode image and texts with CLIP
        txt_emb, heatmaps = self.explainability.get_heatmap(img, texts)
        seg_idx = [0]
        hm_emb = []

        ## Get clip attention
        img_enc_inp = torch.flatten(heatmaps, 0, 1).unsqueeze(1).float()
        inp = self._image_encoder(img_enc_inp)
        inp = inp.reshape(
            (heatmaps.shape[0], heatmaps.shape[1], inp.shape[-1])
        )  # [batch size x seq x feat_dim]
        seg_idx += [1] * inp.shape[1]
        seg_idx = torch.tensor(seg_idx).repeat(inp.shape[0]).to(self.device)

        if return_heatmaps:
            return inp, seg_idx, heatmaps, 0
        return inp, seg_idx

    def forward(self, feat=None, seg_idx=None, img=None, texts=None):
        pos_enc_simple = position_encoding_simple(
            feat.size()[-2], self.dim_ft, self.device
        )
        feat = pos_enc_simple(feat)

        out = self.transformer_encoder(feat)  # L x dim_ft
        out = torch.amax(out, dim=-2)  # Max pooling
        contact, energy, vel = self.transformer_decoder(out)
        return contact, energy, vel, out


class contact_mlp_decoder(nn.Module):
    def __init__(self, dim_ft, last_activation=False, dim_out=150 ** 2):
        super(contact_mlp_decoder, self).__init__()
        self.dim_out = dim_out
        self.dim_out_sqrt = int(np.sqrt(self.dim_out))
        self.last_activation = last_activation

        self.l1 = nn.Linear(dim_ft, 1024)
        self.l2 = nn.Linear(1024, 1024)

        self.l3_1 = nn.Linear(1024, dim_out)
        self.l3_2 = nn.Linear(1024, dim_out)
        self.l3_3 = nn.Linear(1024, dim_out)

        nn.init.kaiming_normal_(
            self.l1.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l3_1.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l3_2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l3_3.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))

        x_cnt = self.sigmoid(self.l3_1(x)).reshape(
            -1, self.dim_out_sqrt, self.dim_out_sqrt
        )
        x_cost = self.l3_2(x).reshape(-1, self.dim_out_sqrt, self.dim_out_sqrt)
        x_vel = self.l3_3(x).reshape(-1, self.dim_out_sqrt, self.dim_out_sqrt)

        return x_cnt, x_cost, x_vel
