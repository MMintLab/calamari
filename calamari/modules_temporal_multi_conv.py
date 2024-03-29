import os
import torch

from torch import nn
import numpy as np
from PIL import Image

# from .Transformer_MM_Explainability.CLIP import clip
from calamari.modules_shared import *
from calamari.temporal_transformer import TemporalTransformer, PrenormPixelLangEncoder
import calamari.utils as utils
from typing import Any, Callable, List, Optional, Type, Union
from calamari.config.config_multi_conv import Config
from calamari.unet import UNet_Decoder


class policy(nn.Module):
    def __init__(
        self, dim_in, dim_out, image_size=255, Config: Config = None, device=None
    ):
        super(policy, self).__init__()
        self.Config = Config
        self.dim_ft = Config.dim_ft
        self.device = device
        self.B = None  # Batch size.
        self.L = self.Config.contact_seq_l  # Sequence Length.

        ## Input processing for Transformer
        self.explainability = ClipExplainability(self.device)

        Config.dim_emb = 512  # TODO: For now
        # self._image_encoder = image_encoder(self.device, dim_in = 1 , dim_out = int(Config.dim_emb))

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
        self.pos_enc = PositionalEncoding(Config.dim_emb, dropout=0.1, max_len=30).to(
            self.device
        )

        self.vl_transformer_encoder = PrenormPixelLangEncoder(
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            mha_dropout_rate=0.0,
            dff=Config.dim_emb,
            device=self.device,
        )
        self.vl_transformer_encoder = self.vl_transformer_encoder.to(torch.float).to(
            self.device
        )

        # self.transformer_decoder = contact_mlp_decoder(self.dim_ft, dim_out = dim_out).to(self.device)
        self.transformer_decoder = UNet_Decoder().to(torch.float).to(self.device)

        # temporal transformer.
        self.tp_transformer = (
            TemporalTransformer(
                dim_in=Config.dim_emb,
                d_model=Config.dim_emb,
                sequence_length=self.L,
                device=self.device,
            )
            .to(torch.float)
            .to(self.device)
        )
        # self.model_grd = [self._image_encoder, self.vl_transformer_encoder,  self.transformer_decoder, self.tp_transformer]

    #     self.text_embs = self._get_text_emb()

    # def _get_text_emb(self):
    #     txt_emb = {}
    #     for task_n, task in self.Config.dataset_config.items():
    #         texts = task["txt_cmd"]
    #         words = sentence2words(texts)
    #         words.insert(0, texts)
    #         text = clip.tokenize(words).to(self.device)
    #         txt_emb_i = self.explainability.model.encode_text(text)
    #         txt_emb[task_n] = txt_emb_i.detach()

    #     return txt_emb

    def image_segment(self, pil_img, patch_size=32):
        patches = pil_img.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        patches = patches.reshape(-1, patch_size, patch_size)
        return patches

    def forward_lava(self, key, query, vl_mask, tp_mask):
        # print("sums" , torch.sum(visual_sentence), torch.sum(fused_x))
        # breakpoint()
        self.B = tp_mask.shape[0]

        key = key[~vl_mask[:, 0]]
        query = query[~vl_mask[:, 0]]
        vl_mask_ = vl_mask[~vl_mask[:, 0]]

        # visual_sentence = self.vis_layer_norm(visual_sentence)
        # visual_sentence = self.vis_dim_reducer(visual_sentence)

        # fused_x = self.lang_layer_norm(fused_x)
        # fused_x = self.lang_dim_reducer(fused_x)

        # out = self.vl_transformer_encoder(visual_sentence, fused_x, padding_mask = vl_mask_) # L x (B * l_contact_seq) X ft
        # out = torch.mean(out, axis = 1) # TODO: remove?
        out = torch.zeros((self.B * self.L, key.shape[-1])).to(self.device)
        vl_transformer_output = self.vl_transformer_encoder(
            key=key, query=query, padding_mask=vl_mask_
        )
        out[~vl_mask[:, 0]] = torch.mean(vl_transformer_output, axis=1)
        out = out.reshape((self.B, self.L, key.shape[-1]))
        # print("out", out[0])
        # breakpoint()

        tp_mask_tmp = tp_mask.unsqueeze(2).repeat((1, 1, out.shape[-1]))  # 1 padding
        out = torch.where(tp_mask_tmp, torch.zeros_like(out).to(self.device), out)

        # tp_output = self.tp_transformer(out, padding_mask = tp_mask, type = 'stack')
        tp_output = self.tp_transformer(out, padding_mask=tp_mask)
        # print("tp_output", tp_output[0])

        tp_output = tp_output.unsqueeze(2).unsqueeze(3).to(torch.float)

        contact = self.transformer_decoder(tp_output)  # (B * seq_l) X w x h
        # print("tp_output", contact[0])
        # breakpoint()

        w = int(contact.shape[-1])
        contact = contact.reshape((self.B, w, w))
        return contact


class contact_mlp_decoder(nn.Module):
    def __init__(self, dim_ft, last_activation=False, dim_out=150**2):
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
        nn.init.kaiming_normal_(
            self.l1.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l3.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l4.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        # nn.init.kaiming_normal_(self.l5.weight, a=0.0, nonlinearity='relu', mode='fan_in')

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        # x = self.relu(self.l4(x))
        x_cnt = self.l4(x)
        # x_cnt = self.sigmoid(self.l4(x))

        return x_cnt
