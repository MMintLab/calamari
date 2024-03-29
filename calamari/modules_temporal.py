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

from .Transformer_MM_Explainability.CLIP.clip.simple_tokenizer import (
    SimpleTokenizer as _Tokenizer,
)
from .Transformer_MM_Explainability.CLIP import clip
from language4contact.modules_shared import *
from language4contact.temporal_transformer import (
    TemporalTransformer,
    PrenormPixelLangEncoder,
)
import language4contact.utils as utils
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from .resnet import ResNet, ResidualBlock
from language4contact.config.config import Config


class policy(nn.Module):
    def __init__(self, dim_in, dim_out, image_size=255, Config: Config = None):
        super(policy, self).__init__()
        self.Config = Config
        self.dim_ft = Config.dim_ft
        self.device = Config.device
        self.B = self.Config.B  # Batch size.
        self.L = self.Config.contact_seq_l  # Sequence Length.

        ## Input processing for Transformer
        self.explainability = ClipExplainability(self.device)
        self._image_encoder = image_encoder(
            self.device, dim_in=1, dim_out=int(self.dim_ft)
        )
        # self._image_encoder = image_encoder(self.device, dim_in = 1 , dim_out = int(self.dim_ft/8))

        ## Transformer Encoder
        # self.pos_enc = PositionalEncoding(self.dim_ft, dropout=0.1, max_len=30).to(self.device)

        # # vision language transformer.
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_ft, dim_feedforward=128, nhead=1) #default: nhead = 8
        # self.vl_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.vl_transformer_encoder.to(self.device)

        self.vl_transformer_encoder = PrenormPixelLangEncoder(
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            mha_dropout_rate=0.0,
            dff=self.dim_ft,
            device=self.Config.device,
        )
        self.vl_transformer_encoder = self.vl_transformer_encoder.to(torch.float)

        self.transformer_decoder = contact_mlp_decoder(self.dim_ft, dim_out=dim_out).to(
            self.device
        )

        # temporal transformer.
        self.tp_transformer = TemporalTransformer(
            dim_in=self.dim_ft,
            d_model=self.dim_ft,
            sequence_length=self.L,
            device=self.device,
        )
        # self.model_grd = [self._image_encoder, self.vl_transformer_encoder,  self.transformer_decoder, self.tp_transformer]

        self.text_embs = self._get_text_emb(self.Config.txt_cmd)

    def _get_text_emb(self, texts):
        words = sentence2words(texts)
        words.insert(0, texts)
        text = clip.tokenize(words).to(self.device)
        txt_emb = self.explainability.model.encode_text(text)
        return txt_emb.detach()

    def image_segment(self, pil_img, patch_size=32):
        patches = pil_img.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        patches = patches.reshape(-1, patch_size, patch_size)
        return patches

    def read_heatmap_temporal(self, img_batch_pths, texts):
        ## text processing
        words = sentence2words(texts)

        if self.Config.heatmap_type == "chefer":
            cnt_dir = "heatmap/"
            words.insert(0, texts)

        elif self.Config.heatmap_type == "huy":
            cnt_dir = "heatmap_huy/"

        txt_emb = self.text_embs
        # text = clip.tokenize(words).to(self.device)

        ## image processing
        txt_emb_batch = []
        heatmaps_batch = []
        img_batch = []
        padding_masks = []
        # print(img_batch_pths)
        for img_pths in img_batch_pths:
            padding_mask = []
            for img_pth in img_pths:
                if len(img_pth) > 0:
                    heatmaps = []
                    for wd in words:
                        hm_pth = img_pth.replace("rgb/", cnt_dir).split(".")[0]
                        wd = wd.replace(".", "")
                        hm_pth = os.path.join(hm_pth, wd + ".png")
                        heatmap = Image.open(hm_pth).resize(
                            (self.Config.heatmap_size[0], self.Config.heatmap_size[1])
                        )
                        heatmap = torch.tensor(np.array(heatmap)).to(self.device)

                        # heatmap = self.show_image_relevance(R_image[i], img, orig_image=img) #pilimage open
                        heatmaps.append(heatmap)

                    txt_emb_batch.append(txt_emb)
                    heatmaps_batch.append(torch.stack(heatmaps))
                    # img_batch.append(img)
                    padding_mask.append(0)

                elif len(img_pth) == 0:  # type(None):
                    txt_emb = torch.zeros_like(txt_emb_batch[0]).to(self.device)
                    heatmaps = torch.zeros_like(heatmaps_batch[0]).to(self.device)
                    # img = torch.zeros_like(img_batch[0]).to(self.device)

                    txt_emb_batch.append(txt_emb)
                    heatmaps_batch.append(heatmaps)
                    # img_batch.append(img)
                    padding_mask.append(1)
            padding_masks.append(padding_mask)
        return (
            txt_emb_batch,
            torch.stack(heatmaps_batch),
            torch.tensor(padding_masks).bool(),
        )

    def input_processing(self, img, texts, mode="train"):
        """
        mode = 'train' : load heatmap
        mode = 'test' : generate heatmap every time step
        """
        # txt_emb, heatmaps, padding_mask = self.explainability.get_heatmap_temporal(img, texts)

        txt_emb, heatmaps, padding_mask = self.read_heatmap_temporal(img, texts)
        self.B = len(img)
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

        return inp, txt_emb, padding_mask

    def forward(self, feat, seg_idx=None, txt_token=None, padding_mask=None):
        feat = feat.permute(
            (1, 0, 2)
        )  # pytorch transformer takes L X (B * l_contact_seq) X ft
        pos_enc_simple = position_encoding_simple(
            K=feat.size()[0], M=self.dim_ft, device=self.device
        )
        feat = pos_enc_simple(feat)

        out = self.vl_transformer_encoder(feat)  # L x (B * l_contact_seq) X ft
        out = torch.mean(out, axis=0)
        out = out.reshape((self.B, self.L, -1))
        print("out shape", out.shape)
        # out = out.permute((1,0,2))

        tp_output = self.tp_transformer(out, padding_mask=padding_mask)
        contact = self.transformer_decoder(tp_output)  # (B * seq_l) X w x h
        w = int(np.sqrt(contact.shape[-1]))
        contact = contact.reshape((self.B, w, w))
        return contact

    def forward_lava(self, visual_sentence, fused_x, padding_mask):
        # visual_sentence = visual_sentence.permute((1,0,2)) # pytorch transformer takes L X (B * l_contact_seq) X ft
        # fused_x = fused_x.permute((1,0,2)) # pytorch transformer takes L X (B * l_contact_seq) X ft

        # pos_enc_simple = position_encoding_simple(K= feat.size()[0], M=self.dim_ft, device=self.device)
        # feat = pos_enc_simple(feat)

        out = self.vl_transformer_encoder(
            visual_sentence, fused_x
        )  # L x (B * l_contact_seq) X ft
        out = torch.mean(out, axis=1)  # TODO: remove?
        out = out.reshape((self.B, self.L, -1))
        # out = out.permute((1,0,2))

        tp_output = self.tp_transformer(out, padding_mask=padding_mask)
        contact = self.transformer_decoder(tp_output)  # (B * seq_l) X w x h

        w = int(np.sqrt(contact.shape[-1]))
        contact = contact.reshape((self.B, w, w))
        return contact

    def input_processing_from_heatmap(self, heatmap_folder: str, sentence: List[str]):
        ## Encode image and texts with CLIP
        text_token = clip.tokenize(sentence).to(self.device)
        text_emb = self.explainability.model.encode_text(text_token)
        texts = utils.sentence2words(sentence)

        # Record config.
        self.B = len(heatmap_folder)
        self.L = 1 + len(texts)

        heat_map_batch = []
        src_padding_mask_batch = []
        for hm_folder_i in heatmap_folder:
            heatmaps = []
            src_padding_masks = [0]
            for idx, txt in enumerate(texts):
                pilimage = torch.tensor(
                    np.array(
                        Image.open(os.path.join(hm_folder_i, txt + ".png")).resize(
                            (224, 224)
                        )
                    )
                )

                heatmaps.append(pilimage)

                # # All zero image = Padding
                if torch.sum(pilimage) == 0:
                    src_padding_masks.append(1)
                else:
                    src_padding_masks.append(0)
            heat_map_batch.append(torch.stack(heatmaps, dim=0))
            src_padding_mask_batch.append(src_padding_masks)

        heat_map_batch = torch.stack(heat_map_batch).to(self.device)
        heat_map_batch = utils.image_reg_255(heat_map_batch)

        seg_idx = [0]
        hm_emb = []

        ## Get clip attention
        img_enc_inp_2 = (
            torch.flatten(heat_map_batch, 0, 1).unsqueeze(1).float()
        )  # [320, 1, 256, 256]

        inp = self._image_encoder(img_enc_inp_2)
        inp = inp.reshape(
            (heat_map_batch.shape[0], heat_map_batch.shape[1], inp.shape[-1])
        )  # [batch size x seq x feat_dim]

        text_emb = text_emb.unsqueeze(0).repeat((heat_map_batch.shape[0], 1, 1))
        inp = torch.cat([text_emb, inp], dim=1)

        seg_idx += [1] * heat_map_batch.shape[1]
        seg_idx = torch.tensor(seg_idx).repeat(heat_map_batch.shape[0]).to(self.device)

        return (
            inp,
            seg_idx,
            text_token,
            torch.tensor(src_padding_mask_batch).bool().to(self.device),
        )


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

        x_cnt = self.sigmoid(self.l4(x))

        return x_cnt
