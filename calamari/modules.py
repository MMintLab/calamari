import os
import torch
import math

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import visualization
from Transformer_MM_Explainability.CLIP.clip.simple_tokenizer import (
    SimpleTokenizer as _Tokenizer,
)
import Transformer_MM_Explainability.CLIP.clip as clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from resnet import ResNet, ResidualBlock


class position_encoding_simple(nn.Module):
    def __init__(self, K: int, M: int, device) -> torch.Tensor:
        """
        An implementation of the simple positional encoding using uniform intervals
        for a sequence.

        args:
            K: int representing sequence length
            M: int representing embedding dimension for the sequence

        return:
            y: a Tensor of shape (1, K, M)
        """
        super().__init__()
        self.K = K
        self.M = M
        self.device = device

    def forward(self, x):
        n = torch.arange(0, self.K)
        y_i = (n / self.K).view(1, -1, 1)
        y = y_i.repeat(1, 1, self.M) * 0.001
        x = x + y.to(self.device)
        return x


class PositionalEncoding(nn.Module):
    ## Non trainable positional encoding
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x

        # return self.dropout(x)


class image_encoder(nn.Module):
    def __init__(self, device, dim_out):
        super(image_encoder, self).__init__()
        self.enc1 = ResNet(ResidualBlock, [1, 1, 1, 1], dim_out=dim_out).to(device)

    def forward(self, x):
        x = self.enc1(x)
        return x


class ClipExplainability(nn.Module):
    def __init__(self, device):
        super(ClipExplainability, self).__init__()
        self.device = device
        self._tokenizer = _Tokenizer()
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )

    def sentence2words(self, s):
        s = s.replace(".", "")
        words = s.split(" ")
        return words

    def get_heatmap(self, img, texts):
        ## text processing
        words = self.sentence2words(texts)
        words.insert(0, texts)
        text = clip.tokenize(words).to(self.device)

        ## image processing
        img = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)

        R_text, R_image, txt_emb = self.interpret(
            model=self.model, image=img, texts=text, device=self.device
        )
        batch_size = text.shape[0]
        heatmaps = []
        for i in range(batch_size):
            heatmap = self.show_image_relevance(
                R_image[i], img, orig_image=img
            )  # pilimage open
            heatmaps.append(heatmap)
        return txt_emb, heatmaps

    def show_image_relevance(self, image_relevance, image, orig_image):
        # create heatmap from mask on image
        # fig, axs = plt.subplots(1, 2)

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(
            image_relevance, size=224, mode="bilinear"
        )
        image_relevance = (
            image_relevance.reshape(224, 224).cuda().data
        )  # .cpu().numpy()

        image_relevance = (image_relevance - image_relevance.min()) / 0.04
        return image_relevance

        # image = image[0].permute(1, 2, 0).data #.cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        #
        # heatmap = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        #
        # return heatmap

    def interpret(
        self, image, texts, model, device, start_layer=-1, start_layer_text=-1
    ):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)

        txt_emb = model.encode_text(texts)

        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros(
            (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32
        )
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(
            dict(model.visual.transformer.resblocks.named_children()).values()
        )

        if start_layer == -1:
            # calculate index of last layer
            start_layer = len(image_attn_blocks) - 1

        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(
            num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
        ).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
                0
            ].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        text_attn_blocks = list(
            dict(model.transformer.resblocks.named_children()).values()
        )

        if start_layer_text == -1:
            # calculate index of last layer
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(
            num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype
        ).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
                0
            ].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text

        return text_relevance, image_relevance, txt_emb


class policy(nn.Module):
    def __init__(self, device, dim_ft):
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
        self.transformer_decoder = contact_mlp_decoder(self.dim_ft).to(self.device)

        # self.decoder_contact = contact_mlp_decoder(self.dim_ft, last_activation = True).to(self.device)
        # self.decoder_energy = contact_mlp_decoder(self.dim_ft, last_activation = False).to(self.device)

    def input_processing(self, img, texts):
        ## Encode image and texts with CLIP
        txt_emb, heatmaps = self.explainability.get_heatmap(img, texts)
        seg_idx = [0]
        hm_emb = []

        ## Get clip attention
        for hm_i in heatmaps:
            hm_emb_i = self._image_encoder(hm_i.unsqueeze(0).unsqueeze(1).float())
            hm_emb.append(hm_emb_i)
            seg_idx.append(1)
        hm_emb = torch.cat(hm_emb, dim=0)

        # inp  = torch.cat([txt_emb[0].unsqueeze(0), hm_emb], dim = 0) # sentence embedding + heatmap embedding
        inp = hm_emb.unsqueeze(0) * 0.01

        seg_idx = torch.tensor(seg_idx).to(self.device)
        return inp, seg_idx

    def forward(self, feat=None, seg_idx=None, img=None, texts=None):
        # ## Add positional + segment embeddings
        pos_enc_simple = position_encoding_simple(
            feat.size()[-2], self.dim_ft, self.device
        )
        feat = pos_enc_simple(feat)
        # feat =  self.pos_enc(feat)
        # # feat = feat + self.segment_emb(seg_idx) ## This one has issue

        out = self.transformer_encoder(feat)
        out = torch.amax(out, dim=1) * 0.1
        contact, energy = self.transformer_decoder(out)

        return contact, energy, out


class contact_mlp_decoder(nn.Module):
    def __init__(self, dim_ft, last_activation=False):
        super(contact_mlp_decoder, self).__init__()
        self.last_activation = last_activation

        # self.l6 = nn.Linear(dim_ft,  150**2)
        # self.l6_2 = nn.Linear(dim_ft,  150**2)

        # self.l2 = nn.Linear(1024, 1024)
        # self.l3 = nn.Linear(1024, 1024)
        # self.l4 = nn.Linear(1024, 1024)
        # self.l5 = nn.Linear(1024, 1024)
        # self.l6 = nn.Linear(1024, 150**2)
        # self.l6_2 = nn.Linear(1024, 150**2)

        self.l1 = nn.Linear(dim_ft, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1024)
        self.l4 = nn.Linear(1024, 1024)
        # self.l5 = nn.Linear(1024, 1024)
        self.l6 = nn.Linear(1024, 150**2)
        self.l6_2 = nn.Linear(1024, 150**2)

        nn.init.kaiming_normal_(
            self.l1.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        # nn.init.kaiming_normal_(self.l3.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # nn.init.kaiming_normal_(self.l4.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # nn.init.kaiming_normal_(self.l5.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_normal_(
            self.l6.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        nn.init.kaiming_normal_(
            self.l6_2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )

        self.relu = nn.ReLU()  # nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))
        # x = self.relu(self.l4(x))
        # x = self.relu(self.l5(x))
        x_cnt = self.sigmoid(self.l6(x)).reshape((1, 150, 150))
        x_eng = self.l6_2(x).reshape((1, 150, 150))

        # x_eng =  self.sigmoid(self.l6_2(x)).reshape((1,150,150))
        return x_cnt, x_eng
        # if self.last_activation:
        #     x = self.sigmoid(x*0.1)
        # return x.reshape((1,150,150))


class contact_decoder(nn.Module):
    def __init__(self, dim_ft):
        super(contact_decoder, self).__init__()
        # self.dconv1 = nn.ConvTranspose2d(in_channels= dim_ft, out_channels=32, kernel_size=4, stride=2) # 1x1 -> 4x4
        # self.dconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # 3x3 -> 10 x 10
        # self.dconv3 = nn.ConvTranspose2d(16, 8, 8, stride=2)  #  10 x 10 -> 26 x 26
        # self.dconv4 = nn.ConvTranspose2d(8, 4, 8, stride=2)  # 26 x 26 -> 58 x 58
        # self.dconv5 = nn.ConvTranspose2d(4, 2, 16, stride=1)  # 58 x 58 -> 130x130
        # self.dconv6 = nn.ConvTranspose2d(2, 1, 21, stride=1)  # 70x70 -> 150x150
