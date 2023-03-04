import os
import torch
import math

from torch import nn
import numpy as np
from PIL import Image
import cv2

from language4contact.modules_shared import *
from .resnet import ResNet, ResidualBlock
from .Transformer_MM_Explainability.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .Transformer_MM_Explainability.CLIP import clip 


def sentence2words(s):
    if type(s) == list:
        words = []
        for s_i in s:
            s_i = s_i.replace('.', '')
            words.append(s_i.split(' '))

        
    elif type(s) == str:
        s = s.replace('.', '')
        words = s.split(' ')
    return words


class ClipExplainability(nn.Module):
    def __init__(self, device):
        super(ClipExplainability,self).__init__()
        self.device = device
        self._tokenizer = _Tokenizer()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

    def get_heatmap_temporal(self, img_batch_pths, texts):
        ## text processing
        words = sentence2words(texts)
        words.insert(0, texts)
        text = clip.tokenize(words).to(self.device)

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
                    img = Image.open(img_pth)
                    img = self.preprocess(img).to(self.device)
                    R_text, R_image, txt_emb = self.interpret(model=self.model, image=img, texts=text, device=self.device)
                    batch_size = text.shape[0]

                    heatmaps = []
                    for i in range(batch_size):
                        heatmap = self.show_image_relevance(R_image[i], img, orig_image=img) #pilimage open
                        heatmaps.append(heatmap)

                    txt_emb_batch.append(txt_emb)
                    heatmaps_batch.append( torch.stack(heatmaps))
                    img_batch.append(img)
                    padding_mask.append(0)

                elif len(img_pth) == 0: #type(None):
                    txt_emb = torch.zeros_like(txt_emb_batch[0]).to(self.device)
                    heatmaps = torch.zeros_like(heatmaps_batch[0]).to(self.device)
                    img = torch.zeros_like(img_batch[0]).to(self.device)

                    txt_emb_batch.append(txt_emb)
                    heatmaps_batch.append(heatmaps)
                    img_batch.append(img)
                    padding_mask.append(1)
            padding_masks.append(padding_mask)
        return txt_emb, torch.stack(heatmaps_batch), torch.tensor(padding_masks).bool()


    def get_heatmap(self, img_pths, texts):
        ## text processing
        words = self.sentence2words(texts)
        words.insert(0, texts)
        text = clip.tokenize(words).to(self.device)

        ## image processing
        txt_emb_batch = []
        heatmaps_batch = []
        img_batch = []
        print(img_pths)
        for img_pth in img_pths:
            if type(img_pth) == str:
                img = Image.open(img_pth)
            else:
                img = Image.fromarray(img_pth)
            img = self.preprocess(img).to(self.device)
            R_text, R_image, txt_emb = self.interpret(model=self.model, image=img, texts=text, device=self.device)
            batch_size = text.shape[0]

            heatmaps = []
            for i in range(batch_size):
                heatmap = self.show_image_relevance(R_image[i], img, orig_image=img) #pilimage open
                heatmaps.append(heatmap)

            txt_emb_batch.append(txt_emb)
            heatmaps_batch.append( torch.stack(heatmaps))
            img_batch.append(img)

        return txt_emb, torch.stack(heatmaps_batch)


    def show_image_relevance(self, image_relevance, image, orig_image):
        # create heatmap from mask on image
        # fig, axs = plt.subplots(1, 2)

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data #.cpu().numpy()

        image_relevance = (image_relevance - image_relevance.min()) / 0.04
        return image_relevance

        # image = image[0].permute(1, 2, 0).data #.cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        # 
        # heatmap = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # 
        # return heatmap


    def interpret(self, image, texts, model, device, start_layer= -1, start_layer_text = -1):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)

        txt_emb = model.encode_text(texts)

        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if start_layer == -1:
          # calculate index of last layer
          start_layer = len(image_attn_blocks) - 1

        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
              continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]


        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if start_layer_text == -1:
          # calculate index of last layer
          start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
              continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text

        return text_relevance, image_relevance, txt_emb

class image_encoder_mlp(nn.Module):
    def __init__(self, device, dim_in=16*16, dim_out = 10):
        super(image_encoder_mlp, self).__init__()
        self.l1 = nn.Linear(dim_in, 256)
        # self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, dim_out)

        nn.init.kaiming_normal_(
            self.l1.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )
        # nn.init.kaiming_normal_(
        #     self.l2.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        # )
        nn.init.kaiming_normal_(
            self.l3.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        # x = self.relu(self.l2(x))
        x = self.l3(x)
        return x


class image_encoder(nn.Module):
    def __init__(self, device, dim_in=224, dim_out = 10):
        super(image_encoder, self).__init__()
        self.enc1 = ResNet(ResidualBlock, [1, 1, 1, 1 ], inplanes = dim_in, dim_out=dim_out).to(device)
    def forward(self, x):
        x = self.enc1(x)
        return x


class position_encoding_simple(nn.Module):
    def __init__(self, K: int, M: int, device) -> torch.Tensor:
        """
        An implementation of the simple positional encoding using uniform intervals
        for a sequence.

        args:
            K: int representing sequence length
            M: int representing embedding dimension for the sequence

        return:
            y: a Tensor of shape (K, B,M)
        """
        super().__init__()
        self.K = K
        self.M = M
        self.device = device

    def forward(self,x):

        n = torch.arange(0, self.K)
        y_i = (n/self.K).view(-1,1,1)
        y = y_i.repeat(1,x.shape[1],self.M) * 0.01
        x = x + y.to(self.device)
        return x

class PositionalEncoding(nn.Module):
    ## Non trainable positional encoding
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
