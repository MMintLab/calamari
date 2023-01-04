import time

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *
from modules import contact_decoder, contact_mlp_decoder

class ContactEnergy:
    def __init__(self, len):
        # relu = torch.nn.ReLU()
        self.epoch = 80
        self.gamma = 0.8
        self.N = 50
        self.W = 20
        self.device = 'cuda'
        self.dim_ft = 64 # 32
        self.seed = 42
        self.len = len

        # With square kernels and equal stride
        self.feat  = torch.nn.Embedding(self.len, self.dim_ft).to(self.device)
        print(self.feat.weight.size())
        self.feat.weight.data.normal_(mean=0.0, std=0.1)


        self.energy_decoder = contact_mlp_decoder(self.dim_ft, last_activation=False).to(self.device)
        self.contact_decoder = contact_mlp_decoder(self.dim_ft, last_activation=True).to(self.device)
        self.optim = torch.optim.Adam(
            [   {"params" : self.energy_decoder.parameters()},
                {"params" : self.contact_decoder.parameters()},
                {"params": self.feat.parameters(), "lr" : 0.003}
            ], lr=0.001)

    def save_model(self, model):
        save_path = f'logs/{model}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({"epoch": self.epoch,
                    "energy_decoder": self.contact_decoder.parameters(),
                    "contact_decoder": self.energy_decoder.parameters()}, save_path + '/policy.pth')


    def get_energy_field(self, folder_path):
        for i in range (self.epoch):
            tot_tot_loss = 0
            for l in range(self.len):
                folder_path = f'dataset/logs/t_{l:02d}'
                traj_gt_path = os.path.join(folder_path, 'contact')
                traj_gt_lst = folder2filelist(traj_gt_path)
                traj_gt_lst.sort()
                mask_ = get_traj_mask(traj_gt_lst)
                mask_t = torch.tensor(mask_).to(self.device)


                tot_loss = 0
                feat = self.feat(torch.tensor(l).to(self.device)).unsqueeze(0)  # .unsqueeze(2).unsqueeze(3)
                energy = self.energy_decoder(feat)
                contact = self.contact_decoder(feat).squeeze()

                # print(torch.amax(energy), torch.amin(energy))
                loss0 = torch.norm( mask_t - contact, p =2) / len(energy)
                loss_aux = torch.norm(energy, p=2) / len(energy) * 1 + torch.norm(feat) * 0.01
                tot_loss += loss0  + loss_aux

                # energy_f = torch.where(energy > 20, 20*torch.ones_like(energy).to(self.device), energy)
                # energy_f = torch.where(energy_f < -20, -20*torch.ones_like(energy_f).to(self.device), energy_f)
                energy_f = energy
                print(l,  torch.amax(energy_f), torch.amin(energy_f))

                for st in range(len(traj_gt_lst) - self.W + 1 ):
                    seq_gt = traj_gt_lst[st:st+self.W]
                    img_gt = fn2img(seq_gt)

                    gt_score = trajectory_score(energy_f, img_gt, range(0, self.W), self.gamma, device= self.device)
                    random_score = 0
                    cnt = 0
                    while cnt < self.N:

                        cnt_idx = random_order(self.W)
                        random_score_i = trajectory_score(energy_f, img_gt, idx = cnt_idx, gamma = self.gamma, device = self.device)
                        random_score += random_score_i
                        cnt += 1


                    loss1 = torch.log(1 + torch.exp( 1e-3 * ( random_score/ float(self.N) - gt_score ))) * 1e2
                    tot_loss += loss1

                self.optim.zero_grad()
                tot_loss.backward(retain_graph=True)
                self.optim.step()
                tot_tot_loss += copy.copy(tot_loss)

                if i % 10 == 0 or i == self.epoch -1:
                    energy_ = energy_f.detach() if self.device == 'cpu' else energy_f.detach().cpu()
                    # print( torch.amax(energy_), torch.amin(energy_))
                    save_energy(energy_.detach().cpu(), mask_, f'{folder_path}/result_{i}_6ly.png')
                    cv2.imwrite(f'{folder_path}/contact_{i}_6ly.png' , contact.detach().cpu().numpy() * 255.)
                    print("saved",f'{folder_path}/result_{i}_6ly.png' )

            tqdm.write("epoch: {}, loss: {}".format(i, tot_tot_loss))

L = 20
CE = ContactEnergy(L)
# for i in range(L):
CE.get_energy_field(f'dataset/logs/')
CE.save_model('after_transformer')