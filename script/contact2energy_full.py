import time
from argparse import ArgumentParser
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

from language4contact.utils import *
from modules import contact_decoder, contact_mlp_decoder, policy
from config.config import Config
from dataset import Dataset
from torch.utils.data import DataLoader
import loss


parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0,1], help="used gpu")
parser.add_argument("--test_idx", type=tuple, default=(30, 37), help="index of test dataset")

args = parser.parse_args()

TXT  = "Use the sponge to clean up the dirt."
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# torch.cuda.set_device("cuda:"+args.gpu_id)


class ContactEnergy():
    def __init__(self, log_path, test_idx = (30, 37)):
        self.Config = Config()
        torch.manual_seed(self.Config.seed)    

        self.test_idx = test_idx
        self.log_path = f'logs/{log_path}'
        self._initlize_writer(self.log_path)
        self._initialize_loss(mode = 'a')

        ## Data-loader
        self.DataLoader = DataLoader(dataset=Dataset(self.Config),
                         batch_size=self.Config.B, shuffle=True)

        ## Define policy model
        self.policy = policy(self.Config.device, self.Config.dim_ft).cuda()
        if  len(args.gpu_id) > 1:
            self.policy.transformer_encoder = nn.DataParallel(self.policy.transformer_encoder)
            self.policy._image_encoder = nn.DataParallel(self.policy._image_encoder)
            self.policy.segment_emb = nn.DataParallel(self.policy.segment_emb)
            self.policy.transformer_decoder = nn.DataParallel(self.policy.transformer_decoder)

        ## Set optimizer
        self.test = False if test_idx is None else True

        self.optim = torch.optim.Adam(
            [   {"params" : self.policy.transformer_encoder.parameters()},
                {"params" : self.policy._image_encoder.parameters()},
                {"params" : self.policy.segment_emb.parameters()},
                {"params": self.policy.transformer_decoder.parameters()}, #, "lr":0.005
                # {"params" : self.feat}
            ], lr=0.0001)


    def save_model(self):
        path = self.logdir + "/policy.pth"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save({
                    "transformer_encoder" : self.policy.transformer_encoder.state_dict(),
                    "image_encoder" : self.policy._image_encoder.state_dict(),
                    "transformer_decoder" : self.policy.transformer_decoder.state_dict(),
                    "segment_emb" : self.policy.segment_emb.state_dict(),
                    "optim" : self.optim.state_dict()
                    }, path)


    def _initlize_writer(self, log_dir):
        # Sets up a timestamped log directory.
        logdir = os.path.join(log_dir , datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.logdir = logdir

        # Clear out any prior log data.
        if os.path.exists(logdir):
            val = input( f"Delete the existing path {logdir} and all its contents? (y/n)")
            if val == 'y':
                os.remove(logdir)
            else:
                raise Exception("Error : the path exists")
               
        # Creates a file writer for the log directory.
        self.file_writer = tf.summary.create_file_writer(logdir)
        self._save_script_log(logdir)
    
    def _save_script_log(self, logdir):
        save_script('contact2energy_full.py', logdir)
        save_script('modules.py', logdir)
        save_script('utils.py', logdir)
        save_script('loss.py', logdir)
        save_script('config.py', logdir)


    def _evaluate_testdataset(self):
        contact_hist = []
        energy_hist = []
        energy_loss = 0
        contact_loss = 0
        for l in range(self.test_idx[0],self.test_idx[1]):
            folder_path = f'dataset/logs/t_{l:02d}'

            traj_rgb_lst = folder2filelist(os.path.join(folder_path, 'rgb'), sort = True)
            traj_cnt_lst = folder2filelist( os.path.join(folder_path, 'contact'), sort = True)

            feat, seg_idx =  self.policy.input_processing(traj_rgb_lst[0], TXT)

            contact, energy, out = self.policy(feat, seg_idx)
            mask_t = get_traj_mask(traj_cnt_lst)

            energy_reg = energy_regularization(energy.detach().cpu(), torch.tensor(mask_t))
            contact_hist.append(contact.detach().cpu())
            energy_hist.append(energy_reg.unsqueeze(0).detach().cpu())


            contact_loss += torch.norm( torch.tensor(mask_t) - contact.detach().cpu(), p =2) / len(energy) * 1e2


            for st in range(len(traj_cnt_lst) - self.Config.W + 1 ):
                seq_gt = traj_cnt_lst[st:st+self.Config.W]
                cnt_gt = fn2img(seq_gt, d = 1)

                gt_score = loss.trajectory_score(energy, cnt_gt, range(0, self.Config.W), self.Config).detach().cpu()
                random_score = loss.fast_score_negative(energy, cnt_gt, self.Config).detach().cpu()

                energy_loss_i = torch.log(1 + torch.exp( 1e-3 * ( random_score - gt_score ))) * 1e2
                energy_loss += energy_loss_i

       
        energy_loss_ave = energy_loss / ((self.test_idx[1] - self.test_idx[0] + 1) / 4 )
        contact_loss_ave = contact_loss / ((self.test_idx[1] - self.test_idx[0] + 1) / 4 )
        return contact_hist, energy_hist, energy_loss_ave, contact_loss_ave

    def write_tensorboard_test(self, step, contact, energy, eng_loss_t, cnt_loss_t):
        contact = torch.cat(contact, dim = 0).unsqueeze(3)
        energy = torch.cat(energy, dim = 0)
        with self.file_writer.as_default():
            tf.summary.image("contact_test", contact, max_outputs=len(contact), step=step)
            tf.summary.image("energy_test", energy, max_outputs=len(energy), step=step)
            tf.summary.scalar("loss1_test", eng_loss_t.detach().cpu() , step=step)
            tf.summary.scalar("loss0_test", cnt_loss_t.detach().cpu(), step=step)



    def write_tensorboard(self, step, contact, energy):
        contact = torch.cat(contact, dim = 0).unsqueeze(3)
        energy = torch.cat(energy, dim = 0)
        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("energy", energy, max_outputs=len(energy), step=step)
            tf.summary.scalar("tot_loss", self.tot_loss['sum'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss0", self.tot_loss['loss0'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss1", self.tot_loss['loss1'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss_aux", self.tot_loss['loss_aux'].detach().cpu() / self.Config.len, step=step)

    def _initialize_loss(self, mode = 'p'): # 'p' = partial, 'a' = all
        if mode == 'a':
            self.tot_loss = {'sum': 0, "loss0_i":0, "loss1_i": 0, "loss_aux_i": 0, "loss0":0, "loss1": 0, "loss_aux": 0}
        elif mode == 'p':
            self.tot_loss['loss0_i'] = 0
            self.tot_loss['loss1_i'] = 0
            self.tot_loss['loss_aux_i'] = 0
        else:
            raise Exception("Error : mode not recognized")


    def get_energy_field(self, folder_path):
        for i in range (self.Config.epoch):
            if i % 100 == 0 or i == self.Config.epoch - 1:
                CE.save_model()

            tot_tot_loss = 0
            self._initialize_loss(mode = 'a')

            contact_histories = [0] * self.Config.len
            energy_histories = [0] * self.Config.len

            for data in self.DataLoader:
                l = data["idx"]
                rgb = data['traj_rgb'][0]
                traj_cnt_lst = data['traj_cnt_lst']
                traj_len = data['traj_len']
                mask_t = data['mask_t'].to(self.Config.device).squeeze(0)

                feat, seg_idx =  self.policy.input_processing(rgb, TXT)
                contact, energy, out = self.policy(feat, seg_idx)


                # save histories
                energy_reg = energy_regularization(energy.detach().cpu(), mask_t.detach().cpu())
                energy_histories[l] = energy_reg.unsqueeze(0)
                contact_histories[l] = contact.detach().cpu()

                # loss
                contact = contact.squeeze()
                self.tot_loss['loss0_i'] = self.tot_loss['loss0_i'] +  torch.norm( mask_t - contact, p =2) / len(energy)
                self.tot_loss['loss_aux_i'] = self.tot_loss['loss_aux_i'] + torch.norm(energy, p=2) / (150**2)  # + torch.norm(feat)/ len(feat) * 0.01 + torch.norm(out) * 1

                for st in range( (traj_len - self.Config.W + 1)):

                    fnl = np.amin([st + self.Config.W, traj_len])
                    cnt_gt = traj_cnt_lst[:, st:fnl]

                    # cnt_gt = fn2img(seq_gt, d = 1)

                    gt_score = loss.fast_trajectory_score(energy, cnt_gt, range(0, cnt_gt.shape[1]), self.Config)
                    random_score = loss.fast_score_negative(energy, cnt_gt, self.Config)

                    energy_loss_i = gt_score / (gt_score + random_score)  
                    # energy_loss_i = random_score / (gt_score + random_score)           
                    energy_loss_i = torch.log( 1 + energy_loss_i )

                    self.tot_loss['loss1_i'] = self.tot_loss['loss1_i'] + energy_loss_i

                    # self.tot_loss['loss1_i'] = self.tot_loss['loss1_i'] + torch.log(1 + torch.exp( 1e-3 * ( random_score/ float(self.Config.N) - gt_score ))) 
                
                if l % self.Config.B == self.Config.B - 1 or l == self.Config.len -1:
                    self.tot_loss['loss0'] = self.tot_loss['loss0']  + self.tot_loss['loss0_i'].detach().cpu()
                    self.tot_loss['loss1'] = self.tot_loss['loss1'] + self.tot_loss['loss1_i'].detach().cpu()
                    self.tot_loss['loss_aux'] = self.tot_loss['loss_aux'] +  self.tot_loss['loss_aux_i'].detach().cpu()
                    self.tot_loss['sum'] = self.tot_loss['loss0_i'] * 1e2 + self.tot_loss['loss_aux_i']  * 1e-4 + self.tot_loss['loss1_i'] * 1e3

                    self.optim.zero_grad()
                    self.tot_loss['sum'].backward()
                    self.optim.step()

                    torch.cuda.empty_cache()
                    tot_tot_loss += copy.copy(self.tot_loss['sum'].detach().cpu())
                    self._initialize_loss(mode = 'p')



            if i % 5 == 0 or i == self.Config.epoch -1:
                self.write_tensorboard(i, contact_histories, energy_histories)
            
            # if i % 100 == 0 or i == self.Config.epoch -1:               
            #     contact_histories_t, energy_histories_t, eng_loss_t, cnt_loss_t  = self._evaluate_testdataset()
            #     self.write_tensorboard_test(i, contact_histories_t, energy_histories_t, eng_loss_t, cnt_loss_t)

            tqdm.write("epoch: {}, loss: {}".format(i, tot_tot_loss))


CE = ContactEnergy( log_path = 'transformer', test_idx = args.test_idx)
CE.get_energy_field(f'dataset/logs/')
