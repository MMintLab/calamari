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

from utils import *
from modules_gt import policy
from config import Config
from dataset import DatasetSeq_front_gt_feedback as Dataset
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
        train_dataset = Dataset(self.Config, mode = 'train')
        self.DataLoader = DataLoader(dataset = train_dataset,
                         batch_size=self.Config.B, shuffle=True)

        self.DataLoader_test = DataLoader(dataset=Dataset(self.Config, mode = 'test'),
                         batch_size=self.Config.B, shuffle=False)

        ## Define policy model
        dimout = train_dataset.cnt_w * train_dataset.cnt_h
        self.policy = policy(self.Config.device, self.Config.dim_ft, dim_out= dimout).cuda()


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

    def  overlay_cnt_rgb(self, rgb_path, cnt_pred):
        ## open rgb image with cv2
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:,:,:3]

        # uic = union_img(cnt_pred.squeeze()).numpy()
        uic = cnt_pred.squeeze().detach().cpu().numpy()
        iidx, jidx = np.where( np.sum(uic, axis = -1) != 0)
        rgb[iidx, jidx,:] = uic[iidx, jidx,:] * 255.
        return torch.tensor(rgb)

    def _evaluate_testdataset(self):
        contact_hist = []
        contact_histories_ovl = []
        cost_hist = []
        vel_hist = []

        energy_loss = 0
        contact_loss = 0
        for data in self.DataLoader_test:
            l = data["idx"]
            rgb = data['traj_rgb'][0]
            cost_gt = data['cost_map'][0]
            traj_cnt_lst = data['traj_cnt_lst']
            traj_len = data['traj_len']
            mask_t = data['mask_t'].squeeze(0).detach().cpu()

            ## Feed forward
            feat, seg_idx =  self.policy.input_processing(rgb, TXT)
            contact, cost, vel, out = self.policy(feat, seg_idx)

            ## save history
            cost_reg, cost_reg_ori = energy_regularization(cost.detach().cpu(), torch.tensor(mask_t), minmax = (0,1), return_original = True)
            vel_reg = energy_regularization(vel.detach().cpu(), torch.tensor(mask_t), minmax = (0,1), return_original = False)

            contact_hist.append(contact.detach().cpu())
            cost_hist.append(cost_reg.detach().cpu())
            vel_hist.append(vel_reg.detach().cpu())
            contact_histories_ovl.append(self.overlay_cnt_rgb(rgb, round_mask(contact.detach().cpu()).unsqueeze(3) * cost_reg_ori))

        self.data_summary = self.get_data_summary()


    def write_tensorboard_test(self, step, contact, energy, vel, contact_ovl, eng_loss_t, cnt_loss_t):
        contact = torch.cat(contact, dim = 0).unsqueeze(3)
        energy = torch.stack(energy, dim = 0)
        velocity = torch.stack(vel, dim = 0)

        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact_ovl_test", contact_ovl, max_outputs=len(contact_ovl), step=step)

            tf.summary.image("contact_test", contact, max_outputs=len(contact), step=step)
            tf.summary.image("energy_test", energy, max_outputs=len(energy), step=step)
            tf.summary.image("velocity_test", velocity, max_outputs=len(energy), step=step)

            tf.summary.scalar("loss1_test", eng_loss_t.detach().cpu() , step=step)
            tf.summary.scalar("loss0_test", cnt_loss_t.detach().cpu(), step=step)

    def write_tensorboard(self, step, contact, energy, vel, contact_ovl):
        contact = torch.cat(contact, dim = 0).unsqueeze(3)
        energy = torch.cat(energy, dim = 0)
        vel = torch.cat(vel, dim = 0)

        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("energy", energy, max_outputs=len(energy), step=step)
            tf.summary.image("velocity", vel, max_outputs=len(energy), step=step)

            tf.summary.image("contact_ovl", contact_ovl, max_outputs=len(contact_ovl), step=step)


            tf.summary.scalar("tot_loss", self.tot_loss['sum'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss0", self.tot_loss['loss0'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss1", self.tot_loss['loss1'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss_aux", self.tot_loss['loss_aux'].detach().cpu() / self.Config.len, step=step)

    def _initialize_loss(self, mode = 'p'): # 'p' = partial, 'a' = all
        if mode == 'a':
            self.tot_loss = {'sum': 0, "loss0_i":0, "loss1_i": 0, "loss2_i": 0, "loss_aux_i": 0, "loss0":0, "loss1": 0, "loss_aux": 0}
        elif mode == 'p':
            self.tot_loss['loss0_i'] = 0
            self.tot_loss['loss1_i'] = 0
            self.tot_loss['loss2_i'] = 0

            self.tot_loss['loss_aux_i'] = 0
        else:
            raise Exception("Error : mode not recognized")


    def training(self, folder_path):
        for i in range (self.Config.epoch):
            if i % 100 == 0 or i == self.Config.epoch - 1:
                CE.save_model()

            tot_tot_loss = 0
            self._initialize_loss(mode = 'a')

            contact_histories = [0] * self.DataLoader.__len__()
            energy_histories = [0] * self.DataLoader.__len__()
            contact_histories_ovl = [0] * self.DataLoader.__len__()
            vel_histories = [0] * self.DataLoader.__len__()


            for data in self.DataLoader:
                l = data["idx"]
                rgb = data['traj_rgb'][0]
                cost_gt = data['cost_map'][0].to(self.Config.device)
                vel_gt = data['vel_map'][0].to(self.Config.device)

                traj_cnt_lst = data['traj_cnt_lst']
                traj_len = data['traj_len']
                mask_t = data['mask_t'].to(self.Config.device).squeeze(0)

                feat, seg_idx =  self.policy.input_processing(rgb, TXT)
                contact, cost, vel_map, out = self.policy(feat, seg_idx)


                # save histories
                energy_reg, cost_reg_ori = energy_regularization(cost.detach().cpu(), mask_t.detach().cpu(), minmax = (0,1), return_original = True)
                vel_reg = energy_regularization(vel_map.detach().cpu(), mask_t.detach().cpu(), minmax = (0,1))

                contact_histories[l] = contact.detach().cpu()
                energy_histories[l] = energy_reg.unsqueeze(0)
                vel_histories[l] = vel_reg.unsqueeze(0)
                contact_histories_ovl[l] = self.overlay_cnt_rgb(rgb, round_mask(contact.detach().cpu()).unsqueeze(3) * cost_reg_ori)


                # loss
                contact = contact.squeeze()
                self.tot_loss['loss0_i'] = self.tot_loss['loss0_i'] +  torch.norm( mask_t - contact, p =2) / len(cost)
                self.tot_loss['loss1_i'] = self.tot_loss['loss1_i'] +  torch.norm( cost_gt - cost, p =2) / len(cost)
                self.tot_loss['loss2_i'] = self.tot_loss['loss2_i'] +  torch.norm( vel_gt - vel_map, p =2) / len(cost)

                self.tot_loss['loss_aux_i'] = self.tot_loss['loss_aux_i'] + torch.norm(cost, p=2) / (150**2)  # + torch.norm(feat)/ len(feat) * 0.01 + torch.norm(out) * 1

                if l % self.Config.B == self.Config.B - 1 or l == self.Config.len -1:
                    self.tot_loss['loss0'] = self.tot_loss['loss0']  + self.tot_loss['loss0_i'].detach().cpu()
                    self.tot_loss['loss1'] = self.tot_loss['loss1'] + self.tot_loss['loss1_i'].detach().cpu()
                    self.tot_loss['loss_aux'] = self.tot_loss['loss_aux'] +  self.tot_loss['loss_aux_i'].detach().cpu()
                    self.tot_loss['sum'] = self.tot_loss['loss0_i'] * 1e2 + self.tot_loss['loss_aux_i']  * 1e-4 + self.tot_loss['loss1_i'] * 1e2 + self.tot_loss['loss2_i'] * 1e2

                    self.optim.zero_grad()
                    self.tot_loss['sum'].backward()
                    self.optim.step()

                    torch.cuda.empty_cache()
                    tot_tot_loss += copy.copy(self.tot_loss['sum'].detach().cpu())
                    self._initialize_loss(mode = 'p')


            if i % 5 == 0 or i == self.Config.epoch -1:
                self.write_tensorboard(i, contact_histories, energy_histories, vel_histories, contact_histories_ovl)
            
            if i % 100 == 0 or i == self.Config.epoch -1:               
                contact_histories_t, energy_histories_t, vel_histories_t, contact_histories_ovl, eng_loss_t, cnt_loss_t  = self._evaluate_testdataset()
                self.write_tensorboard_test(i, contact_histories_t, energy_histories_t, vel_histories_t, contact_histories_ovl, eng_loss_t, cnt_loss_t)

            tqdm.write("epoch: {}, loss: {}".format(i, tot_tot_loss))


CE = ContactEnergy( log_path = 'transformer_w_gt_feedback', test_idx = args.test_idx)
CE.training(f'dataset/logs/')
