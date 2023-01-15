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
from language4contact.modules_gt import policy
from torch.utils.data import DataLoader


from config.config import Config
from language4contact.dataset import DatasetSeq_front_feedback as Dataset


parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0,1], help="used gpu")
parser.add_argument("--test_idx", type=tuple, default=(30, 37), help="index of test dataset")

args = parser.parse_args()

TXT  = "Use the sponge to clean up the dirt."
# os.environ["CUDA_VISIBLE_DEVICES"] = 3
torch.cuda.set_device(int(args.gpu_id))


class ContactEnergy():
    def __init__(self, log_path, test_idx = (30, 37)):
        self.Config = Config()
        torch.manual_seed(self.Config.seed)    

        self.test_idx = test_idx
        self.log_path = f'logs/{log_path}'
        self._initlize_writer(self.log_path)
        self._initialize_loss(mode = 'a')

        ## Data-loader
        self.train_dataset = Dataset(self.Config, mode = 'train', seq_l = 1)
        self.DataLoader = DataLoader(dataset = self.train_dataset,
                         batch_size=self.Config.B, shuffle=True)

        self.DataLoader_test = DataLoader(dataset=Dataset(self.Config, mode = 'test', seq_l = 1),
                         batch_size=1, shuffle=False)

        ## Define policy model
        dimout = self.train_dataset.cnt_w * self.train_dataset.cnt_h
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
        self._save_script_log( os.path.abspath( logdir))
    
    def _save_script_log(self, logdir):
        save_script('script/contact2energy_w_gt_feedback.py', logdir)
        save_script('language4contact/modules.py', logdir)
        save_script('language4contact/utils.py', logdir)
        save_script('language4contact/loss.py', logdir)
        save_script('config/config.py', logdir)

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

        contact_loss = 0
        for data in self.DataLoader_test:
            l = data["idx"]
            print(l)
            rgb = data['traj_rgb']


            # traj_cnt_lst = data['traj_cnt_lst']
            mask_t = data['mask_t'].squeeze(0)

            ## Feed forward
            feat, seg_idx =  self.policy.input_processing(rgb, TXT)
            contact, _, _, out = self.policy(feat, seg_idx)
            # contact_round = round_mask(contact.detach().cpu())
            # contact_round = torch.dstack([contact_round, contact_round, contact_round]).reshape((256, 256, 3))


            ## save history

            contact_hist.append(contact.detach().cpu())

            cost_reg_ori_ = round_mask(contact[0].detach().cpu()).unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_,cost_reg_ori_,cost_reg_ori_])
            contact_ovl = self.overlay_cnt_rgb(rgb[0], cost_reg_ori_)
            contact_histories_ovl.append(contact_ovl)

            ## Add loss
            contact_loss +=  torch.norm( mask_t.detach().cpu() - contact.detach().cpu(), p =2) / len(contact)
       
       
       ## summary of result
        contact_loss_ave = contact_loss / ((self.test_idx[1] - self.test_idx[0] + 1) / 4 )
        return contact_hist, contact_histories_ovl, contact_loss_ave

    def write_tensorboard_test(self, step, contact, contact_ovl, cnt_loss_t):
        contact = torch.cat(contact, dim = 0).unsqueeze(3)

        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact_ovl_test", contact_ovl, max_outputs=len(contact_ovl), step=step)

            tf.summary.image("contact_test", contact, max_outputs=len(contact), step=step)
            tf.summary.scalar("loss0_test", cnt_loss_t, step=step)

    def write_tensorboard(self, step, contact, contact_ovl):
        contact = torch.stack(contact, dim = 0).unsqueeze(3)

        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl", contact_ovl, max_outputs=len(contact_ovl), step=step)

            tf.summary.scalar("tot_loss", self.tot_loss['sum'].detach().cpu()/ self.Config.len, step=step)
            tf.summary.scalar("loss0", self.tot_loss['loss0'].detach().cpu()/ self.Config.len, step=step)
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

            l_len = 10
            contact_histories = [0] * l_len # self.train_dataset.__len__() 
            contact_histories_ovl = [0] * l_len #self.train_dataset.__len__() 

            l_i_hist = []
            for data in self.DataLoader:
                l = data["idx"]
                rgb = data['traj_rgb']

                # traj_cnt_lst = data['traj_cnt_lst']
                mask_t = data['mask_t'].to(self.Config.device).squeeze(0)

                feat, seg_idx =  self.policy.input_processing(rgb, TXT)
                contact, cost, vel_map, out = self.policy(feat, seg_idx)


                # save histories
                energy_reg, cost_reg_ori = energy_regularization(cost.detach().cpu(), mask_t.detach().cpu(), minmax = (0,2.5), return_original = True)
                vel_reg = energy_regularization(vel_map.detach().cpu(), mask_t.detach().cpu(), minmax = (0,1))


                for l_ir, l_i in enumerate(l):
                    if l_i < l_len: 
                        l_i_hist.append(l_i)
                        contact_histories[l_i] = contact[l_ir].detach().cpu()
                        
                        cost_reg_ori_ = round_mask(contact[l_ir].detach().cpu()).unsqueeze(2)
                        cost_reg_ori_ = torch.dstack([cost_reg_ori_,cost_reg_ori_,cost_reg_ori_])
                        contact_histories_ovl[l_i] = self.overlay_cnt_rgb(rgb[l_ir], cost_reg_ori_)


                # loss
                contact = contact.squeeze()
                self.tot_loss['loss0_i'] = self.tot_loss['loss0_i'] +  torch.norm( mask_t - contact, p =2) / len(cost)                # print(self.tot_loss['loss0_i'],self.tot_loss['loss1_i'], self.tot_loss['loss2_i'])
                self.tot_loss['loss_aux_i'] = torch.tensor(0) # self.tot_loss['loss_aux_i'] + torch.norm(cost, p=2) / (150**2)  # + torch.norm(feat)/ len(feat) * 0.01 + torch.norm(out) * 1
                self.tot_loss['sum'] = self.tot_loss['loss0_i'] * 1e2 + self.tot_loss['loss_aux_i']  * 1e-4 + self.tot_loss['loss1_i'] * 1e2 + self.tot_loss['loss2_i'] * 1e2

                self.optim.zero_grad()
                self.tot_loss['sum'].backward()
                self.optim.step()

                # Save log
                self.tot_loss['loss0'] = self.tot_loss['loss0']  + self.tot_loss['loss0_i'].detach().cpu()
                self.tot_loss['loss_aux'] = self.tot_loss['loss_aux'] +  self.tot_loss['loss_aux_i'].detach().cpu()



                torch.cuda.empty_cache()
                tot_tot_loss += copy.copy(self.tot_loss['sum'].detach().cpu())
                self._initialize_loss(mode = 'p')

            if i % 5 == 0:
                self.write_tensorboard(i, contact_histories, contact_histories_ovl)
            
            if i % 10 == 0:               
                contact_histories_t, contact_histories_ovl, cnt_loss_t  = self._evaluate_testdataset()
                self.write_tensorboard_test(i, contact_histories_t, contact_histories_ovl, cnt_loss_t)

            tqdm.write("epoch: {}, loss: {}".format(i, tot_tot_loss))


CE = ContactEnergy( log_path = 'transformer_single_patch_feedback', test_idx = args.test_idx)
CE.training(f'dataset/logs/')
