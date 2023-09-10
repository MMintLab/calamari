from argparse import ArgumentParser
from datetime import datetime
import time
from tqdm import tqdm
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

from language4contact.utils import *
from language4contact.modules_temporal_multi_conv import  policy
from language4contact.config.config_multi_conv import Config
from torch.utils.data import DataLoader
from language4contact.modules_shared import *

from language4contact.dataset_temporal_multi_fast_random import DatasetTemporal as Dataset, augmentation
import tensorflow as tf

import torchvision.transforms.functional as F
import torchvision.transforms as T

parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0,1], help="used gpu")
parser.add_argument("--test_idx", type=tuple, default=(30, 37), help="index of test dataset")
parser.add_argument("--from_log", type=str, default='', help= "log to the previous path")


args = parser.parse_args()
# TXT  = "Use the sponge to clean up the dirt."
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

# device = torch.device("cuda:0,2" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

torch.cuda.set_device(f"cuda:{args.gpu_id}")
# torch.cuda.set_device(f"cuda:2")

# torch.cuda.set_device(args.gpu_id)
from torch.utils.data.distributed import DistributedSampler



class ContactEnergy():
    def __init__(self, log_path, test_idx = (30, 37)):
        self.Config = Config()
        torch.manual_seed(self.Config.seed)    



        self.activation = {}
        ## Image Encoder
        # self.image_encoder = image_encoder(self.Config.device, dim_in = 1 , dim_out = int(self.Config.dim_emb))
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.image_encoder.to(self.Config.device).eval()
        self.image_encoder.avgpool.register_forward_hook(self.get_activation('avgpool'))




        ## Data-loader
        self._image_encoder = {"model": self.image_encoder, "activation": self.activation}
        self.train_dataset = Dataset(self.Config, mode = "train", image_encoder = self._image_encoder)
        self.test_dataset = Dataset(self.Config, mode = "test", image_encoder = self._image_encoder)

        world_size = 2
        train_sampler = DistributedSampler(self.train_dataset,num_replicas=world_size)
        val_sampler = DistributedSampler(self.test_dataset,num_replicas=world_size)
        batch_size = int(128 / float(world_size))

        self.train_dataLoader = DataLoader(dataset= train_sampler,
                         batch_size=self.Config.B, shuffle=True, drop_last=False)

        self.test_dataLoader = DataLoader(dataset=val_sampler,
                         batch_size=self.Config.B, shuffle=False, drop_last=False)




        # print("Train dataset size", self.train_dataset.__len__(), "Test dataset size", self.test_dataset.__len__())


        ## Define policy model
        dimout = self.train_dataset.cnt_w * self.train_dataset.cnt_h

        self.policy = policy(dim_in = self.train_dataset.cnt_w, dim_out= dimout, image_size = self.train_dataset.cnt_w, Config=self.Config)
        self.policy = nn.DataParallel(self.policy.to(self.Config.device),device_ids=[1,2])

        ## Set optimizer
        self.test = False if test_idx is None else True
        self.optim = torch.optim.Adam(
                [   {"params" : self.policy.parameters()}
                ], lr=1e-3)


        self.start_epoch = 0
        self.log_path = f'logs/{log_path}'

        if len(args.from_log) > 0 :
            pretrained = torch.load(os.path.join(args.from_log, "policy.pth"), map_location=torch.device('cpu'))
            pretrained_optim = torch.load(os.path.join(args.from_log, "optim.pth"), map_location=torch.device('cpu'))

            self.policy.module.load_state_dict(pretrained["param"])
            self.start_epoch = pretrained["epoch"]
            self.log_path  = pretrained["path"]
            self.optim.load_state_dict(pretrained_optim["optim"])

        self._initlize_writer(self.log_path)
        self._initialize_loss(mode = 'a')

        self.randomaffine = T.RandomAffine(degrees = 10, 
                                           translate = (0.15, 0.15), 
                                           scale = (0.9, 1.1), 
                                           shear = (5, 5))


    def cnt_aug(self, traj_cnt_img, aug_param):
        if aug_param[0] != 0.:
            traj_cnt_img = F.affine(traj_cnt_img.unsqueeze(0), 
                                    *aug_param, 
                                    interpolation=self.randomaffine.interpolation, 
                                    fill=self.randomaffine.fill).squeeze()
        return traj_cnt_img

    def hm_to_visual_sentense(self, heatmaps, vl_mask, aug_param):
        with torch.no_grad():
            heatmap_shape_ori = heatmaps.shape

            if aug_param[0] != 0.:
                heatmaps = F.affine(heatmaps.flatten(0,1).unsqueeze(0), 
                                *aug_param, 
                                interpolation=self.randomaffine.interpolation,
                                    fill=self.randomaffine.fill)
                heatmaps = heatmaps.squeeze().reshape(heatmap_shape_ori)

            heatmaps_ = heatmaps[ ~vl_mask[:,0]] # zero image to padding


            img_enc_inp = torch.flatten(heatmaps_, 0, 1).unsqueeze(1).float()
            img_enc_inp = torch.cat([img_enc_inp,
                                    img_enc_inp,
                                    img_enc_inp] , dim = 1)

            out_tot = []
            N = 100
            for i in range( img_enc_inp.shape[0] // N):
                self._image_encoder["model"](img_enc_inp[i * N : (i+1)*N])
                out = self._image_encoder["activation"]['avgpool'].squeeze()
                out_tot.append(out.detach())

            if img_enc_inp.shape[0]  % N != 0:
                self._image_encoder["model"](img_enc_inp[(i+1)*N:])
                out = self._image_encoder["activation"]['avgpool'].squeeze()
                out_tot.append(out.detach())
            out_tot = torch.cat(out_tot, dim=0)

            visual_sentence = torch.zeros((heatmaps.shape[0], heatmaps.shape[1], out_tot.shape[-1])).to(self.Config.device)
            visual_sentence[ ~vl_mask[:,0]] = out_tot.reshape((heatmaps_.shape[0], heatmaps.shape[1], out_tot.shape[-1]))
                
        return visual_sentence



    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def get_aug_idx(self, idx_raw):
        if idx_raw % len(self.train_dataset.idx_lst) !=0:
            aug_param = self.randomaffine.get_params(self.randomaffine.degrees, self.randomaffine.translate, 
                                                        self.randomaffine.scale, self.randomaffine.shear, (256, 256))
            
        else:
            aug_param =  (0., (0., 0.), 1., (0.0, 0.0))
    
        return aug_param

    def feedforward(self, dataloader, write = False, N = 200, mode = 'test'):
        contact_histories = [0 for _ in range(N)] #self.train_dataLoader.__len__()
        contact_histories_ovl = [0 for _ in range(N)]  #self.train_dataLoader.__len__()

        tot_loss = 0
        # l_i_hist = []
        t = time.time()
        for iter_idx, data in enumerate(dataloader):
            l = data["idx"]
            rgb = list(zip(*data['traj_rgb_paths']))
            traj_cnt_lst = list(zip(*data['traj_cnt_paths'])) #([B, input_length, img_w, img_h])\
            traj_cnt_img = torch.cat(data['traj_cnt_img'], dim = 0)
            txt = list(data['txt'])
            tasks = list(data["task"])
            # aug_idx = data["aug_idx"] # B,
            # print("1:", time.time()-t)
            # t = time.time()
            # print(flip)
            # breakpoint()


            # inp, txt_emb, vl_mask, tp_mask
            heatmaps = data["heatmaps"].flatten(0,1)
            # visual_sentence = data["visual_sentence"].flatten(0,1)
            fused_x = data["fused_x"].flatten(0,1)
            vl_mask = data["vl_mask"].flatten(0,1)
            tp_mask = data["tp_mask"]





            aug_param = self.get_aug_idx(iter_idx) # data["aug_idx"]

            # visual_sentence, fused_x, vl_mask, tp_mask =  self.policy.module.input_processing(rgb, txt, tasks, flip = flip)
            # fused_x = torch.flatten(fused_x, 0, 1)
            # print(visual_sentence.shape, fused_x.shape, vl_mask.shape, tp_mask.shape )

            visual_sentence = self.hm_to_visual_sentense(heatmaps, vl_mask, aug_param)
            contact_seq = self.policy.module.forward_lava(visual_sentence, fused_x, vl_mask = vl_mask, tp_mask = tp_mask)
            # print("2:",time.time()-t)
            t = time.time()
            # contact_seq = self.policy(feat, seg_idx, padding_mask = padding_mask.to(self.Config.device))

            # loss
            traj_cnt_img =  self.cnt_aug(traj_cnt_img.to(self.Config.device), aug_param)
            # print(traj_cnt_img.shape, contact_seq.shape)
            loss0_i = torch.norm( traj_cnt_img - contact_seq, p =2) / ( 150 **2 * self.train_dataset.contact_seq_l )
            # print(loss0_i,  torch.norm(traj_cnt_img.to(self.Config.device)), torch.norm(contact_seq.to(self.Config.device)))
            loss0_i = 1e5 * loss0_i
            
            if mode == 'train':
                self.optim.zero_grad()
                loss0_i.backward()
                self.optim.step()

            # print("3:", time.time()-t)
            # t = time.time()


            if write:
                rgb = zip(*data['traj_rgb_paths'])
                for l_ir, rgb_i in enumerate(rgb):
                    rgb_i = list(rgb_i)
                    rgb_i_ = rgb_i.pop()
                    while len(rgb_i_) == 0:
                        rgb_i_ = rgb_i.pop()

                    l_i = l[l_ir] # Batch index -> real idx
                    if l_i < N: 
                        # print(l_i, rgb_i_)
                        # contact_seq_round = round_mask(traj_cnt_img[l_ir].detach().cpu())
                        # print( contact_seq_round.shape)
                        contact_seq_round = round_mask(contact_seq[l_ir].detach().cpu())
                        contact_histories[l_i] = contact_seq_round
                        contact_histories_ovl[l_i] = self.overlay_cnt_rgb(rgb_i_, contact_seq_round, aug_param)
                        # l_i_hist.append(l_i)
                    # l_ir += 1
            # print("4:", time.time()-t)
            # t = time.time()


            # self.tot_loss['loss0'] = self.tot_loss['loss0']  + loss0_i.detach().cpu()
            tot_loss += loss0_i.detach().cpu()
            # torch.cuda.empty_cache()
            # self._initialize_loss(mode = 'p')

        # return contact patches, patches overlaid with RGB, normalized total loss
        return contact_histories, contact_histories_ovl, tot_loss


    def get_energy_field(self):
        for i in tqdm(range (self.start_epoch, self.Config.epoch)):
            t = time.time()
            if i % 15 == 0 or i == self.Config.epoch - 1:
                self.save_model(i)
            # print("model saved:", time.time()-t)
            t = time.time()

            self.policy.module.train(True)
            if i % 15 == 0 or i == self.Config.epoch -1: 
                contact_histories, contact_histories_ovl, tot_loss = self.feedforward(self.train_dataLoader, 
                                                                                      write = True,
                                                                                        N = np.amin([60, self.train_dataset.__len__()]),
                                                                                        mode = 'train')
                self.write_tensorboard(i, contact_histories, contact_histories_ovl, tot_loss)
            else:
                _, _, tot_loss = self.feedforward(self.train_dataLoader, write = False, 
                                                  N = np.amin([60, self.train_dataset.__len__()]), mode = 'train')
            # print("training loop:", time.time()-t)
            tqdm.write("epoch: {}, loss: {}".format(i, tot_loss))

            if i % 15 == 0 or i == self.Config.epoch -1: 
                self.policy.module.train(False)
                contact_histories, contact_histories_ovl, tot_loss = self.feedforward(self.test_dataLoader, 
                                                                                      write = True, 
                                                                                      N = self.test_dataset.__len__(),
                                                                                      mode = 'test')
                self.write_tensorboard_test(i, contact_histories, contact_histories_ovl, tot_loss)


    def save_model(self, epoch):
        path = self.logdir + "/policy.pth"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save({"epoch": epoch,
            "path" : self.logdir ,
            "param" : self.policy.module.state_dict(),
            }, self.logdir + "/policy.pth")
        torch.save({"epoch": epoch,
                    "optim" : self.optim.state_dict()
                    }, self.logdir + "/optim.pth")


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
        # get the file directory.
        filename = os.path.realpath(__file__).split('/')[-1]
        save_script(f'script/{filename}', logdir)
        save_script('language4contact/config/config_multi.py', logdir)
        save_script('language4contact/config/task_policy_configs.py', logdir)
        save_script('language4contact/modules_temporal_multi.py', logdir)
        save_script('language4contact/modules_temporal.py', logdir)
        save_script('language4contact/temporal_transformer.py', logdir)





    def write_tensorboard_test(self, step, contact, contact_ovl, loss0):
        contact = torch.stack(contact).unsqueeze(3)
        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact_test", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl_test", contact_ovl, max_outputs=len(contact_ovl), step=step)
            tf.summary.scalar("loss0_test", loss0/ self.test_dataLoader.__len__() , step=step)

    def write_tensorboard(self, step, contact, contact_ovl, loss):
        contact = torch.stack(contact, dim = 0).unsqueeze(3)
        contact_ovl = torch.stack(contact_ovl, dim = 0)
        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl", contact_ovl, max_outputs=len(contact_ovl), step=step)
            tf.summary.scalar("loss0", loss/ self.train_dataset.__len__() , step=step)

    def _initialize_loss(self, mode = 'p'): # 'p' = partial, 'a' = all
        if mode == 'a':
            self.tot_loss = {'sum': 0, "loss0_i":0, "loss1_i": 0, "loss_aux_i": 0, "loss0":0, "loss1": 0, "loss_aux": 0}
        elif mode == 'p':
            self.tot_loss['loss0_i'] = 0
            self.tot_loss['loss1_i'] = 0
            self.tot_loss['loss_aux_i'] = 0
        else:
            raise Exception("Error : mode not recognized")

    def  overlay_cnt_rgb(self, rgb_path, cnt_pred, aug_param):
        ## open rgb image with cv2
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:,:,:3]
        rgb = torch.tensor(rgb)
        
        rgb = F.affine(rgb.permute((2,0,1)), 
                    *aug_param, 
                    interpolation=self.randomaffine.interpolation,
                        fill=self.randomaffine.fill)
        rgb = rgb.permute(1,2,0)
        # rgb = augmentation( aug_idx, rgb, rgb = True)
        # torch.flip(rgb, (-2,))

        # print(rgb, cnt_pred)
        cnt_pred = torch.tensor(cnt_pred) * 255
        cnt_pred = cnt_pred.to(torch.uint8)
        iidx, jidx = torch.where( cnt_pred != 0)
        rgb[iidx, jidx,0] = cnt_pred[iidx, jidx] #* 255.
        rgb[iidx, jidx,1] = cnt_pred[iidx, jidx] #* 255.
        rgb[iidx, jidx,2] = cnt_pred[iidx, jidx] #* 255.

        return rgb

# CE = ContactEnergy( log_path = 'multi_conv_aug_rep_push_bce')
CE = ContactEnergy( log_path = 'multi_conv_affine_four')
CE.get_energy_field()