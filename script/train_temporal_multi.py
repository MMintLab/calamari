from argparse import ArgumentParser
from datetime import datetime

from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

from language4contact.utils import *
from language4contact.modules_temporal_multi import  policy
from language4contact.config.config_multi import Config
from language4contact.dataset_temporal_multi import DatasetTemporal as Dataset
from torch.utils.data import DataLoader


parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0,1], help="used gpu")
parser.add_argument("--test_idx", type=tuple, default=(30, 37), help="index of test dataset")

args = parser.parse_args()
# TXT  = "Use the sponge to clean up the dirt."
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

# device = torch.device("cuda:0,2" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

torch.cuda.set_device(f"cuda:{args.gpu_id}")
# torch.cuda.set_device(1)

class ContactEnergy():
    def __init__(self, log_path, test_idx = (30, 37)):
        self.Config = Config()
        torch.manual_seed(self.Config.seed)    

        self.test_idx = test_idx
        self.log_path = f'logs/{log_path}'
        self._initlize_writer(self.log_path)
        self._initialize_loss(mode = 'a')

        ## Data-loader
        self.train_dataset = Dataset(self.Config, mode = "train")
        self.test_dataset = Dataset(self.Config, mode = "test")
        self.train_dataLoader = DataLoader(dataset= self.train_dataset,
                         batch_size=self.Config.B, shuffle=True, drop_last=False)

        self.test_dataLoader = DataLoader(dataset=self.test_dataset,
                         batch_size=self.Config.B, shuffle=False, drop_last=False)

        print("Train dataset size", self.train_dataset.__len__(), "Test dataset size", self.test_dataset.__len__())


        ## Define policy model
        dimout = self.train_dataset.cnt_w * self.train_dataset.cnt_h
        self.policy = policy(dim_in = self.train_dataset.cnt_w, dim_out= dimout, image_size = self.train_dataset.cnt_w, Config=self.Config)
        self.policy = nn.DataParallel(self.policy.to(self.Config.device),device_ids=[1,2])

        ## Set optimizer
        self.test = False if test_idx is None else True
        self.optim = torch.optim.Adam(
                [   {"params" : self.policy.parameters()}
                ], lr=1e-5)
        

        # logdir = 'logs/multi/20230317-235231/policy.pth'
        # pretrained = torch.load(logdir, map_location=torch.device('cpu'))
        # print(pretrained.keys())
        # self.policy.module.load_state_dict(pretrained["param"])
        # self.optim.load_state_dict(pretrained["optim"])





    def feedforward(self, dataloader, write = False, N = 200):
        contact_histories = [0 for _ in range(N)] #self.train_dataLoader.__len__()
        contact_histories_ovl = [0 for _ in range(N)]  #self.train_dataLoader.__len__()

        tot_loss = 0
        # l_i_hist = []

        for data in dataloader:
            l = data["idx"]
            rgb = list(zip(*data['traj_rgb_paths']))
            traj_cnt_lst = list(zip(*data['traj_cnt_paths'])) #([B, input_length, img_w, img_h])\
            traj_cnt_img = torch.cat(data['traj_cnt_img'], dim = 0)
            txt = list(data['txt'])
            tasks = list(data["task"])

            visual_sentence, fused_x, vl_mask, tp_mask =  self.policy.module.input_processing(rgb, txt, tasks)

            # visual_sentence = torch.flatten(visual_sentence[:,1:,:], start_dim=1, end_dim=2).unsqueeze(1)
            # fused_x= torch.stack(fused_x)[:,0,:].unsqueeze(1)

            visual_sentence = visual_sentence[:,:,:]
            fused_x= torch.stack(fused_x)

            contact_seq = self.policy.module.forward_lava(visual_sentence, fused_x, vl_mask = vl_mask, tp_mask = tp_mask)
            # contact_seq = self.policy(feat, seg_idx, padding_mask = padding_mask.to(self.Config.device))

            # loss
            loss0_i = torch.norm( traj_cnt_img.to(self.Config.device) - contact_seq, p =2) / ( 150 **2 * self.train_dataset.contact_seq_l )
            # print(loss0_i,  torch.norm(traj_cnt_img.to(self.Config.device)), torch.norm(contact_seq.to(self.Config.device)))
            loss0_i = 1e6 * loss0_i
            self.optim.zero_grad()
            loss0_i.backward()
            self.optim.step()

            if write:
                rgb = zip(*data['traj_rgb_paths'])
                for l_ir, rgb_i in enumerate(rgb):
                    rgb_i = list(rgb_i)
                    rgb_i_ = rgb_i.pop()
                    while len(rgb_i_) == 0:
                        rgb_i_ = rgb_i.pop()
                    
                    l_i = l[l_ir]
                    if l_i < N: 
                        # print(l_i, rgb_i_)
                        contact_seq_round = round_mask(contact_seq[l_ir].detach().cpu())
                        contact_histories[l_i] = contact_seq_round
                        contact_histories_ovl[l_i] = self.overlay_cnt_rgb(rgb_i_, contact_seq_round)
                        # l_i_hist.append(l_i)
                    # l_ir += 1

            # self.tot_loss['loss0'] = self.tot_loss['loss0']  + loss0_i.detach().cpu()
            tot_loss += loss0_i.detach().cpu()
            # torch.cuda.empty_cache()
            # self._initialize_loss(mode = 'p')

        # return contact patches, patches overlaid with RGB, normalized total loss
        return contact_histories, contact_histories_ovl, tot_loss



    def get_energy_field(self):
        for i in range (self.Config.epoch):
            if i % 10 == 5 or i == self.Config.epoch - 1:
                self.save_model(i)
            
            self.policy.module.train(True)
            if i % 5 == 0 or i == self.Config.epoch -1: 
                contact_histories, contact_histories_ovl, tot_loss = self.feedforward(self.train_dataLoader, write = True, N = self.train_dataset.__len__())
                self.write_tensorboard(i, contact_histories, contact_histories_ovl, tot_loss)
            else:
                _, _, tot_loss = self.feedforward(self.train_dataLoader, write = False, N = 60)
            
            tqdm.write("epoch: {}, loss: {}".format(i, tot_loss))

            if i % 5 == 0 or i == self.Config.epoch -1: 
                self.policy.module.train(False)
                contact_histories, contact_histories_ovl, tot_loss = self.feedforward(self.test_dataLoader, write = True, N = self.test_dataset.__len__())
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
            tf.summary.scalar("loss0_test", loss0/ len(self.Config.test_idx) , step=step)

    def write_tensorboard(self, step, contact, contact_ovl, loss):
        contact = torch.stack(contact, dim = 0).unsqueeze(3)
        contact_ovl = torch.stack(contact_ovl, dim = 0)
        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl", contact_ovl, max_outputs=len(contact_ovl), step=step)
            tf.summary.scalar("loss0", loss/ len(self.Config.train_idx) , step=step)

    def _initialize_loss(self, mode = 'p'): # 'p' = partial, 'a' = all
        if mode == 'a':
            self.tot_loss = {'sum': 0, "loss0_i":0, "loss1_i": 0, "loss_aux_i": 0, "loss0":0, "loss1": 0, "loss_aux": 0}
        elif mode == 'p':
            self.tot_loss['loss0_i'] = 0
            self.tot_loss['loss1_i'] = 0
            self.tot_loss['loss_aux_i'] = 0
        else:
            raise Exception("Error : mode not recognized")

    def  overlay_cnt_rgb(self, rgb_path, cnt_pred):
        ## open rgb image with cv2
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:,:,:3]

        iidx, jidx = torch.where( cnt_pred != 0)
        rgb[iidx, jidx,0] = cnt_pred[iidx, jidx] * 255.
        rgb[iidx, jidx,1] = cnt_pred[iidx, jidx] * 255.
        rgb[iidx, jidx,2] = cnt_pred[iidx, jidx] * 255.

        return torch.tensor(rgb)


CE = ContactEnergy( log_path = 'multi')
CE.get_energy_field()
