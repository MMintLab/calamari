from argparse import ArgumentParser
from datetime import datetime

from tqdm import tqdm

from language4contact.utils import *
from language4contact.modules_seq import  policy
from config.config import Config
from language4contact.dataset import DatasetSeq_front_feedback as Dataset
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0,1], help="used gpu")
parser.add_argument("--test_idx", type=tuple, default=(30, 37), help="index of test dataset")

args = parser.parse_args()

TXT  = "Use the sponge to clean up the dirt."
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.cuda.set_device("cuda:"+args.gpu_id)


class ContactEnergy():
    def __init__(self, log_path, test_idx = (30, 37)):
        self.Config = Config()
        torch.manual_seed(self.Config.seed)    

        self.test_idx = test_idx
        self.log_path = f'logs/{log_path}'
        self._initlize_writer(self.log_path)
        self._initialize_loss(mode = 'a')

        ## Data-loader
        train_dataset = Dataset(self.Config, mode = "train")
        self.train_dataLoader = DataLoader(dataset= train_dataset,
                         batch_size=self.Config.B, shuffle=True)

        self.test_dataLoader = DataLoader(dataset=Dataset(self.Config, mode = "test"),
                         batch_size=1, shuffle=False)



        ## Define policy model
        dimout = train_dataset.cnt_w * train_dataset.cnt_h
        self.policy = policy(self.Config.device, self.Config.dim_ft, dim_in = train_dataset.cnt_w, dim_out= dimout, image_size = train_dataset.cnt_w).cuda()
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
            ], lr=0.00005)


    def save_model(self, epoch):
        path = self.logdir + "/policy.pth"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save({"epoch": epoch,
                    "path" : self.logdir ,
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
        save_script('script/contact2seq_front_feedback.py', logdir)
        save_script('language4contact/modules.py', logdir)
        save_script('language4contact/utils.py', logdir)
        save_script('language4contact/loss.py', logdir)
        save_script('config/config.py', logdir)


    def _evaluate_testdataset(self):
        tot_tot_loss = 0

        self._initialize_loss(mode = 'a')

        contact_histories = []
        contact_histories_ovl = []

        for data in self.test_dataLoader:
            l = data["idx"]
            rgb = data['traj_rgb']

            traj_cnt_lst = data['traj_cnt_lst'] #([B, input_length, img_w, img_h])
            _, l_inp, c_img_w, c_img_h = traj_cnt_lst.shape


            feat, seg_idx =  self.policy.input_processing(rgb, TXT)
            contact_seq = self.policy(feat, seg_idx).reshape(-1, c_img_w, c_img_h)
            contact_seq = contact_seq[:traj_cnt_lst.shape[1], :, :]

            contact_histories.append( union_img( contact_seq.detach().cpu()) )
            contact_histories_ovl.append(self.overlay_cnt_rgb(rgb[0], contact_seq.detach().cpu()))


            # loss
            loss0_i =  torch.norm( traj_cnt_lst.to(self.Config.device) - contact_seq, p =2) / ( 150 **2 *l_inp )
            tot_tot_loss += copy.copy(loss0_i.detach().cpu())
               ## summary of result
        tot_tot_loss = tot_tot_loss / (self.test_idx[1] - self.test_idx[0] + 1)

        return contact_histories, contact_histories_ovl, tot_tot_loss

    def write_tensorboard_test(self, step, contact, contact_ovl, loss0):
        contact = torch.cat(contact, dim = 0)
        contact_ovl = torch.stack(contact_ovl, dim = 0)

        with self.file_writer.as_default():
            tf.summary.image("contact_test", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl_test", contact_ovl, max_outputs=len(contact_ovl), step=step)
            tf.summary.scalar("loss0_test", loss0/ len(self.Config.test_idx) , step=step)

    def write_tensorboard(self, step, contact, contact_ovl):
        contact = torch.cat(contact, dim = 0)
        contact_ovl = torch.stack(contact_ovl, dim = 0)
        with self.file_writer.as_default():
            tf.summary.image("contact", contact, max_outputs=len(contact), step=step)
            tf.summary.image("contact_ovl", contact_ovl, max_outputs=len(contact_ovl), step=step)
            tf.summary.scalar("loss0", self.tot_loss['loss0'].detach().cpu()/ len(self.Config.train_idx) , step=step)

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

        uic = union_img(cnt_pred.squeeze()).numpy()
        uic = uic.squeeze()
        iidx, jidx = np.where( np.sum(uic, axis = -1) != 0)
        rgb[iidx, jidx,:] = uic[iidx, jidx,:] * 255.
        return torch.tensor(rgb)

    def get_energy_field(self):
        for i in range (self.Config.epoch):
            if i % 10 == 5 or i == self.Config.epoch - 1:
                CE.save_model(i)

            self._initialize_loss(mode = 'a')

            N = 200
            contact_histories = [0 for _ in range(N)] #self.train_dataLoader.__len__()
            contact_histories_ovl = [0 for _ in range(N)]  #self.train_dataLoader.__len__()

            tot_loss = 0
            l_i_hist = []

            for data in self.train_dataLoader:
                txt = data['txt'][0]
                heatmap_folder = data['heatmap_folder']

                l = data["idx"]
                rgb = data['traj_rgb']
                traj_cnt_lst = data['traj_cnt_lst'] #([B, input_length, img_w, img_h])
                _, l_inp, c_img_w, c_img_h = traj_cnt_lst.shape


                # feat, seg_idx =  self.policy.input_processing(rgb, TXT)
                feat, seg_idx, txt_token , src_padding_mask_batch =  self.policy.input_processing_from_heatmap(heatmap_folder, txt)
                contact_seq = self.policy(feat, seg_idx, txt_token = txt_token, src_key_padding_mask = None).permute((1, 0, 2))

                contact_seq = contact_seq.reshape(contact_seq.shape[0], contact_seq.shape[1], c_img_w, c_img_h)
                contact_seq = contact_seq[:, :traj_cnt_lst.shape[1], :, :]
                # print(torch.amax(torch.flatten(contact_seq[0], -2, -1), axis = 1))

        
                    # loss
                loss0_i = torch.norm( traj_cnt_lst.to(self.Config.device).squeeze() - contact_seq.squeeze(), p =2) / ( 150 **2 *l_inp )
                loss0_i = 1e6 * loss0_i
                # loss0_i = torch.norm( torch.flatten(traj_cnt_lst.to(self.Config.device), 0, 1) - 
                #                             torch.flatten(contact_seq, 0, 1), dim = (1,2)) 
                # loss0_i = torch.norm( traj_cnt_lst.to(self.Config.device)- contact_seq, dim = (-2, -1)) 
                # print(contact_seq.shape, loss0_i.shape)
                # loss0_i = torch.mean(loss0_i) / ( 225 * 2 )
                # loss0_i = torch.norm( traj_cnt_lst.to(self.Config.device).squeeze() - contact_seq.squeeze(), p =2) / ( 150 **2 *l_inp )
                # loss0_i = 1e6 * loss0_i
                self.optim.zero_grad()
                loss0_i.backward()
                self.optim.step()


                for l_ir, l_i in enumerate(l):
                    if l_i < N: 
                        contact_histories[l_i] = union_img(contact_seq[l_ir].detach().cpu())
                        contact_histories_ovl[l_i] = self.overlay_cnt_rgb(rgb[l_ir], contact_seq[l_ir].detach().cpu())
                        l_i_hist.append(l_i)

                self.tot_loss['loss0'] = self.tot_loss['loss0']  + loss0_i.detach().cpu()
                tot_loss += loss0_i.detach().cpu()
                torch.cuda.empty_cache()
                self._initialize_loss(mode = 'p')


            print( np.amax(l_i_hist))
            if i % 5 == 0 or i == self.Config.epoch -1:
                self.write_tensorboard(i, contact_histories, contact_histories_ovl)
            
            # if i % 10 == 0 or i == self.Config.epoch -1:               
            #     contact_histories, contact_histories_ovl, loss0_i  = self._evaluate_testdataset()
            #     self.write_tensorboard_test(i, contact_histories, contact_histories_ovl, loss0_i)

            tqdm.write("epoch: {}, loss: {}".format(i, tot_loss))


CE = ContactEnergy( log_path = 'transformer_seq2seq_feedback_v3')
CE.get_energy_field()