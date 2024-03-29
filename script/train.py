from argparse import ArgumentParser
from datetime import datetime
import time
from tqdm import tqdm
import os
import torch

parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=0, help="used gpu")
parser.add_argument(
    "--test_idx", type=tuple, default=(30, 37), help="index of test dataset"
)
parser.add_argument("--from_log", type=str, default="", help="log to the previous path")
parser.add_argument("--logdir", type=str, default="", help="log directory")
parser.add_argument("--task", type=str, default="", help="task: wipe, sweep, push")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from calamari.utils import save_script, round_mask
from calamari.modules_temporal_multi_conv import policy
from calamari.config.config_multi_conv import Config
from torch.utils.data import DataLoader
from calamari.modules_shared import *

from calamari.dataset_temporal_multi_fast import (
    DatasetTemporal as Dataset,
    augmentation,
)
import wandb

class ContactEnergy:
    def __init__(self, log_path, test_idx=(30, 37)):
        self.Config = Config(args.task)
        torch.manual_seed(self.Config.seed)

        self.activation = {}
        self.image_encoder = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self.image_encoder.to(self.Config.device).eval()
        self.image_encoder.avgpool.register_forward_hook(self.get_activation("avgpool"))
        self.device = "cuda"

        ## Data-loader
        self.train_dataset = Dataset(
            self.Config,
            mode="train",
            image_encoder={"model": self.image_encoder, "activation": self.activation},
            device=self.device,
        )
        self.test_dataset = Dataset(
            self.Config,
            mode="test",
            image_encoder={"model": self.image_encoder, "activation": self.activation},
            device=self.device,
        )
        self.train_dataLoader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.Config.B,
            shuffle=True,
            drop_last=False,
        )

        self.test_dataLoader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.Config.B,
            shuffle=False,
            drop_last=False,
        )


        ## Define policy model
        dimout = self.train_dataset.cnt_w * self.train_dataset.cnt_h

        self.policy = policy(
            dim_in=self.train_dataset.cnt_w,
            dim_out=dimout,
            image_size=self.train_dataset.cnt_w,
            Config=self.Config,
            device=self.device,
        )

        ## Set optimizer
        self.test = False if test_idx is None else True
        self.optim = torch.optim.Adam([{"params": self.policy.parameters()}], lr=1e-3)

        self.start_epoch = 0
        self.log_path = f"logs/{log_path}"

        if len(args.from_log) > 0:
            pretrained = torch.load(
                os.path.join(args.from_log, "policy.pth"),
                map_location=torch.device("cpu"),
            )
            pretrained_optim = torch.load(
                os.path.join(args.from_log, "optim.pth"),
                map_location=torch.device("cpu"),
            )

            self.policy.load_state_dict(pretrained["param"])
            self.start_epoch = pretrained["epoch"]
            self.log_path = pretrained["path"]
            self.optim.load_state_dict(pretrained_optim["optim"])

        self._initlize_writer(self.log_path)
        self._initialize_loss(mode="a")

    def change_layers(self, model):
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.fc = nn.Linear(2048, 10, bias=True)
        return model

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def feedforward(self, dataloader, write=False, N=200, mode="test"):
        contact_histories = [0 for _ in range(N)]  # self.train_dataLoader.__len__()
        contact_histories_ovl = [0 for _ in range(N)]  # self.train_dataLoader.__len__()

        tot_loss = 0
        # l_i_hist = []
        t = time.time()
        for data in dataloader:
            l = data["idx"]
            rgb = list(zip(*data["traj_rgb_paths"]))
            traj_cnt_lst = list(
                zip(*data["traj_cnt_paths"])
            )  # ([B, input_length, img_w, img_h])\
            traj_cnt_img = torch.cat(data["traj_cnt_img"], dim=0).to(self.device)
            txt = list(data["txt"])
            tasks = list(data["task"])
            aug_idx = data["aug_idx"]  # B,
            # print("1:", time.time()-t)
            # t = time.time()
            # print(flip)
            # breakpoint()

            # inp, txt_emb, vl_mask, tp_mask
            query = data["query"].flatten(0, 1)
            key = data["key"].flatten(0, 1)
            vl_mask = data["vl_mask"].flatten(0, 1)
            tp_mask = data["tp_mask"]
            # visual_sentence, fused_x, vl_mask, tp_mask =  self.policy.module.input_processing(rgb, txt, tasks, flip = flip)
            # fused_x = torch.flatten(fused_x, 0, 1)
            # print(visual_sentence.shape, fused_x.shape, vl_mask.shape, tp_mask.shape )
            # breakpoint()

            contact_seq = self.policy.forward_lava(
                key=key, query=query, vl_mask=vl_mask, tp_mask=tp_mask
            )
            # print("2:",time.time()-t)
            t = time.time()
            # contact_seq = self.policy(feat, seg_idx, padding_mask = padding_mask.to(self.Config.device))

            # loss
            loss0_i = torch.norm(traj_cnt_img - contact_seq, p=2) / (
                150**2 * self.train_dataset.contact_seq_l
            )
            # print(loss0_i,  torch.norm(traj_cnt_img.to(self.Config.device)), torch.norm(contact_seq.to(self.Config.device)))
            loss0_i = 1e5 * loss0_i

            if mode == "train":
                self.optim.zero_grad()
                loss0_i.backward()
                self.optim.step()

            # print("3:", time.time()-t)
            # t = time.time()

            if write:
                rgb = zip(*data["traj_rgb_paths"])
                for l_ir, rgb_i in enumerate(rgb):
                    rgb_i = list(rgb_i)
                    rgb_i_ = rgb_i.pop()
                    while len(rgb_i_) == 0:
                        rgb_i_ = rgb_i.pop()

                    l_i = l[l_ir]  # Batch index -> real idx
                    if l_i < N:
                        # print(l_i, rgb_i_)
                        # contact_seq_round = round_mask(traj_cnt_img[l_ir]).detach().cpu()
                        contact_seq_round = round_mask(contact_seq[l_ir].detach().cpu())
                        contact_histories[l_i] = contact_seq_round
                        contact_histories_ovl[l_i] = self.overlay_cnt_rgb(
                            rgb_i_, contact_seq_round, aug_idx[l_ir].numpy()
                        )
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

    def training(self):
        with wandb.init() as run:
            for i in tqdm(range(self.start_epoch, self.Config.epoch)):
                t = time.time()
                if i % 20 == 0 or i == self.Config.epoch - 1:
                    self.save_model(i)
                t = time.time()

                self.policy.train(True)
                if i % 20 == 0 or i == self.Config.epoch - 1:
                    contact_histories, contact_histories_ovl, tot_loss = self.feedforward(
                        self.train_dataLoader,
                        write=True,
                        N=np.amin([600, self.train_dataset.__len__()]),
                        mode="train",
                    )
                    self.write_train(
                        run, i, contact_histories, contact_histories_ovl, tot_loss
                    )
                else:
                    _, _, tot_loss = self.feedforward(
                        self.train_dataLoader,
                        write=False,
                        N=np.amin([60, self.train_dataset.__len__()]),
                        mode="train",
                    )

                tqdm.write("epoch: {}, loss: {}".format(i, tot_loss))
                if i % 20 == 0 or i == self.Config.epoch - 1:
                    self.policy.train(False)
                    contact_histories, contact_histories_ovl, tot_loss = self.feedforward(
                        self.test_dataLoader,
                        write=True,
                        N=self.test_dataset.__len__(),
                        mode="test",
                    )
                    self.write_test(
                        run, i, contact_histories, contact_histories_ovl, tot_loss
                    )

    def save_model(self, epoch):
        if epoch % 100 == 0:
            path_model = self.logdir + f"/policy_{epoch}.pth"
            path_optim = self.logdir + f"/optim_{epoch}.pth"
        else:
            path_model = self.logdir + "/policy.pth"
            path_optim = self.logdir + "/optim.pth"


        torch.save(
            {
                "epoch": epoch,
                "path": self.logdir,
                "param": self.policy.state_dict(),
            },
            path_model,
        )
        torch.save({"epoch": epoch, "optim": self.optim.state_dict()}, path_optim)

    def _initlize_writer(self, log_dir):
        # Sets up a timestamped log directory.
        logdir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.logdir = logdir

        # Clear out any prior log data.
        if os.path.exists(logdir):
            val = input(
                f"Delete the existing path {logdir} and all its contents? (y/n)"
            )
            if val == "y":
                os.remove(logdir)
            else:
                raise Exception("Error : the path exists")

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)


    def write_test(self, run, step, contact, contact_ovl, loss0):
        contact = torch.stack(contact).unsqueeze(3)
        contact_ovl = torch.stack(contact_ovl, dim=0)

        for i in range(contact.shape[0]):
            run.log({f"contact_test_{i}": wandb.Image(contact[i].detach().cpu().numpy()), 
                     f"contact_ovl_test_{i}": wandb.Image(contact_ovl[i].detach().cpu().numpy()),
                     })
        
        run.log({"loss0_test" : loss0 / self.test_dataLoader.__len__()})

    def write_train(self, run, step, contact, contact_ovl, loss):
        contact = torch.stack(contact, dim=0).unsqueeze(3)
        contact_ovl = torch.stack(contact_ovl, dim=0)

        for i in range(contact.shape[0]):
            run.log({f"contact_{i}": wandb.Image(contact[i].detach().cpu().numpy()), 
                     f"contact_ovl_{i}": wandb.Image(contact_ovl[i].detach().cpu().numpy()),
                     })
        
        run.log({"loss0" : loss / self.train_dataset.__len__()})


    def _initialize_loss(self, mode="p"):  # 'p' = partial, 'a' = all
        if mode == "a":
            self.tot_loss = {
                "sum": 0,
                "loss0_i": 0,
                "loss1_i": 0,
                "loss_aux_i": 0,
                "loss0": 0,
                "loss1": 0,
                "loss_aux": 0,
            }
        elif mode == "p":
            self.tot_loss["loss0_i"] = 0
            self.tot_loss["loss1_i"] = 0
            self.tot_loss["loss_aux_i"] = 0
        else:
            raise Exception("Error : mode not recognized")

    def overlay_cnt_rgb(self, rgb_path, cnt_pred, aug_idx):
        ## open rgb image with cv2
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:, :, :3]
        rgb = torch.tensor(rgb)

        rgb = augmentation(aug_idx, rgb, rgb=True)

        cnt_pred = cnt_pred * 255
        cnt_pred = cnt_pred.to(torch.uint8)
        iidx, jidx = torch.where(cnt_pred != 0)
        rgb[iidx, jidx, 0] = cnt_pred[iidx, jidx]  # * 255.
        rgb[iidx, jidx, 1] = cnt_pred[iidx, jidx]  # * 255.
        rgb[iidx, jidx, 2] = cnt_pred[iidx, jidx]  # * 255.

        return rgb

if __name__ == "__main__":
    CE = ContactEnergy(log_path=args.logdir)
    CE.training()
