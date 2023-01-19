from argparse import ArgumentParser

import torch

from language4contact.modules_gt import policy
from config.config import Config
from language4contact.utils import *

parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0, 1], help="used gpu")
parser.add_argument("--logdir", type=str, help="relative log directory")
parser.add_argument("--model_type", type=str, help="type of model (m1, m2, m3)")
parser.add_argument("--indir", type=str, help="test image dir")
parser.add_argument("--savedir", type=str, help="result save dir (default = log dir)")
args = parser.parse_args()


class main:
    def __init__(self):
        # class variable
        self.Config = Config()
        self.model_type = args.model_type

        # Load model from log.
        pretrained = torch.load(args.logdir)
        for ln, w in pretrained["transformer_decoder"].items():
            if ln == "l3_1.weight":
                dimout = w.shape[0]

        self.policy_pt = policy(
            self.Config.device, self.Config.dim_ft, dim_out=dimout
        ).cuda()
        self.policy_pt._image_encoder.load_state_dict(pretrained["image_encoder"])
        self.policy_pt.transformer_encoder.load_state_dict(
            pretrained["transformer_encoder"]
        )
        self.policy_pt.transformer_decoder.load_state_dict(
            pretrained["transformer_decoder"]
        )

    def feedforward(self, rgb_path, txt) -> torch.tensor:
        feat, seg_idx = self.policy_pt.input_processing(rgb_path, txt)
        contact, cost, vel, out = self.policy_pt(feat, seg_idx)


        cost_reg_ori_ = round_mask(contact[0].detach().cpu()).unsqueeze(2)
        cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
        cv2.imwrite("contact.png", contact[0].detach().cpu().numpy() * 255.0)

        img = self.policy_pt.explainability.preprocess(Image.open(rgb_path[0])).to(
            self.Config.device
        )
        img = img.permute(1,2,0)[:,:,[2,1,0]]
        img = img.detach().cpu().numpy()
        contact_ovl = overlay_cnt_rgb(rgb_path[0], cost_reg_ori_, rgb_image=img)

        # energy_reg, cost_reg_ori = energy_regularization(cost.detach().cpu(), cost_reg_ori_.detach().cpu(), minmax = (0,2.5), return_original = True)
        return contact_ovl


if __name__ == "__main__":
    m = main()
    energy_reg = m.feedforward([args.indir], "Use the sponge to clean up the dirt.")
    cv2.imwrite("energy_reg.png", energy_reg.numpy() * 255.0)
# ## load model for m1, m2, m3
# if args.model_type == 'm1':
#     from language4contact.modules_gt import policy

# elif args.model_type == 'm2':
#     pass
# elif args.model_type == 'm3':
#     from language4contact.modules_gt import policy
# else:
#     raise Exception("Sorry, not a valid model type")


## read pth from the log pth

## perform a feedforward


## save result
