from argparse import ArgumentParser
import torch


parser = ArgumentParser()
parser.add_argument("--gpu_id", type=str, default=[0, 1], help="used gpu")
parser.add_argument("--logdir", type=str, help="relative log directory")
parser.add_argument("--model_type", type=str, help="type of model (m1, m2, m3)")
parser.add_argument("--indir", type=str, help="test image dir")
parser.add_argument("--savedir", type=str, help="result save dir (default = log dir)")
args = parser.parse_args()

# python script/test.py --gpu_id 1 --logdir logs/transformer_seq2seq_feedback/20230116-115230/policy.pth --model_type m2 --indir dataset/real/dust.jpg


from config.config import Config
from language4contact.utils import *

if args.model_type == 'm1':
    from language4contact.modules_gt import policy
elif args.model_type == 'm2':
    from language4contact.modules_seq import policy
elif args.model_type == 'm3':
    from language4contact.modules_gt import policy

class PretrainedPolicy:
    def __init__(self, logdir, model_type, save_result = False):
        # class variable
        self.Config = Config()
        self.model_type = model_type
        self.save_result = save_result

        # Load model from log.
        pretrained = torch.load(logdir)

        for ln, w in pretrained["transformer_decoder"].items():
            if ln == "l3_1.weight":
                dimout = w.shape[0]
            if ln == 'l3.weight':
                dimout = w.shape[0]

        self.policy_pt = policy(
            self.Config.device, self.Config.dim_ft, dim_out=dimout
        ).cuda()

        self.policy_pt.segment_emb.load_state_dict(
            pretrained["segment_emb"]
        )

        self.policy_pt._image_encoder.load_state_dict(pretrained["image_encoder"])
        
        self.policy_pt.transformer_encoder.load_state_dict(
            pretrained["transformer_encoder"]
        )
        self.policy_pt.transformer_decoder.load_state_dict(
            pretrained["transformer_decoder"]
        )


    def feedforward(self, rgb_path, txt) -> torch.tensor:
        feat, seg_idx = self.policy_pt.input_processing(rgb_path, txt)
        contact_goal = {}
        if self.model_type == 'm1' :
            contact, cost, vel, out = self.policy_pt(feat, seg_idx)
            cost_reg_ori_ = round_mask(contact[0].detach().cpu()).unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])

        elif self.model_type == 'm2':
            contact = self.policy_pt(feat, seg_idx)
            w = int(np.sqrt(contact.shape[-1]))
            contact = contact.reshape( -1, w, w)
            contact = contact[:4, :, :] 
            print(torch.amax(contact))
            cost_reg_ori_ = union_img(contact.detach().cpu())
            print(contact.shape)

        elif self.model_type == 'm3':
            contact, _, _, out = self.policy_pt(feat, seg_idx)
            cost_reg_ori_ = round_mask(contact[0].detach().cpu()).unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
            contact_goal ["contact"] = contact.detach().cpu().numpy()
        
        img = cv2.imread(rgb_path[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,:3]
        w, h, _ = img.shape
        w_st = (h - w) // 2
        img = img[:,w_st:w_st+w,: ]

        # img = self.policy_pt.explainability.preprocess(Image.open(rgb_path[0])).to(
        #     self.Config.device
        # )
        # img = img.permute(1,2,0)*255.
        # img = img.detach().cpu().numpy()

        if self.model_type == 'm2':
            contact_ovl = seq_overlay_cnt_rgb(rgb_path[0], cost_reg_ori_[0].detach().cpu(), rgb=img)
            cv2.imwrite("contact_ori.png", cost_reg_ori_[0].detach().cpu().numpy()[:,:,[2,1,0]] *255. )
            cv2.imwrite("contact.png", contact_ovl.numpy()[:,:,[2,1,0]] )
            return contact_ovl

        elif self.model_type == 'm2':
            contact_ovl = overlay_cnt_rgb(rgb_path[0], cost_reg_ori_, rgb_image=img)
            cv2.imwrite("contact_ori.png", contact[0].detach().cpu().numpy() * 255.0)
            cv2.imwrite("contact.png", contact_ovl.numpy()[:,:,[2,1,0]] )
            return contact_ovl

        elif self.model_type == 'm3':
            contact_ovl = overlay_cnt_rgb(rgb_path[0], cost_reg_ori_, rgb_image=img)
            cv2.imwrite("contact_ori.png", contact[0].detach().cpu().numpy() * 255.0)
            cv2.imwrite("contact.png", contact_ovl.numpy()[:,:,[2,1,0]] )
            contact_goal["contact_ovl"] = contact_ovl.numpy()

            return contact_goal

        # energy_reg, cost_reg_ori = energy_regularization(cost.detach().cpu(), cost_reg_ori_.detach().cpu(), minmax = (0,2.5), return_original = True)



if __name__ == "__main__":
    m = PretrainedPolicy(args.logdir, args.model_type)
    energy_reg = m.feedforward([args.indir], "Use the sponge to clean up the dirt.")
    # cv2.imwrite("energy_reg.png", energy_reg.numpy() * 255.0)
