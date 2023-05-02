import torch
import numpy as np
import cv2
import copy
from language4contact.utils import round_mask, union_img, overlay_cnt_rgb, seq_overlay_cnt_rgb
from language4contact.test_input_processing import TestInputProcessing


class PretrainedPolicy:
    def __init__(self, logdir, model_type, heatmap_type, save_result = False, camera_config = None):

        if model_type == 'm1':
            from language4contact.config.config import Config
            from language4contact.modules_gt import policy
        elif model_type == 'm2':
            from language4contact.config.config import Config
            from language4contact.modules_seq import policy
        elif model_type == 'm3':
            from language4contact.config.config import Config
            from language4contact.modules_gt import policy
        elif model_type == 'temporal_old':
            from language4contact.config.config import Config
            from language4contact.modules_temporal_old import  policy
        elif model_type == 'temporal':
            from language4contact.config.config import Config
            from language4contact.modules_temporal import  policy
        elif model_type == 'temporal_multi':
            from language4contact.config.config_multi import Config
            from language4contact.modules_temporal_multi import policy

        # class variable
        self.Config = Config(camera_config = camera_config)
        self.model_type = model_type
        self.heatmap_type = heatmap_type # chefer or huy

        self.save_result = save_result
        # Load model from log.
        pretrained = torch.load(logdir, map_location=torch.device('cpu'))

        self.test_input_processing = None
        if model_type == 'temporal' or model_type == 'temporal_multi':
            self.policy_pt = policy(dim_in=224, dim_out=65536, image_size=224,
                                 Config=self.Config).cuda()
            # print(self.policy_pt.state_dict().keys() == pretrained["param"].keys())

            self.policy_pt.load_state_dict(pretrained["param"])
            # print(self.policy_pt.state_dict()["_image_encoder.enc1.layer1.0.conv1.1.bias"])
            self.policy_pt.train(False)

            self.test_input_processing = TestInputProcessing(self.policy_pt, self.heatmap_type )

        elif model_type == 'temporal_old':
            self.policy_pt = policy(dim_in=224, dim_out=65536, image_size=224,
                                 Config=self.Config).cuda()
            self.policy_pt.load_state_dict(pretrained["param"])
            self.policy_pt.eval()

        else:

            for ln, w in pretrained["transformer_decoder"].items():
                if ln == "l3_1.weight":
                    dimout = w.shape[0]
                if ln == 'l3.weight':
                    dimout = w.shape[0]

            self.policy_pt = policy(
                self.Config.device, self.Config.dim_ft, dim_in = 224, dim_out=dimout
            ).cuda()

            self.policy_pt._image_encoder.load_state_dict(pretrained["image_encoder"])

            self.policy_pt.transformer_encoder.load_state_dict(
                pretrained["transformer_encoder"]
            )
            self.policy_pt.transformer_decoder.load_state_dict(
                pretrained["transformer_decoder"]
            )
            self.policy_pt.eval()
    def feedforward(self, rgb, txt) -> torch.tensor:

        # import imageio
        # rgb = [np.array(imageio.imread("out/scoop/processed/t_045/rgb/rgb_045_000.png"))]
        # txt = "Scoop up the block and lift it with the spatula"

        if self.model_type == 'temporal' or self.model_type == 'temporal_multi':
            # feat, txt_emb, seg_idx = self.policy_pt.input_processing(rgb, txt, mode='test', heatmap_type = self.heatmap_type) # txt_emb = L x ft
            feat, txt_emb, seg_idx = self.test_input_processing.ff(rgb, txt) # txt_emb = L x ft
            # print("after image encoding", feat.shape)
        else:
            feat, txt_emb, seg_idx = self.policy_pt.input_processing(rgb, txt, mode='test') # txt_emb = L x ft

        contact_goal = {}
        if self.model_type == 'm1' :
            contact, cost, vel, out = self.policy_pt(feat, seg_idx)
            cost_reg_ori_ = round_mask(contact[0].detach().cpu()).unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])

        elif self.model_type == 'm2':
            contact = self.policy_pt(feat, seg_idx)
            w = int(np.sqrt(contact.shape[-1]))
            contact = contact.reshape( -1, w, w)
            contact = contact[:4, :, :].detach().cpu()

            thres = 0.5
            contact = round_mask(contact, thres = thres)
            cost_reg_ori_ = union_img(contact.detach().cpu(), thres = thres)

            contact_goal["contact"] = contact.numpy()[..., np.newaxis] * 255.


        elif self.model_type == 'm3':
            """
            feat = L X  l_contact_seq (4) X ft
            """
            contact, _, _, out = self.policy_pt(feat, seg_idx)
            cost_reg_ori = round_mask(contact[0].detach().cpu())

            cost_reg_ori_ = cost_reg_ori .unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])

            contact_goal["contact"] = cost_reg_ori_[np.newaxis,...].numpy() * 255.
            # cv2.imwrite("m3_raw_goal.png", contact_goal["contact"])

        elif self.model_type == 'temporal_old':
            self.policy_pt.train(False)

            fused_x = torch.zeros((self.policy_pt.L, 1, self.policy_pt.dim_ft)).cuda()
            fused_x[:len(rgb), :, ] = txt_emb[0]

            # visual_sentence = torch.zeros((self.policy_pt.L, 1, self.policy_pt.dim_ft)).cuda()
            visual_sentence = torch.flatten(feat[:,1:,:], start_dim=1, end_dim=2).unsqueeze(1)
            # visual_sentence[:len(rgb),: , :] = visual_sentence_

            padding_mask = torch.zeros(1, 4).bool().cuda() # N X S
            padding_mask[:,len(rgb):] = 1  # N X S

            contact = self.policy_pt.forward_lava(visual_sentence, fused_x, padding_mask= padding_mask) # TODO: different forward mode

            cost_reg_ori = round_mask(contact[0].detach().cpu(), thres = 0.5)
            cost_reg_ori_ = cost_reg_ori.unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
            contact_goal["contact"] = cost_reg_ori_[np.newaxis, ...].numpy() * 255.

        elif self.model_type == 'temporal':
            self.policy_pt.train(False)
            if len(txt_emb.shape) == 2:
                txt_emb = txt_emb.unsqueeze(0)

            visual_sentence = feat
            fused_x = torch.zeros_like(visual_sentence)
            fused_x[:len(rgb), :, :] = txt_emb[0]


            padding_mask = torch.zeros(1, 4).bool().cuda()  # N X S
            padding_mask[:, len(rgb):] = 1  # N X S

            contact = self.policy_pt.forward_lava(visual_sentence, fused_x, padding_mask=padding_mask)  # TODO: different forward mode

            cost_reg_ori = round_mask(contact[0].detach().cpu(), thres=0.5)
            cost_reg_ori_ = cost_reg_ori.unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
            contact_goal["contact"] = cost_reg_ori_[np.newaxis, ...].numpy() * 255.

        elif self.model_type == 'temporal_multi':
            vl_mask = torch.zeros(self.Config.contact_seq_l, self.Config.max_sentence_l).bool().to(self.Config.device) # N X S
            vl_mask[:, len(txt_emb):] = 1  # N X S

            tp_mask = torch.zeros(1, self.Config.contact_seq_l).bool().cuda()  # N X S
            tp_mask[:, len(rgb):] = 1  # N X S

            # visual_sentence = torch.zeros(self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.dim_emb).to(self.Config.device)
            visual_sentence = feat

            fused_x = torch.zeros(self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.dim_emb).to(self.Config.device)
            txt_emb_ = txt_emb.unsqueeze(0).repeat([self.Config.contact_seq_l,*([1]*len(txt_emb.shape))]).float()
            fused_x[:, :txt_emb_.shape[1], :] = txt_emb_

            contact = self.policy_pt.forward_lava(visual_sentence, fused_x, vl_mask=vl_mask, tp_mask=tp_mask)

            cost_reg_ori = round_mask(contact[0].detach().cpu(), thres=0.8)
            cost_reg_ori_ = cost_reg_ori.unsqueeze(2)
            cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
            contact_goal["contact"] = cost_reg_ori_[np.newaxis, ...].numpy() * 255.





        # img = cv2.imread(rgb_path[0])
        img = cv2.cvtColor(rgb[0], cv2.COLOR_BGR2RGB)[:,:,:3]
        w, h, _ = img.shape
        w_st = (h - w) // 2
        img = img[:,w_st:w_st+w,: ]


        if self.model_type == 'm2':
            contact_ovl = seq_overlay_cnt_rgb(None, cost_reg_ori_[0].detach().cpu(), rgb=img)
            contact_goal["contact_ovl"] = contact_ovl.numpy()
            cv2.imwrite("contact_ori.png", cost_reg_ori_[0].detach().cpu().numpy()[:,:,[2,1,0]] *255. )
            cv2.imwrite("contact.png", contact_ovl.numpy()[:,:,[2,1,0]] )
            return contact_goal

        elif self.model_type == 'm1':
            contact_ovl = overlay_cnt_rgb(None, cost_reg_ori_, rgb_image=img)
            contact_goal["contact_ovl"] = contact_ovl.numpy()

            return contact_goal

        elif self.model_type == 'm3':
            contact_ovl = overlay_cnt_rgb(None, cost_reg_ori_, rgb_image=img)
            contact_goal["contact_ovl"] = contact_ovl.numpy()
            # if return_heatmaps:
            #     return contact_goal, heatmaps, img_processed
            return contact_goal

        elif self.model_type == 'temporal' or self.model_type == 'temporal_old'  or self.model_type == 'temporal_multi':
            contact_ovl = overlay_cnt_rgb(None, cost_reg_ori_, rgb_image=img)
            contact_goal["contact_ovl"] = contact_ovl.numpy()
            # cv2.imwrite("contact_ori.png", contact[0].detach().cpu().numpy() * 255.0)
            cv2.imwrite("contact.png", contact_ovl.numpy()[:,:,[2,1,0]] )
            return contact_goal