import torch
import cv2
import copy
import time
from collections import Counter, OrderedDict
from sklearn.cluster import DBSCAN
import numpy as np

from calamari.utils import round_mask, union_img, overlay_cnt_rgb, seq_overlay_cnt_rgb
from calamari.test_input_processing import TestInputProcessing
from calamari.cfg.config_multi_conv import Config
from calamari.modules_temporal_multi_conv import policy

class PretrainedPolicy:
    def __init__(self, logdir, heatmap_type, save_result = False, camera_config = None):

        # class variable
        self.Config = Config(camera_config = camera_config)
        self.heatmap_type = heatmap_type # chefer or huy
        self.save_result = save_result
        # Load model from log.


        self.test_input_processing = None
        from_pretrained = True # pretrained image encoder
        pretrained = torch.load(logdir, map_location=torch.device('cpu'))
        self.policy_pt = policy(dim_in=224, dim_out=65536, image_size=224, Config=self.Config, device=self.Config.device)
        self.policy_pt.load_state_dict(pretrained["param"])
        self.policy_pt.eval() #train(False)


        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        if from_pretrained:
            self.activation = {}

            # self.image_encoder = image_encoder(self.Config.device, dim_in = 1 , dim_out = int(self.Config.dim_emb))
            image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # TODO: weights=ResNet18_Weights.DEFAULT
            image_encoder.to(self.Config.device).eval()
            image_encoder.avgpool.register_forward_hook(get_activation('avgpool'))

            self.policy_pt._image_encoder = {"model": image_encoder, "activation": self.activation}


        self.test_input_processing = TestInputProcessing(self.policy_pt,
                                                         self.heatmap_type,
                                                         center_mean = True,
                                                         pretrained=from_pretrained )

    def feedforward(self, rgb_raw, txt) -> torch.tensor:
        print("language prompt:", txt, "rgb seq length:", len(rgb_raw))
        w, h, _ = rgb_raw[0].shape
        w_st = (h - w) // 2
        rgb = [ np.zeros_like(rgb_raw[0]) for _ in range(4) ]
        for i in range(len(rgb_raw)):
            # rgb[-(i+1)] = rgb_raw[i]
            # rgb[-(i+1)][:rgb_raw[i].shape[0]//2] = np.zeros_like(rgb[i][:rgb_raw[i].shape[0]//2])
            rgb[-(i+1)] = rgb_raw[-(i+1)]

        # rgb = []
        # for i in range(len(rgb_raw)):
        #     rgb.append(rgb_raw[i])
        #     rgb[-1][:rgb_raw[i].shape[0] // 2] = np.zeros_like(rgb[-(i + 1)][:rgb_raw[i].shape[0] // 2])
            # rgb[-(i+1)] = rgb_raw[i]
            # rgb[-(i+1)][:rgb_raw[i].shape[0]//2] = np.zeros_like(rgb[-(i+1)][:rgb_raw[i].shape[0]//2])


        img = rgb[-1]
        # rgb[-len(rgb_raw):] = rgb_raw

        # txt = "push the red button"
        # img = img[:, :, :3]
        # txt = txt.replace("push", "Press")
        # txt = txt.replace("button", "Button.")
        # print(txt)

        feat, txt_emb, seg_idx = self.test_input_processing.ff(rgb, txt) # txt_emb = L x ft


        contact_goal = {}

        def get_most_common_label(clustering):
            dists = []
            new_targets = []
            iidx_ = most_common[0][0]
            return iidx_

        ## 1 = ignore, 0 = valid
        vl_mask = torch.ones(self.Config.contact_seq_l, self.Config.max_sentence_l).bool().to(self.Config.device) # N X S
        # vl_mask[-len(rgb_raw):, :len(txt_emb)-1] = 0  # N X S
        vl_mask[-len(rgb_raw):, :len(txt_emb)-1] = 0  # N X S

        tp_mask = torch.ones(1, self.Config.contact_seq_l).bool().cuda()  # N X S
        tp_mask[:, -len(rgb_raw):] = 0  # N X S

        key = torch.zeros(self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.dim_emb).to(self.Config.device)
        # key = feat
        key[-len(rgb_raw):] = feat[-len(rgb_raw):]

        # key[:, len(txt_emb):] = 0

        query = torch.zeros(self.Config.contact_seq_l, self.Config.max_sentence_l, 512).to(self.Config.device)
        txt_emb_ = txt_emb.unsqueeze(0).repeat([self.Config.contact_seq_l,*([1]*len(txt_emb.shape))]).float()
        query[:, :txt_emb_.shape[1], :] = txt_emb_



        # fused_x =
        # print("query x", query)
        # print("key", key)
        # print("vl_mask", vl_mask)
        # print("tp_mask", tp_mask)
        #
        # breakpoint()
        # contact = self.policy_pt.forward_lava(query, key, vl_mask=vl_mask, tp_mask=tp_mask)
        # print( query, key)
        # breakpoint()
        # print(key, query, vl_mask, tp_mask)
        # breakpoint()
        contact = self.policy_pt.forward_lava(key=key, query=query, vl_mask = vl_mask, tp_mask = tp_mask)

        # contact = self.policy_pt.forward_lava(query, key, vl_mask=vl_mask, tp_mask=tp_mask)

        contact = contact[0].detach().cpu()
        lim = min(torch.amax(contact) * 0.8, 0.5)
        # print("round mask contact", lim)

        cost_reg_ori_raw = round_mask(contact, thres=lim)

        try:
            iidx, jidx = torch.where(cost_reg_ori_raw == 1)
            X = torch.stack([iidx, jidx], axis=1)
            clustering = DBSCAN(eps=3, min_samples=5).fit(X)
            c = Counter(clustering.labels_)
            most_common = c.most_common(len(c))
            # print(most_common)
            # most_idx = most_common[0][0]
            c_iidx = np.where(clustering.labels_ == most_common[0][0])[0]
            # c_iidx = get_most_common_label(clustering)
            cost_reg_ori = torch.zeros_like(cost_reg_ori_raw)
            cost_reg_ori[iidx[c_iidx], jidx[c_iidx]] = torch.ones_like(cost_reg_ori[iidx[c_iidx], jidx[c_iidx]])

            # cost_reg_ori[torch.Tensor.float(iidx[c_iidx]).mean().int(),
            #                 torch.Tensor.float(jidx[c_iidx]).mean().int()] = torch.ones((1))


        except:
            c_iidx = []
            cost_reg_ori = torch.zeros_like(cost_reg_ori_raw)
            cost_reg_ori[iidx, jidx] = torch.ones_like(cost_reg_ori[iidx, jidx])


        cost_reg_ori_ = cost_reg_ori.unsqueeze(2)
        cost_reg_ori_ = torch.dstack([cost_reg_ori_, cost_reg_ori_, cost_reg_ori_])
        contact_goal["contact"] = cost_reg_ori_[np.newaxis, ...].numpy() * 255.


        def overlay_cnt_rgb_( rgb, cnt_pred):

            # print(rgb, cnt_pred)
            cnt_pred = torch.tensor(cnt_pred) * 255
            cnt_pred = cnt_pred.to(torch.uint8)
            iidx, jidx = torch.where(cnt_pred != 0)
            rgb[iidx, jidx, 0] = cnt_pred[iidx, jidx]  # * 255.
            rgb[iidx, jidx, 1] = cnt_pred[iidx, jidx]  # * 255.
            rgb[iidx, jidx, 2] = cnt_pred[iidx, jidx]  # * 255.

            return rgb

        def round_mask_(img, thres=0.5):
            ones = torch.ones_like(img)
            zeros = torch.zeros_like(img)

            img_ = torch.where(img > thres, ones, zeros)
            return img_


        contact_ovl = overlay_cnt_rgb_(img,  round_mask_(contact, thres = lim))
        contact_goal["contact_ovl"] = contact_ovl #.numpy()

    # cv2.imwrite("contact_ori.png", contact[0].detach().cpu().numpy() * 255.0)
        cv2.imwrite(f"contact.png", contact_ovl[...,[2, 1, 0]])
        # breakpoint()
        return contact_goal
