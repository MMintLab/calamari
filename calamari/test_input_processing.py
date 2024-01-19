import torch
import numpy as np
import copy
from PIL import Image
from calamari.utils import sentence2words
from calamari.semantic_abstraction.generate_relevancy import *
from calamari.semantic_abstraction.CLIP import clip
from calamari.modules_shared import ClipExplainability

class TestInputProcessing():
    def __init__(self, policy, heatmap_type, center_mean = False, pretrained = False):
        self.Config = policy.Config
        self.L = self.Config.contact_seq_l
        self.policy = policy
        self.policy.B = 1
        self.device = self.Config.device
        self.heatmap_type = heatmap_type
        self.heatmap_model = self.load_heatmap_model()
        self.cur_heatmap_summary = None  # Current heatmap in a single image for visualization
        self.center_mean = center_mean
        self.heatmap_log = torch.zeros((4, self.Config.max_sentence_l, 256, 256))
        self.pretrained = pretrained
        self.explainability = ClipExplainability(self.Config.device)


    def load_heatmap_model(self):
        if self.heatmap_type == 'huy':
            cw = ClipWrapper(clip_model_type="ViT-B/32", device=self.device)
            return cw
        else:
            raise NotImplementedError

    def get_cur_heatmap_summary(self, heatmaps):
        # threshold the heatmaps

        mid = 80
        heatmaps_ = torch.where(heatmaps > mid, torch.ones_like(heatmaps), torch.zeros_like(heatmaps))
        L = heatmaps_.shape[0]
        weights = np.array([1 / L * (l + 1) for l in range(L)])[:, np.newaxis, np.newaxis]
        heatmaps_ = weights * heatmaps_.detach().cpu().numpy()

        heatmaps_idx = np.argmax(heatmaps_, axis=0) / L
        heatmaps_ = np.amax(heatmaps_, axis=0)

        # convert x to color array using matplotlib colormap
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap('hsv')
        img = cmap(heatmaps_)[:, :, :3] * 255.

        self.cur_heatmap_summary = np.array(img).astype(np.uint8)

    def ff(self, img, texts):
            # img_ = [np.zeros((256, 256, 3)).astype(np.uint8) for _ in range(self.L)]
            # img_[:len(img)] = img

            words = sentence2words(texts)
            labels = np.array(words)
            h, w, _ = img[0].shape
            heatmaps = torch.zeros((self.Config.contact_seq_l, self.Config.max_sentence_l, self.Config.heatmap_size[0],
                                    self.Config.heatmap_size[1])).to(self.device)

            for i in range(len(img)):
                # Read history
                if i < 3:
                    heatmap_ = self.heatmap_log[i] # TODO: I don't think we are using this..
                elif i == 3:
                # if n
                # p.sum(img[i]) != 0:
                    heatmap_ = []
                    for label_i in labels:
                        img_i = copy.copy(img[i])
                        # img_i = cv2.fastNlMeansDenoisingColored(img[i], None, 10, 10, 7, 21)
                        # img_i[:img_i.shape[0] // 2] = np.zeros_like(img_i[:img_i[i].shape[0] // 2])


                        grads, txt_emb = self.heatmap_model.get_clip_saliency(
                            img=img_i,
                            text_labels=np.array([label_i]),
                            prompts=["a picture of a {}."],
                            **saliency_configs["ours"](h),
                        )

                        grads = grads.cpu().numpy()
                        vmin = 0.000
                        vmax = 0.050

                        # mask = np.zeros_like(grads)

                        grads -= grads.mean()
                        grad = np.clip((grads - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)

                        if np.sum(img[i])  ==0:
                            grad = np.zeros(grad)

                        # grads -= grads.mean()
                        # grad = np.clip((grads - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)

                        # grad_pil = Image.fromarray(img[i])
                        # grad_pil.putalpha(Image.fromarray(grad))
                        # grad_pil.save(f"heatmap_{label_i}.png")

                        # image = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
                        # vis = show_cam_on_image(image, grad[0])
                        # vis = np.uint8(255 * vis)
                        # cv2.imwrite(f"heatmap_{label_i}.png", vis)

                        grad = np.uint8(grad[0] * 255)

                        # grad = grad.resize((self.Config.heatmap_size[0], self.Config.heatmap_size[1]))


                        heatmap_.append(torch.tensor(np.array(grad)))
                    heatmap_ = torch.stack(heatmap_).to(self.device)
                # else:
                #     heatmap_ = torch.zeros((len(labels), self.Config.heatmap_size[0], self.Config.heatmap_size[1])).to(self.device)
                    # Add observation to history. Keep history length as 3

                heatmaps[i, :len(heatmap_)] = heatmap_

                # self.heatmap_log[i, :len(labels)] = heatmap_
                # new_heatmap_log = torch.zeros_like(self.heatmap_log)
                # new_heatmap_log[:3] = self.heatmap_log[1:]
                # new_heatmap_log[-1, :heatmap_.shape[0]] = heatmap_
            # self.heatmap_log = new_heatmap_log


            # ####### TODO: remove this line after training since May 8th
            ## TODO: immediately remove this line after training since May 8th
            # texts = 'push the maroon button'
            words = sentence2words(texts)
            words.insert(0, texts)
            text = clip.tokenize(words).to(self.Config.device)
            txt_emb =  self.explainability.model.encode_text(text)
            # ################################################

            self.heatmap_log = copy.copy(heatmaps[1:].detach())
            # Compose image encoder input.
            # heatmaps[i, :heatmap_.shape[0]] = heatmap_
            # heatmaps[-(i+1), :heatmap_.shape[0]] = heatmap_

            # self.get_cur_heatmap_summary(heatmaps[len(img) - 1].to(self.device))

            self.B = 1 if len(txt_emb.shape) else txt_emb.shape[0]
            seg_idx = [0]

            ## Get clip attention
            # print(self.heatmap_log.shape)
            # breakpoint()
            img_enc_inp = torch.flatten(heatmaps, 0, 1).unsqueeze(1).float()
            if self.pretrained:
                img_enc_inp = torch.cat([img_enc_inp,
                                        img_enc_inp,
                                        img_enc_inp], dim=1).to(self.device)
                self.policy._image_encoder["model"](img_enc_inp)
                inp = self.policy._image_encoder["activation"]['avgpool'].squeeze()
            else:
                inp = self.policy._image_encoder(img_enc_inp)
            # print("inp", inp.shape)

            ## TODO: zero out the input

            inp = inp.reshape(
                (heatmaps.shape[0], heatmaps.shape[1], inp.shape[-1])
            )  # [batch size x seq x feat_dim]

            seg_idx += [1] * inp.shape[1]
            seg_idx = torch.tensor(seg_idx).repeat(inp.shape[0]).to(self.device)
            return inp, txt_emb, seg_idx