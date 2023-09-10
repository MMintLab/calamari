from PIL import Image
from lang_sam import LangSAM
import numpy as np
import time

# import imageio
import torch

'''
Huy ha's method preserve original image shape which is 256

'''
# TODO: support multiple pretrained model
# os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
torch.cuda.set_device("cuda:3")
device = 'cuda'
model = LangSAM()

## semantic version
def generate_heatmap(
    file_path, 
    labels,
    save_folder, 
    prompts,
):
    """
    :param  img np.nparray (0~255.)
    """
    image_pil = Image.open(file_path).convert("RGB")
    save_folder_ = os.path.join(save_folder, file_path.split('.')[0].split('/')[-1])
    if not os.path.exists(save_folder_):
        os.mkdir(save_folder_) 
    else:
        print(save_folder_, "exists")
        # return

    img = np.array(image_pil)
    h, w, c = img.shape
    
    img[:h//2] = np.zeros_like(img[:h//2] )

    image_pil = Image.fromarray(img)

    for text_prompt in labels:
        
        start = time.time()
        # masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        masks, boxes, phrases, logits = model.predict(image_pil, "sweep dust to the dustpan")

        print("predict", time.time() - start)

        masks = torch.permute(masks, (1,2,0))
        mask_array = (masks).numpy().astype(np.uint8)

        print(masks.shape, logits, phrases)
        mask_1 = Image.fromarray(mask_array[...,0]*255)
        mask_1.save(os.path.join(save_folder_,f"{text_prompt}.png"))

        # mask_2 = Image.fromarray(mask_array[...,1]*255)
        # mask_2.save(f'mask_{text_prompt}_2.png')

        breakpoint()


        # grads, txt_emb = ClipWrapper.get_clip_saliency(
        #     img=img,
        #     text_labels=np.array([label_i]),
        #     prompts=prompts,
        #     **saliency_configs["ours"](h),
        # )
        #     # print(torch.sum(grads))
            
        # # return
        # print(f"get gradcam took {float(time() - start)} seconds", grads.shape)
        
        # grads = grads.cpu().numpy()
        # vmin = 0.000
        # # cmap = plt.get_cmap("jet")
        # vmax = 0.050
        # # for ax, label_grad, label in zip(axes, grads, labels):
        #     # ax.axis("off")
        #     # ax.imshow(img)
        #     # ax.set_title(label, fontsize=12)
        # # print(label_i, label_grad.mean())
        # grads -= grads.mean()
        # grad = np.clip((grads - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
        # # colored_grad = cmap(grad)
        # # grad = 1 - grad
        # # colored_grad[..., -1] = grad * 0.7
        # # ax.imshow(colored_grad)
        # # I8 = (grad).astype(np.uint8)
        # gread_img = Image.fromarray(np.uint8(grad[0]*255))
        # gread_img.save(os.path.join(save_folder_,f"{label_i}.png"))

        # # save vector as pkl
        # np.save(os.path.join(save_folder_,f"{label_i}"), txt_emb.detach().cpu().numpy())

if __name__ == '__main__':
    import os
    # keywords = ["use","the", "sponge","to" ,"clean" ,"up", "the" ,"dirt"]

    keywords = ["sweep", "Use","the","broom","to","brush","dirt","into","dustpan"]
    # keywords = ["Scoop","up","the","block","and","lift","it","with","spatula"]
    # keywords = ["Press","push", "the", "then", "red","orange", "purple", "teal", "azure", "violet", "black", "white", "maroon", "green", "rose", "blue", "navy", "yellow", "cyan", "silver", "gray", "olive", "magenta", "button"]
    # keywords = ["Press","push", "the", "then", "red", "maroon", "button"]



    # data_origrin = "dataset/heuristics_0228"
    # data_origrin = "dataset/scoop_spatula_"
    # data_origrin = "dataset/sweep_to_dustpan_2"
    # data_origrin = "dataset/push_0"
    # data_origrin = "dataset/sweep"
    data_origrin = "dataset/sweep_to_dustpan1/episodes"

    trial_folder = os.listdir(data_origrin)
    trial_folder.sort()
    # trial_folder = trial_folder[105:]

    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origrin, tf, 'rgb')
        save_folder_i = os.path.join(data_origrin, tf, 'mask_sam')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            pass
        
        data_files = [os.path.join(data_origrin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]
        data_files.sort()

        for fn in data_files:
            generate_heatmap(file_path = fn, labels = keywords, save_folder = save_folder_i, prompts=["a picture of a {}."] )

