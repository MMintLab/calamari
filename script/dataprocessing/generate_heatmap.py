from language4contact.semantic_abstraction.generate_relevancy import *
from PIL import Image
from rlbench_utils import *
from language4contact.config.config_multi_conv import *
from skimage.util import random_noise
import copy

'''
Huy ha's method preserve original image shape which is 256

'''
# TODO: support multiple pretrained model
# os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
torch.cuda.set_device("cuda:1")
device = 'cuda'
CONFIG = Config()

  
## semantic version
@app.command()
def generate_heatmap(
    file_path,
    labels: List[str] = typer.Option(
        default=["sponge", "dust","robot", "table", "wall","wipe","desk","with","a"],
        help='list of object categories (e.g.: "nintendo switch")',
    ),
    save_folder: str = typer.Argument(
        default="", help="directory to save the result"
    ), 
    prompts: List[str] = typer.Option(
        default=["a picture of a {}."],
        help="prompt template to use with CLIP.",
    ),
):
    """
    :param  img np.nparray (0~255.)
    """
    img = np.array(imageio.imread(file_path))
    


    # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    depth_path = file_path.replace('rgb', 'front_depth').replace('front_depth_', '')
    depth_path = depth_path.split('.')[0][:-3] + f"{int(depth_path.split('.')[0][-3:])}" + ".png"
    depth = Image.open(depth_path)

    depth = image_to_float_array(depth, DEPTH_SCALE)
    pcd = pointcloud_from_depth_and_camera_params(depth=depth, 
                                            extrinsics = np.array(CONFIG.camera_config['RT']), 
                                            intrinsics = np.array(CONFIG.camera_config['K']))
    # print(pcd.shape)
    pcd_ = pcd.reshape(-1, 3)
    # print(np.amax(pcd_, axis = 0), np.amin(pcd_, axis = 0))

    mask = np.ones_like(pcd[...,0])
    mask = np.where(pcd[..., 0] >  1.2, np.zeros_like(mask), mask)
    mask = np.where(pcd[..., 0] <  1.03, np.zeros_like(mask), mask)
    mask = np.where(pcd[..., 2] >  1.465, np.zeros_like(mask), mask)
    mask =mask[..., np.newaxis]
    

    save_folder_ = os.path.join(save_folder, file_path.split('.')[0].split('/')[-1])
    save_folder_rgb_masked = save_folder_.replace('huy', 'rgb')

    if not os.path.exists(save_folder_):
        os.mkdir(save_folder_) 
    else:
        print(save_folder_, "exists")
        # return

    if not os.path.exists(save_folder_rgb_masked[:-8]):
        os.mkdir(save_folder_rgb_masked[:-8]) 
    else:
        print(save_folder_rgb_masked[-8:], "exists")
        # return

    assert img.dtype == np.uint8
    h, w, c = img.shape
    start = time()
    # img[:h//2] = np.zeros_like(img[:h//2] )


    for i in range(5):
        if i != 0:
            img_noise = (255*random_noise(img, mode='gaussian', var=0.05**2)).astype(np.uint8)

        else:
            img_noise = copy.copy(img)
    

        img_noise = img_noise * mask
        img_noise = np.uint8(img_noise)    # for label_i in labels:

        for idx, label_i in enumerate(labels):
            grads, txt_emb = ClipWrapper.get_clip_saliency(
                img=img_noise,
                text_labels=np.array([label_i]),
                prompts=prompts,
                **saliency_configs["ours"](h),
            )
                # print(torch.sum(grads))
                
            # return
            print(f"get gradcam took {float(time() - start)} seconds", grads.shape)
            
            grads = grads.cpu().numpy()
            vmin = 0.000
            # cmap = plt.get_cmap("jet")
            vmax = 0.050
            # for ax, label_grad, label in zip(axes, grads, labels):
                # ax.axis("off")
                # ax.imshow(img)
                # ax.set_title(label, fontsize=12)
            # print(label_i, label_grad.mean())
            grads -= grads.mean()
            grad = np.clip((grads - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
            # colored_grad = cmap(grad)
            # grad = 1 - grad
            # colored_grad[..., -1] = grad * 0.7
            # ax.imshow(colored_grad)
            # I8 = (grad).astype(np.uint8)



            grad_img = Image.fromarray(np.uint8(grad[0]*255))
            grad_img.save(os.path.join(save_folder_,f"{label_i}_{i}.png"))

            masked_rgb = Image.fromarray(img_noise)
            masked_rgb.save(save_folder_rgb_masked + f"{i}.png")

            # save vector as pkl
            np.save(os.path.join(save_folder_,f"{label_i}"), txt_emb.detach().cpu().numpy())

if __name__ == '__main__':
    import os
    # keywords = ["use","the", "sponge","to" ,"clean" ,"up", "the" ,"dirt","dot"]

    keywords = ["wipe", "the", "dots", "up"]

    # keywords = ["sweep", "Use","the","broom","to","brush","dirt","into","dustpan"]
    # keywords = ["draw", "C","the","letter"]

    # keywords = ["Scoop","up","the","block","and","lift","it","with","spatula"]
    # keywords = ["Press","push", "the", "then", "red","orange", "purple", "teal", "azure", "violet", "black", "white", "maroon", "green", "rose", "blue", "navy", "yellow", "cyan", "silver", "gray", "olive", "magenta", "button"]
    # keywords = ["Press","push", "the", "then", "red", "maroon", "button"]
    # keywords = ["push", "the", "red", "button"]



    # data_origrin = "dataset/heuristics_0228"
    # data_origrin = "dataset/scoop_spatula_"
    # data_origrin = "dataset/sweep_to_dustpan_2"
    # data_origin = "dataset/push_0515"
    # data_origrin = "dataset/sweep"
    # data_origin = "dataset/sweep_to_dustpan1/episodes"
    # data_origrin = "dataset/drawing/episodes"
    data_origin = "dataset/wipe_0603_2"

    trial_folder = os.listdir(data_origin)
    trial_folder.sort()
    # trial_folder = trial_folder[105:]

    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origin, tf, 'rgb')
        save_folder_i = os.path.join(data_origin, tf, 'heatmap_huy_mask_by_sentence')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            pass
        
        data_files = [os.path.join(data_origin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]
        data_files.sort()

        for fn in data_files:
            generate_heatmap(file_path = fn, labels = keywords, save_folder = save_folder_i, prompts=["a picture of a {}."] )