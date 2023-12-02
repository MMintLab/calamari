from PIL import Image
from rlbench_utils import *
from calamari.config.config_multi_conv import *
from skimage.util import random_noise
import copy, imageio, time
from typing import List
from calamari.semantic_abstraction.CLIP.clip import ClipWrapper, saliency_configs, imagenet_templates
import typer
app = typer.Typer()


"""
We extend our gratitude to Huy Ha et al. (https://github.com/real-stanford/semantic-abstraction) for their outstanding CLIP heatmap extraction model. 
Additionally, we note that our method is not confined to this specific heatmap extraction technique.
"""


torch.cuda.set_device("cuda:1")
device = 'cuda'
CONFIG = Config()
near = 0.009999999776
far = 4.5
vmin = 0.000
vmax = 0.050

@app.command()
def generate_heatmap(
    img, labels, save_folder,
    prompts: List[str] = typer.Option(
        default=["a picture of a {}."],
        help="prompt template to use with CLIP.",
    ),
):
    """
    :param  img np.nparray (0~255.)
    """

    assert img.dtype == np.uint8
    h, w, c = img.shape

    mask = np.ones_like(pcd[...,0])
    mask = np.where(pcd[..., 0] <  -0.15, np.zeros_like(mask), mask)
    mask = np.where(pcd[..., 2] >  1.05, np.zeros_like(mask), mask)
    mask =mask[..., np.newaxis]
    

    for i in range(5):
        if i != 0:
            img_noise = (255*random_noise(img, mode='gaussian', var=0.05**2)).astype(np.uint8)

        else:
            img_noise = copy.copy(img)
    

        img_noise = img_noise * mask
        img_noise = np.uint8(img_noise) 


        ## Note: We found that going through labels one by one produces a lot more consistent heatmap
        ## when compared to passing the entire list of keywords.
        for idx, label_i in enumerate(labels):
            grads, txt_emb = ClipWrapper.get_clip_saliency(
                img=img_noise,
                text_labels=np.array([label_i]),
                prompts=prompts,
                **saliency_configs["ours"](h),
            )
                
            
            grads = grads.cpu().numpy()
            grads -= grads.mean()
            grad = np.clip((grads - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)


            grad_img = Image.fromarray(np.uint8(grad[0]*255))
            grad_img.save(os.path.join(save_folder,f"{label_i}_{i}.png"))

            masked_rgb = Image.fromarray(img_noise)
            masked_rgb.save(save_folder_rgb_masked + f"{i}.png")

            # save vector as pkl
            np.save(os.path.join(save_folder,f"{label_i}"), txt_emb.detach().cpu().numpy())

if __name__ == '__main__':
    # keywords = ["use","the", "sponge","to" ,"clean" ,"up", "the" ,"dirt","dot"]
    keywords = ["wipe", "the", "dots", "up"]
    # keywords = ["sweep", "Use","the","broom","to","brush","dirt","into","dustpan"]
    # keywords = ["Scoop","up","the","block","and","lift","it","with","spatula"]
    # keywords = ["Press","push", "the", "then", "red", "button"]
    # keywords = ["push", "the", "red", "button"]


    # data_origrin = "dataset/heuristics_0228"
    # data_origrin = "dataset/sweep_to_dustpan_2"
    # data_origin = "dataset/push_0515"
    # data_origrin = "dataset/sweep"
    # data_origin = "dataset/sweep_to_dustpan1/episodes"
    data_origin = "dataset/wipe_desk_final/all_variations/episodes"

    trial_folder = os.listdir(data_origin)
    trial_folder.sort()

    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origin, tf, 'rgb')
        save_folder_i = os.path.join(data_origin, tf, 'heatmap')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            pass
        
        data_files = [os.path.join(data_origin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]
        data_files.sort()

        for file_path in data_files:
            img = np.array(imageio.imread(file_path))
            depth_path = file_path.replace('rgb', 'front_depth').replace('front_depth_', '')
            depth_path = depth_path.split('.')[0][:-3] + f"{int(depth_path.split('.')[0][-3:])}" + ".png"

            depth = image_to_float_array(Image.open(depth_path), DEPTH_SCALE)

            depth = near + depth * (far - near)

            pcd = pointcloud_from_depth_and_camera_params(depth=depth, 
                                                    extrinsics = np.array(CONFIG.camera_config['RT']), 
                                                    intrinsics = np.array(CONFIG.camera_config['K']))



            save_folder = os.path.join(save_folder_i, file_path.split('.')[0].split('/')[-1])
            save_folder_rgb_masked = save_folder.replace('heatmap', 'heatmap_rgb')

            if not os.path.exists(save_folder):
                os.mkdir(save_folder) 
            else:
                print(save_folder, "exists")

            if not os.path.exists(save_folder_rgb_masked[:-8]):
                os.mkdir(save_folder_rgb_masked[:-8]) 
            else:
                print(save_folder_rgb_masked[-8:], "exists")

            start = time.time()
            generate_heatmap(img = img, labels = keywords, save_folder = save_folder, prompts=["a picture of a {}."] )
            
            print(time.time() - start)