from language4contact.semantic_abstraction.generate_relevancy import *
from PIL import Image
'''
Huy ha's method preserve original image shape which is 256

'''
# TODO: support multiple pretrained model
# os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
torch.cuda.set_device("cuda:1")
device = 'cuda'

## semantic version
@app.command()
def generate_heatmap(
    file_path: str = typer.Argument(
        default="../../dataset/keyframes/t_00/rgb/rgb_000_088.png", help="path of image file"
    ), 
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
    save_folder_ = os.path.join(save_folder, file_path.split('.')[0].split('/')[-1])
    if not os.path.exists(save_folder_):
        os.mkdir(save_folder_) 
    else:
        print(save_folder_, "exists")
        # return

    assert img.dtype == np.uint8
    h, w, c = img.shape
    start = time()

    # for _ in range(2):
    grads = ClipWrapper.get_clip_saliency(
        img=img,
        text_labels=np.array(labels),
        prompts=prompts,
        **saliency_configs["ours"](h),
    )[0]
        # print(torch.sum(grads))
        
    # return
    print(f"get gradcam took {float(time() - start)} seconds", grads.shape)
    
    grads = grads.cpu().numpy()
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    vmin = 0.000
    # cmap = plt.get_cmap("jet")
    vmax = 0.020
    for ax, label_grad, label in zip(axes, grads, labels):
        # ax.axis("off")
        # ax.imshow(img)
        # ax.set_title(label, fontsize=12)
        label_grad -= label_grad.mean(axis=0)
        grad = np.clip((label_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
        # colored_grad = cmap(grad)
        # grad = 1 - grad
        # colored_grad[..., -1] = grad * 0.7
        # ax.imshow(colored_grad)
        # I8 = (grad).astype(np.uint8)
        img = Image.fromarray(np.uint8(grad*255))
        img.save(os.path.join(save_folder_,f"{label}.png"))

if __name__ == '__main__':
    import os
    keywords = ["Use","the", "sponge","to" ,"clean" ,"up", "the" ,"dirt", "block"]
    # keywords = ["Use","the","broom","to","brush","dirt","into","dustpan"]
    # keywords = ["Scoop","up","the","block","and","lift","it","with","spatula"]

    
    data_origrin = "dataset/heuristics_0228"
    # data_origrin = "dataset/scoop_spatula_"
    # data_origrin = "dataset/sweep_to_dustpan"

    trial_folder = os.listdir(data_origrin)
    trial_folder.sort()
    # trial_folder = trial_folder[200:]

    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origrin, tf, 'rgb')
        save_folder_i = os.path.join(data_origrin, tf, 'heatmap_huy')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            pass
        
        data_files = [os.path.join(data_origrin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]
        data_files.sort()

        for fn in data_files:
            generate_heatmap(file_path = fn, labels = keywords, save_folder = save_folder_i, prompts=["a picture of a {}."] )