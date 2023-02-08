from language4contact.semantic_abstraction.generate_relevancy import *
from PIL import Image

# TODO: support multiple pretrained model


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
    assert img.dtype == np.uint8
    h, w, c = img.shape
    start = time()
    grads = ClipWrapper.get_clip_saliency(
        img=img,
        text_labels=np.array(labels),
        prompts=prompts,
        **saliency_configs["ours"](h),
    )[0]
    print(f"get gradcam took {float(time() - start)} seconds", grads.shape)
    grads -= grads.mean(axis=0)
    grads = grads.cpu().numpy()
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    vmin = 0.002
    # cmap = plt.get_cmap("jet")
    vmax = 0.008
    for ax, label_grad, label in zip(axes, grads, labels):
        ax.axis("off")
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        grad = np.clip((label_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
        # colored_grad = cmap(grad)
        # grad = 1 - grad
        # colored_grad[..., -1] = grad * 0.7
        # ax.imshow(colored_grad)
        # I8 = (grad).astype(np.uint8)
        img = Image.fromarray(np.uint8(grad*255))
        save_folder_ = os.path.join(save_folder, file_path.split('.')[0].split('/')[-1])
        os.mkdir(save_folder_) if not os.path.exists(save_folder_) else print("path exist:", save_folder)
        img.save(os.path.join(save_folder_,f"{label}.png"))

if __name__ == '__main__':
    import os
    keywords = ["Use","the", "sponge" ,"to" ,"clean" ,"up", "the" ,"dirt"]
    data_origrin = "dataset/keyframes"
    trial_folder = os.listdir(data_origrin)
    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origrin, tf, 'rgb')
        save_folder_i = os.path.join(data_origrin, tf, 'heatmap')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            continue
        
        data_files = [os.path.join(data_origrin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]

        for fn in data_files:
            generate_heatmap(file_path = fn, labels = keywords, save_folder = save_folder_i, prompts=["a picture of a {}."] )