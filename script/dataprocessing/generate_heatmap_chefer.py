# from language4contact.semantic_abstraction.generate_relevancy import *
from language4contact.modules_shared import *
import imageio
from PIL import Image

TXT  = "Use the sponge to clean up the dirt."
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

torch.cuda.set_device("cuda:2")
device = 'cuda'
explainability = ClipExplainability(device)

def generate_heatmap(file_path, labels, save_folder, prompts=["a picture of a {}."]):

    img = np.array(imageio.imread(file_path))
    save_folder_ = os.path.join(save_folder, file_path.split('.')[0].split('/')[-1])
    if not os.path.exists(save_folder_):
        os.mkdir(save_folder_) 
    else:
        print(save_folder_, "exists")
        return
    
    for label in keywords:
        text = clip.tokenize(label).to(device)
        img = Image.open(file_path)
        img = explainability.preprocess(img).to(device)
        R_text, R_image, txt_emb = explainability.interpret(model=explainability.model, 
                                                            image=img, 
                                                            texts=text, 
                                                            device=device)

        heatmap = explainability.show_image_relevance(R_image, img, orig_image=img) #pilimage open
        heatmap = heatmap.detach().cpu().numpy()
        img = Image.fromarray(np.uint8(heatmap*255))
        img.save(os.path.join(save_folder_,f"{label}.png"))
        print("saved image in", os.path.join(save_folder_,f"{label}.png"))



if __name__ == '__main__':
    import os
    keywords = ["Use the sponge to clean up the dirt", "Use","the", "sponge" ,"to" ,"clean" ,"up", "the" ,"dirt"]
    data_origrin = "dataset/heuristics_0228"
    trial_folder = os.listdir(data_origrin)
    trial_folder.sort()

    dir_list = []
    for tf in trial_folder:
        data_folder_i = os.path.join(data_origrin, tf, 'rgb')
        save_folder_i = os.path.join(data_origrin, tf, 'heatmap')
        
        if not os.path.exists(save_folder_i):
            os.mkdir(save_folder_i) 
        else:
            pass
        
        data_files = [os.path.join(data_origrin, tf, 'rgb',f) for f in os.listdir(data_folder_i)]
        data_files.sort()

        for fn in data_files:
            generate_heatmap(file_path = fn, labels = keywords, save_folder = save_folder_i, prompts=["a picture of a {}."] )