import numpy as np
import imageio
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import torch
import cv2
import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import tensorflow as tf
import io

relu = torch.nn.ReLU()


from language4contact.utils import *

def draw_line(start, end):

    if start[0] == end[0]:
        pixels = []
        y_min = np.amin([start[1], end[1]])
        y_max = np.amax([start[1], end[1]])

        for y_i in range(y_min, y_max+1 ):
            cand = [start[0], y_i]
            pixels.append(cand)
        return np.array(pixels)

    elif start[1] == end[1]:
        pixels = []
        x_min = np.amin([start[0], end[0]])
        x_max = np.amax([start[0], end[0]])

        for x_i in range(x_min, x_max+1 ):
            cand = [x_i, start[1]]
            pixels.append(cand)
        return np.array(pixels)
    else:
        pixels = []
        slant = (start[1] - end[1]) / (start[0] - end[0])

        if start[0] > end[0]:
            start, end = end, start

        pixels.append(start[:2].tolist())
        for i in range(1, end[0] - start[0] - 1):
            cand = [start[0]+i, np.floor(slant * i + start[1]).astype(int)]
            if cand not in pixels:
                pixels.append(cand)

        pixels.append(end[:2].tolist())

        if start[1] > end[1]:
            start, end = end, start

        slant =  (start[0] - end[0]) / (start[1] - end[1])
        for i in range(1, end[1] - start[1] - 1):
            cand = [np.floor(slant * i + start[0]).astype(int), start[1]+i]
            if cand not in pixels:
                pixels.append(cand)

        return np.array(pixels)

def rotation(rot_T, offset):
    ## remove the offset
    t_offset_r = np.identity(3)
    t_offset_r[:2, 2] = -offset

    ## add the offset
    t_offset_a = np.identity(3)
    t_offset_a[:2, 2] = offset

    return t_offset_a @ rot_T @ t_offset_r


def get_transform(idx, moves, start_end):
    if idx in [4,5]:
        offset = 0.5 * (start_end[0] + start_end[-1])[:2]
        return rotation(moves[idx], offset)
    else:
        return moves[idx]


def forsee(contact_field, start, options, moves, gamma):
    scores = np.zeros((options.shape[0]))
    # get contact transforms
    for i in range(options.shape[0]):
        cur = copy.deepcopy(start)
        cur_cf = copy.deepcopy(contact_field)
        for j in range(options.shape[1]):

            # get the score of the current contact
            cur_int =  np.round(cur[0]).astype(int), np.round(cur[-1]).astype(int)
            pixels = draw_line(cur_int[0], cur_int[-1]) # N x 3
            try:
                scores[i] += np.sum(cur_cf[pixels[:,0], pixels[:,1]]) * (gamma ** j)
            except:
                print(pixels)
            cur_cf[pixels[:, 0], pixels[:, 1]] = -20

            # transform the current contact
            tf = get_transform(options[i][j], moves, cur)
            cur[0] = tf @ cur[0]
            cur[1] = tf @ cur[1]



    #         start_imgs = np.repeat(contact_field[np.newaxis, :, :], next.shape[0], axis=0)
    # start_ = np.repeat(start, transforms.shape[0], axis=0)
    # next = np.round(start_ @ transforms).astype(int) # (15625, W, 3)
    # start_imgs = np.repeat(contact_field[np.newaxis, :, :], next.shape[0], axis=0)
    #
    # contact_mask = np.zeros_like(start_imgs)
    # for i in range(start_.shape[1]):
    #     contact_mask[:, next[:,i,0], next[:,i,1]] = 1 * (gamma ** i)
    # score_2d = start_imgs * contact_mask
    # score = np.sum(score_2d.reshape(score_2d.shape[0], -1) , axis=1)

    return int(options[np.argsort(scores)[-1]][0])
# a function that returns random order of 0 ~ N-1 numbers


def random_order(n):
    rnd = np.random.permutation(n)
    rnd_ori = np.arange(n)

    while np.sum(rnd == rnd_ori) == 0:
        rnd = np.random.permutation(n)
    return rnd

def n_random_order(n, m):
    cnt = 0
    rnd = []
    while cnt < m:
        rnd.append(random_order(n))
        cnt += 1
    return torch.tensor(rnd)



def trajectory_score(energy, seq, idx = None, gamma = 0.8, device = 'cpu'):

    score = 0
    if idx is None:
        for i, s_i in enumerate(seq):
            num = torch.sum(seq[i])
            # score += torch.sum( energy * s_i.to(device)) * (gamma ** i) / num
            score += torch.sum( energy * s_i.to(device)) * (gamma ** i) # / num
    else:
        for i, j in enumerate(idx):
            s_i = seq[j].to(device) * energy #relu(seq[i] * energy)
            num = torch.sum(seq[j])
            # score += torch.sum(s_i) * (gamma ** i) / num
            score += torch.sum(s_i) * (gamma ** i) # / num
    return score #* 5e2

def read_mask(fn, d):
    if d == 1:
        return torch.tensor(cv2.imread(fn))[:,:,0] / 255.0
    if d == 3:
        bgr = cv2.imread(fn)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.tensor(rgb) / 255.0

def fn2img(fn_lst, d = 1):
    input_lst = []
    for fn in fn_lst:
        img = read_mask(fn, d = d)
        input_lst.append(img)
    return input_lst

def folder2filelist(traj_gt_path, sort = True):
    fn_lst = []
    for (dirpath, dirnames, filenames) in os.walk(traj_gt_path):
        for f in filenames:
            if f.endswith('.png'):
                fn_lst.append( os.path.abspath(os.path.join(dirpath, f)))

    if sort:
        fn_lst.sort()
    return fn_lst

def get_traj_mask(traj):
    mask = torch.zeros_like(read_mask(traj[0], d = 1))
    for fn in traj:
        img = read_mask(fn, d = 1)
        mask = torch.where(img == 1, img, mask)

    # save mask as fn
    # cv2.imwrite(save_fn, mask.numpy()* 255.)
    return mask.numpy()

def energy_regularization(energy, mask = None, minmax = None, return_original = False):

    ## if variable is numpy
    if not type(energy) is np.ndarray:
        energy = energy.detach().cpu().numpy()
    # print(energy.shape)
    if len(energy.shape) == 2:
        energy = energy[np.newaxis,...]

    if mask is None:
        mask = np.ones_like(energy)
    else:
        mask = mask.numpy()

    if len(mask.shape) == 2:
        mask = mask[np.newaxis,...]

    bidx, iidx, jidx = np.where(mask != 0)
    bidx_m, iidx_m, jidx_m = np.where(mask == 0)


    if minmax is None:
        max = np.amax(energy[iidx, jidx]) #.reshape(mask.shape[0], -1), axis = -1)# [:, np.newaxis, np.newaxis]
        min = np.amin(energy[iidx, jidx])#.reshape(mask.shape[0], -1), axis = -1)#[:, np.newaxis, np.newaxis]
    else:
        min, max = minmax

    # convert x to color array using matplotlib colormap
    cmap = mpl.cm.get_cmap('YlOrRd')
    # min = -max
    img_ori = cmap((energy - min) / (max - min))[..., :3]
    img = img_ori.copy()
    
    if len(mask.shape) == 2:
        img[ iidx_m, jidx_m, :] = 0
    elif len(mask.shape) == 3:
        img[ bidx_m, iidx_m, jidx_m, :] = 0


    if return_original:
        return torch.tensor(img), torch.tensor(img_ori)

    return torch.tensor(img)



def save_energy(energy, mask, f_n):
    img = energy_regularization(energy, mask)
    img = img[:,:, [2,1,0]] * 255.
    cv2.imwrite(f_n, img )
    # plt.imshow(img, cmap='YlOrRd')
    # plt.colorbar()

def save_script(filename, target_folder):
    # not included in output file
    out_filename = os.path.join(target_folder, filename.split('/')[-1])

    with open(filename, 'r') as f:
        with open(out_filename, 'w') as out:
            for line in (f.readlines()[:-7]): #remove last 7 lines
                print(line, end='', file=out)

### Image Processing ###

def gradient_img(img, device = 'cuda'):
    img = img.unsqueeze(1)
    img = img.squeeze(0)
    ten=torch.unbind(img)
    x=ten[0].unsqueeze(0).unsqueeze(0)
    
    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x=conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

def extract_cnt_pxls(img):
    iidx, jidx = np.where(img > 0.8)
    pts = np.stack([jidx, iidx], axis = 1)
    return pts

def round_mask(img):
    ones = torch.ones_like(img)
    zeros = torch.zeros_like(img)

    img  = torch.where(img > 0.8, ones, zeros)
    return img


# union of list of image array
def union_img_binary(img_lst):
    img = np.zeros_like(img_lst[0])
    for idx, i in enumerate(img_lst):
        new = np.ones_like(img_lst[0])
        img  = np.where(i > 0.8, new, img)
    return img

# union of list of image array
def union_img(img_lst):

    img = torch.zeros_like(img_lst[0])
    for idx, i in enumerate(img_lst):
        new = torch.ones_like(img_lst[0]) * (len(img_lst) - idx) / len(img_lst)
        img  = np.where(i > 0.8, new, img)
    
    img_color = energy_regularization(img, minmax= (0,1))
    # mask out zeros for visualization
    iidx, jidx = np.where(img == 0.)
    img_color[:, iidx, jidx, :] = 0
    return img_color


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def gen_plot(s_i):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(s_i
    )
    plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def  overlay_cnt_rgb(rgb_path, uib, uic, Config):

    ## open rgb image with cv2
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:,:,:3]


    ## iidx, jidx of pixels that are in contact from table surface
    cnt_pxls = extract_cnt_pxls(uib)

    if len(cnt_pxls) == 0:
        overlaid = copy.copy(rgb)
    else:
        ## convert contact pixels to 3D Cartesian world coordinates
        cnt_pts = Config.contact_frame_to_world(cnt_pxls) # project from desk to front camera image space
        
        ## convert 3D Cartesian world coordinates to 2D image coordinates        
        mapped = Config.world_to_camera(cnt_pts)

        ## idx =  (iidx , jidx) converted to camera frame
        idx = mapped[:2, :] / mapped[2, :]
        idx = np.round(idx).astype(int)
        idx = np.clip(idx, 0, 255)
        
        ## contact map in camera frame
        overlaid = copy.copy(rgb)

        ## extract contact color from original contact map (indicate contact order)
        mapped_cnt = uic[cnt_pxls[:,1],cnt_pxls[:,0],:]
        overlaid[ idx[1,:], idx[0,:] ,:] = mapped_cnt * 255.

    return torch.tensor(overlaid)