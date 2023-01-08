import numpy as np
import torch
import itertools, time
from language4contact.utils import *

def trajectory_score(energy, seq, idx = None, Config = None):

    score = 0
    if idx is None:
        for i, s_i in enumerate(seq):
            num = torch.sum(seq[i])
            # score += torch.sum( energy * s_i.to(Config.device)) * (Config.gamma ** i) / num
            score += torch.sum( energy * s_i.to(Config.device)) * (Config.gamma ** i)  / num
    else:
        for i, j in enumerate(idx):
            s_i = energy * seq[j].to(Config.device)  #relu(seq[i] * energy)
            num = torch.sum(seq[j])
            # score += torch.sum(s_i) * (Config.gamma ** i) / num
            score += torch.sum(s_i) * (Config.gamma ** i)  / num
    return score * 1e3

def fastest_score(s_i, Config, N_bs = 100000, mode = 'n'):
    # s_i : B X window_length X img_w X img_h    
    if mode == 'n':
        negative_idxs = n_random_order( s_i.squeeze().shape[0], N_bs)
        G = Config.g_mat[negative_idxs].squeeze().T
    elif mode == 'p':
        N_bs = 1
        positive_idx = np.arange(s_i.squeeze().shape[0])
        G = Config.g_mat[positive_idx].squeeze().T

    score = s_i @ G #Config.g_mat[: seq.shape[0]] # B X W  
    # score = torch.exp( 1e-3 * s_i ) 
    score = torch.sum(score) / N_bs
    return score


def faster_score_negative(s_i, Config, N_bs = 100000):
    # s_i : B X window_length X img_w X img_h    
    
    negative_idxs = n_random_order( s_i.squeeze().shape[0], N_bs)
    G = Config.g_mat[negative_idxs].squeeze().T

    s_i = s_i @ G #Config.g_mat[: seq.shape[0]] # B X W  
    # score = torch.exp( 1e-3 * s_i ) 
    score = torch.sum(score) / N_bs

    return score

def fast_trajectory_score(energy, seq, idx = None, Config = None, return_si = False):

    score = 0
    seq = seq.squeeze(0)
    try:
        seq_neg = seq[idx].to(Config.device)
    except:
        seq_neg = torch.zeros((len(idx), seq.shape[0], seq.shape[1], seq.shape[2])).to(Config.device)
        for i in range(len(idx)):
            seq_neg[i] = seq[idx[i]].to(Config.device)

    s_i_ori = seq_neg * energy # (B X W x img_w x img_h)
    # s_i_ori = torch.sum(torch.sum(s_i_ori, dim = -1), dim = -1)  
    s_i_ori = torch.mean(torch.mean(s_i_ori, dim = -1), dim = -1)
    # s_i_ori = torch.exp( 5e-2 * s_i_ori ) 

    Config.set_gamma_mat(W = seq.shape[0], mode = 'exp')
    score = s_i_ori @ Config.g_mat[: seq.shape[0]] # B X W  

    # score = torch.sum(score)

    if return_si:
        return score, s_i_ori
    return score



def fast_score_negative(energy, cnt_gt, Config):
    random_score = 0
    cnt = 0
    N_bs = 400

    start = time.time()
    negative_idxs = n_random_order( cnt_gt.shape[1], N_bs)
    for i in range(len(negative_idxs) // N_bs +1 ) :
        st = N_bs * i
        end = min(N_bs * (i+1), len(negative_idxs))
        cnt_idx = negative_idxs[ st : end]

        random_score_i = fast_trajectory_score(energy, cnt_gt, idx = cnt_idx, Config = Config)
        random_score = random_score + random_score_i

    return random_score

    # return random_score / len(negative_idxs)

    # return torch.mean(torch.stack(random_score))

    
    # while cnt < Config.N:
    #     cnt_idx = random_order(Config.W)
    #     random_score_i = trajectory_score(energy, cnt_gt, idx = cnt_idx, Config = Config)
    #     random_score = random_score + random_score_i
    #     cnt += 1
    return random_score / Config.N



def score_negative(energy, cnt_gt, Config):
    random_score = 0
    cnt = 0

    while cnt < Config.N:
        cnt_idx = random_order(Config.W)
        random_score_i = trajectory_score(energy, cnt_gt, idx = cnt_idx, Config = Config)
        random_score = random_score + random_score_i
        cnt += 1
    return random_score / Config.N 