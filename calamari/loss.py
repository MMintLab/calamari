import numpy as np
import torch
import itertools, time
from language4contact.utils import *


def w_binary_contact(
    metric, contact_seq, contact_seq_hat, binary_contact, binary_contact_hat, device
):
    contact_patch_pred = metric(
        torch.flatten(contact_seq_hat, -2, -1),
        torch.flatten(contact_seq, -2, -1).to(device),
    )
    binary_patch_pred = metric(
        binary_contact_hat.squeeze().float(),
        binary_contact.squeeze().float().to(device),
    )

    return contact_patch_pred * 1e4 + binary_patch_pred * 1e6


def trajectory_score(energy, seq, idx=None, Config=None):
    score = 0
    if idx is None:
        for i, s_i in enumerate(seq):
            num = torch.sum(seq[i])
            # score += torch.sum( energy * s_i.to(Config.device)) * (Config.gamma ** i) / num
            score += (
                torch.sum(energy * s_i.to(Config.device)) * (Config.gamma**i) / num
            )
    else:
        for i, j in enumerate(idx):
            s_i = energy * seq[j].to(Config.device)  # relu(seq[i] * energy)
            num = torch.sum(seq[j])
            # score += torch.sum(s_i) * (Config.gamma ** i) / num
            score += torch.sum(s_i) * (Config.gamma**i) / num
    return score * 1e3
