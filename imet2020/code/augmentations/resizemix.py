import torch
import numpy as np
from torchvision.transforms import Resize


def rand_region(target_size, source_size):
    t_h, t_w = target_size[2:]
    s_h, s_w = source_size[2:]
    cut_h = s_h // 2
    cut_w = s_w // 2

    cx = np.random.randint(cut_w, t_w - cut_w)
    cy = np.random.randint(cut_h, t_h - cut_h)
    x1 = cx - cut_w
    x2 = x1 + s_w
    y1 = cy - cut_h
    y2 = y1 + s_h
    return x1, y1, x2, y2


def resizemix(x, y, alpha=0.1, beta=0.8):
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'

    rand_index = torch.randperm(x.size()[0]).cuda()
    tau = np.random.uniform(alpha, beta)
    lam = tau ** 2

    H, W = x.size()[2:]
    resize_transform = Resize((int(H*tau), int(W*tau)))
    resized_x = resize_transform(x[rand_index])

    target_a = y[rand_index]
    target_b = y
    x1, y1, x2, y2 = rand_region(x.size(), resized_x.size())
    x[:, :, y1:y2, x1:x2] = resized_x
    return x, target_a, target_b, lam
