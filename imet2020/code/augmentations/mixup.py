import torch
import numpy as np


def mixup(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam
