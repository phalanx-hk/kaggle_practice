import torch
import numpy as np
from torchvision.transforms import Resize

from aug_utils import rand_bbox, rand_region


__all__ = [
    "cutmix",
    "mixup",
    "resizemix",
]


def cutmix(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index,
                                      :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def mixup(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


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
