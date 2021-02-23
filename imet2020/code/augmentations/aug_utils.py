import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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
