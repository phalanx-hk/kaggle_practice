import numpy as np
import cv2
from torch.utils.data import Dataset


class MetDataset(Dataset):
    def __init__(self, mode, X, y=None, transform=None):
        self.mode = mode
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if np.random.rand() < 0 and self.mode == 'train':
            image, label = self.load_mosaic(idx)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 320))
        else:
            image = cv2.imread(self.X[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)['image']
            label = self.y[idx] if self.y is not None else None
        image = image.transpose(2, 0, 1)
        if label is not None:
            return image, label
        else:
            return image
