import cv2
import torch
from torch.utils.data import Dataset


class MetDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = cv2.imread(self.X[idx])[..., ::-1]
        image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)
        label = self.y[idx]
        return image, label
