import torch
import torchvision.transforms as T
import albumentations as A

mean = (0.485, 0.456, 0.406)  # RGB
std = (0.229, 0.224, 0.225)  # RGB


def met_transform1(size):
    transform = {
        'albu_train': A.Compose([
            A.RandomResizedCrop(size[0], size[1])
        ]),
        'torch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(15, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'albu_val': A.Compose([
            A.Resize(size[0], size[1]),
        ]),
        'torch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }
    return transform
