import os
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from paths import PROJECT_PATH, DATA_PATH

__all__ = ['celeba']

"""
    Dataset Class CelebA
"""
class CelebA(datasets.CelebA):


    def __init__(self, root, image_shape, train):
        super().__init__(root, split = train, download = True)
        # options from https://www.kaggle.com/ashishpatel26/gan-beginner-tutorial-for-pytorch-celeba-dataset
        # if train:
        self.transform = transforms.Compose([
            transforms.Resize(image_shape[1]),
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def celeba(batch_size, num_workers, image_shape):
    # labels
    return {
        'shape': image_shape,
        'train': DataLoader(
            CelebA(DATA_PATH, image_shape, train = True),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            CelebA(DATA_PATH, image_shape, train = False),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = False,
            shuffle = False
        )
    }


