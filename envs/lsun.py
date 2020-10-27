import os
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from paths import PROJECT_PATH, DATA_PATH

__all__ = ['lsun_bedroom']

LSUN_CLASSES = ['bedroom']
TRAIN_CLASSES = ['train', 'val', 'test']

"""
    Dataset Class LSUN
"""

class LSun(datasets.LSUN):


    def __init__(self, root, image_shape, train):
        super().__init__(root, classes = train, download = True)

        # options from https://www.programcreek.com/python/example/105101/torchvision.datasets.LSUN
        self.transform = transforms.Compose([
            transforms.Resize(image_shape[1]),
            transforms.CenterCrop(image_shape[1]),
            transforms.ToTensor()
        ])

def lsun_bedroom(batch_size, num_workders, image_shape):
    
    return {
        'shape': image_shape,
        'train': DataLoader(
            LSun(DATA_PATH, image_shape, train = True),
            batch_size = batch_size,
            num_workders = num_workders,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            LSun(DATA_PATH, image_shape, train = False),
            batch_size = batch_size,
            num_workders = num_workders,
            drop_last = False,
            shuffle = False
        )
    }

