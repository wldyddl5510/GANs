import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from paths import PROJECT_PATH, DATA_PATH

import json
import logging
import logging.config


__all__ = ['mnist', 'fashion_mnist']

class MNIST(datasets.MNIST):
    def __init__(self, root, image_shape, train):
        super().__init__(root, train = train, download = False)
        if train:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(28, 4),
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, image_shape, train):
        super().__init__(root, train = train, download = False)
        if train:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(28, 4),
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081]),
            ])


def mnist(batch_size, num_workers, image_shape):
    # Labels = 0 ~ 9
    labels = list(range(10))
    return {
        # 'size': (1, 28, 28),
        'shape': image_shape,
        'labels': labels,
        'train': DataLoader(
            MNIST(DATA_PATH, image_shape, train = True),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            MNIST(DATA_PATH, image_shape, train = False),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = False,
            shuffle = False
        )
    }


def fashion_mnist(batch_size, num_workers, image_shape):
    # labels
    labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return {
        'shape': image_shape,
        'labels': labels,
        'train': DataLoader(
            FashionMNIST(DATA_PATH, image_shape, train = True),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            FashionMNIST(DATA_PATH, image_shape, train = False),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,
            drop_last = False,
            shuffle = False
        )
    }
