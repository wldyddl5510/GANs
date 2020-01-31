from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from gan import Generator, Discriminator

import json
import logging
import logging.config

import numpy as np

"""
    Abstract class for Generator
"""
class LGenerator(Generator):
    
    # Generator __init__() 
    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__(image_shape, latent_dim, logger)

        self.model = nn.Sequential(
            # 1st
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace = True),

            # 2nd
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace = True),

            # 3rd
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace = True),

            # 4th
            nn.Linear(512, 1024),
            nn.LeakyReLU(1024, inplace = True),

            # Activate
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )

        self.model = self.model.cuda()

    def forward(self, z):
        image = super().forward(z)
        image = image.view(image.shape[0], *self.image_shape)
        return image

class LDiscriminator(Discriminator):

    # Discriminator __init__()
    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__(image_shape, latent_dim, logger)

        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace = True),

            # 2nd layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),

            # activate
            nn.Linear(256, 1)
        )

        self.model = self.model.cuda()
    
    def forward(self, image):
        image_flat = image.view(image.shape[0], -1)
        validity = self.model(image_flat)
        return validity