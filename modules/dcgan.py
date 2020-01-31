import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from gan import Generator, Discriminator

import json
import logging
import logging.config

import numpy as np


class DCGenerator(Generator):
    

    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__(image_shape, latent_dim, logger)

        self.model = nn.Sequential(
            # Filters [256, 512, 1024]
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels = self.latent_dim, out_channels = 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(True),

            # State (256x16x16)
            # nn.ConvTranspose2d(in_channels = 256, out_channels = int(np.prod(self.image_shape)), kernel_size = 4, stride = 2, padding = 1),
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 2),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(True),
            
            # State
            nn.ConvTranspose2d(in_channels = 128, out_channels = self.image_shape[0], kernel_size = 4, stride = 2, padding = 1),
            # output of main module --> Image (Cx32x32)

            # activation function
            nn.Tanh()
        )

        self.model = self.model.cuda()

    def forward(self, z):
        return super().forward(z)


class DCDiscriminator(Discriminator):


    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__(image_shape, latent_dim, logger)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = self.image_shape[0], out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128, affine = True),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 2),
            nn.InstanceNorm2d(256, affine = True),
            nn.LeakyReLU(0.2, inplace = True),

            # State (256x16x16)
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine = True),
            nn.LeakyReLU(0.2, inplace = True),

            # State (512x8x8)
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(1024, affine = True),
            nn.LeakyReLU(0.2, inplace = True),
            # output of model --> State (1024x4x4)

             # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        )

        self.model = self.model.cuda()

    def forward(self, image):
        validity = self.model(image)
        validity = validity.view(image.shape[0], -1)
        return validity