import torch.nn as nn
import torch.autograd as autograd
import torch
from gan import Generator, Discriminator

import numpy as np

class HiLGenerator(Generator):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)
        self.original_shape = (args.channels, args.depth_size, args.image_size, args.image_size)

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
            nn.LeakyReLU(0.2, inplace = True),

            # Activate
            # Generator produce higher dim object
            nn.Linear(1024, int(np.prod(self.original_shape))),
            nn.Tanh()
        ).to(device = self.device)

        # Projection from original_shape to image_shape original image
        # self.final_layer = nn.Linear(int(np.prod(original_shape)), int(np.prod)).cuda(device = self.device)

    def forward(self, z):
        # model returns original shape
        original_object = super().forward(z)
        # final layer returns projected shape
        # image = self.final_layer(original_object)

        # original shape in 3d
        original_object_view = original_object.view(self.original_shape)
        # image shape in 2d
        # image_view = image.view(image_shape)

        # return image_view, original_object_view
        return original_object_view


class HiLDiscriminator(Discriminator):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)

        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace = True),

            # 2nd layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),

            # activate
            nn.Linear(256, 1)
        ).to(device = self.device)
        
    def forward(self, image):
        image_flat = image.view(image.shape[0], -1)
        validity = self.model(image_flat)
        return validity


"""
    Class for generating Projection Operator distribution on Stiefel Manifold
    @params:
        Input: Noise(latent_dim)
        Return: Projection Matrix
"""
class HiProjectionLGenerator(nn.Module):


    def __init__(self, args, device, logger = None):
        super().__init__()
        image_shape = (args.channels, args.image_size, args.image_size)
        original_shape = (args.channels, args.depth_size, args.image_size, args.image_size)
        self.latent_dim = args.latent_dim
        self.device = device

        # projection from where to where? 
        self.original_dim = int(np.prod(original_shape))
        self.projected_dim = int(np.prod(image_shape))

        # Projection from n to d has dimension d * n
        projection_dim = self.original_dim * self.projected_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(True),

            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),
            nn.ReLU(True),

            nn.Linear(self.latent_dim * 4, self.latent_dim * 8),
            self.ReLU(True),

            nn.Linear(self.latent_dim * 8, projection_dim)
        ).to(device = self.device)

    def forward(self, z):
        projection = self.model(z)
        # convert to matrix form
        projection = projection.view(self.projected_dim, self.original_dim)
        projection = np.asmatrix(projection)
        return projection
        