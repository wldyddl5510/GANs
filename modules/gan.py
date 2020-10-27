from abc import abstractmethod, ABC
import torch.nn as nn

import json
import logging
import logging.config

"""
    Abstract class for Generator
"""

class Generator(ABC, nn.Module):
    
    @abstractmethod
    def __init__(self, args, device, activation = 'Tanh', logger = None):
        super().__init__()
        self.image_shape = (args.channels, args.img_size, args.img_size)
        self.model = None
        self.latent_dim = args.latent_dim
        self.device = device
        self.logger = logger
        self.activation = getattr(nn, activation)()


    def forward(self, z):
        image = self.model(z)
        if self.logger is not None:
            self.logger.info("Successfully generated image")
        return image

"""
    Abstract class for Discriminator
"""

class Discriminator(ABC, nn.Module):

    @abstractmethod
    def __init__(self, args, device, logger = None):
        super().__init__()
        self.image_shape = (args.channels, args.image_size, args.image_size)
        self.model = None
        self.latent_dim = args.latent_dim
        self.device = device
        self.logger = logger

    @abstractmethod
    def forward(self, image):
        pass
        # image_flat = img.view(image.shape[0], -1)
        # image_flat = image.view(image.shape[0], -1)
        # validity = self.model(image_flat)
        # return validity
