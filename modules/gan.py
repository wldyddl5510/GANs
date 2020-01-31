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
    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__()
        self.model = None
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.logger = logger


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
    def __init__(self, image_shape, latent_dim, logger = None):
        super().__init__()
        self.model = None
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.logger = logger

    @abstractmethod
    def forward(self, image):
        pass
        # image_flat = img.view(image.shape[0], -1)
        # image_flat = image.view(image.shape[0], -1)
        # validity = self.model(image_flat)
        # return validity
