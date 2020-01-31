from settings import MODULES_GENERATOR, MODULES_DISCRIMINATOR

"""
    Class to load modules
    contain generator and discriminator
"""

class ModuleLoader:

 
    def __init__(self, generator, discriminator, image_shape, latent_dim, logger = None):

        # Check implemenation
        if generator not in [*MODULES_GENERATOR]:
            raise NotImplementedError
        if discriminator not in [*MODULES_DISCRIMINATOR]:
            raise NotImplementedError

        # get generator and discriminator
        GeneratorClass = MODULES_GENERATOR[generator]
        self.generator = GeneratorClass(image_shape, latent_dim, logger)

        DiscriminatorClass = MODULES_DISCRIMINATOR[discriminator]
        self.discriminator = DiscriminatorClass(image_shape, latent_dim, logger)