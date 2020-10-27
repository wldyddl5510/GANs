from utils.loader_lists import MODULES_GENERATOR, MODULES_DISCRIMINATOR, HI_MODULES_GENERATOR, HI_MODULES_DISCRIMINATOR, HI_MODULES_PROJECTION_GENERATOR 

"""
    Class to load modules
    contain generator and discriminator
"""

class ModuleLoader:

 
    def __init__(self, args, device,logger = None):
        self.logger = logger
        generator = args.generator
        discriminator = args.discriminator

        # task is higan
        if args.highdim_task:
            projection_generator = args.projection_generator
            if generator not in [*HI_MODULES_GENERATOR]:
                raise NotImplementedError
            if discriminator not in [*HI_MODULES_DISCRIMINATOR]:
                raise NotImplementedError
            if projection_generator not in [*HI_MODULES_PROJECTION_GENERATOR]:
                raise NotImplementedError

            ProjectionGeneratorClass = HI_MODULES_PROJECTION_GENERATOR[projection_generator]
            self.projection_generator = ProjectionGeneratorClass(args, device, logger = logger)

            if logger is not None:
                self.logger.info("Sucessfully loaded module: " + projection_generator)
            
        else:
            # retrieve generator and discriminator
            # Check implemenation
            if generator not in [*MODULES_GENERATOR]:
                raise NotImplementedError
            if discriminator not in [*MODULES_DISCRIMINATOR]:
                raise NotImplementedError

            self.projection_generator = None

        # get generator
        GeneratorClass = MODULES_GENERATOR[generator]
        self.generator = GeneratorClass(args, device, logger = logger)

        if logger is not None:
            self.logger.info("Sucessfully loaded module: " + generator)

        # get discriminator
        DiscriminatorClass = MODULES_DISCRIMINATOR[discriminator]
        self.discriminator = DiscriminatorClass(args, device, logger = logger)
        
        if logger is not None:
            self.logger.info("Sucessfully loaded module: " + discriminator)