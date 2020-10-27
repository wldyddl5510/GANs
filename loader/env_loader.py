from utils.loader_lists import ENVS

"""
    Loader for Environment ( = dataset)
    return (size, labels, train_loader, test_loader)
"""

class EnvLoader:

    def __init__(self, args, logger = None):
        env = args.env

        # Check Implementation
        if env not in [*ENVS]:
            raise NotImplementedError
            
        # set shape
        image_shape = (args.channels, args.img_size, args.img_size)

        # Load dataset
        loader = ENVS[env](args.batch_size, args.num_workers, image_shape)
        for key, value in loader.items():
            setattr(self, key, value)
        
        # set name
        self.env_name = env

        # logging
        if logger is not None:
            self.logger = logger
            self.logger.info("Sucessfully loaded env: " + self.env_name)