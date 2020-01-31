from settings import ENVS
import pdb

"""
    Loader for Environment ( = dataset)
    return (size, labels, train_loader, test_loader)
"""

class EnvLoader:


    def __init__(self, env, batch_size, num_workers, image_shape, logger = None):
        # Check for implementation
        if env not in [*ENVS]:
            raise NotImplementedError

        # load dataset
        loader = ENVS[env](batch_size, num_workers, image_shape)
        for key, value in loader.items():
            setattr(self, key, value)
        self.env_name = env
        if logger is not None:
            self.logger = logger
            self.logger.info("Sucessfully loaded env: " + self.env_name)