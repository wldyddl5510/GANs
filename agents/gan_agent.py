# Abstract Class
from abc import abstractmethod, ABC
from modules.gan import Generator, Discriminator

# Torch
import torch
import numpy as np
import torch.autograd as autograd

# Logger
import json
import logging
import logging.config

# time calculation
import time

# Saving
from torchvision.utils import save_image
from paths import LOAD_PATH
from pathlib import Path
import os.path

# Tensorboard
from tensorboardX import SummaryWriter


"""
    Abstract class for GAN Agent
    universal Generator_loss and Train process
    different Discriminator_loss for each module
"""
class GanAgent(ABC):


    def __init__(self, args, module, env, logger = None, log_dir = None):
        # Modules
        self.generator = module.generator
        self.discriminator = module.discriminator

        # env
        self.env = env
        self.sample_interval = args.sample_interval

        # agent
        self.agent = args.agent

        # training options
        self.epoches = args.epoches
        self.latent_dim = args.latent_dim
        self.n_critic = args.n_critic

        # optimizers
        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr = args.lr, betas = (args.b1, args.b2))
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr = args.lr, betas = (args.b1, args.b2))

        # cuda
        self.cuda = args.cuda
        self.tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # tensorboard
        self.tensorboard = args.tensorboard
        self.log_dir = log_dir

        # logger
        self.logger = logger
        self.exist_logger = args.logger
        self.additional_logging = {}
        
        # save option
        self.image_save = args.image_save
        
    """
        Loss function for discriminator
        differs by each module
    """
    @abstractmethod
    def loss_D(self, real_imgs, fake_imgs) -> dict:
        # FIXME: implement loss function for each module in different module
        pass

    """
        Loss function for generator
        universal for all modules
    """
    def loss_G(self, fake_imgs) -> dict:
        # Can discriminate generated image?
        fake_validity = self.discriminator(fake_imgs)
        loss = -torch.mean(fake_validity)
        return {"loss_G": loss}

    """
        train process for GAN
        universal for all modules
    """
    def train(self):
        # Tensorboard Logging
        if self.tensorboard:
            writer = SummaryWriter()
        
        # load dataloader
        train_dataloader = self.env.train

        # Setting
        batches_done = 0

        # path for image
        if self.image_save:
            image_path = os.path.join(str(self.log_dir), "images")
            Path(image_path).mkdir(parents=True, exist_ok=True)

        # Loop epoches
        for epoch in range(self.epoches):

            # Loop dataloader
            for i, (images, _) in enumerate(train_dataloader):

                # measure time
                start_time = time.time()

                # real image
                real_images = autograd.Variable(images.type(self.tensor))

                # generate noise
                z = self.generate_noise(images)
                
                # Generate image
                fake_images = self.generate_image(z)

                # train discriminator
                discriminator_loss = self.train_discriminator(real_images, fake_images)

                # train generator for every n_critics
                if batches_done % self.n_critic == 0:
                    generator_loss = self.train_generator(z)

                # measure time
                elapsed_time = time.time() - start_time

                self.additional_logging['elapsed_time_per_batch'] = elapsed_time

                batches_done = epoch * len(train_dataloader) + i

                if batches_done % self.sample_interval == 0:
                    if self.image_save:
                        image_file = os.path.join(image_path, "%d.png" %batches_done)
                        save_image(fake_images[:25], str(image_file), nrow = 5, normalize=True)

                        if self.exist_logger:
                            self.logger.info("Successfully saved fake image "+ str(image_file))

                    logging_info = discriminator_loss.copy()
                    logging_info.update(generator_loss)
                    logging_info.update(self.additional_logging)
                    #logging_info['elapsed_time'] = elapsed_time

                    if self.exist_logger:
                        self.log_training(epoch, i, len(train_dataloader), batches_done, logging_info) 
                    
                    if self.tensorboard:
                        for tag, value in logging_info.items():
                            writer.add_scalar(tag, value, batches_done)
                            writer.add_images(image_file, fake_images, batches_done)

    
    """
        Discriminator training
    """
    def train_discriminator(self, real_images, fake_images):
        
        # init
        self.discriminator.zero_grad()
        
        # calculate loss
        discriminator_loss = self.loss_D(real_images, fake_images)

        # optimize discriminator
        discriminator_loss.get("loss_D").backward()
        self.optim_D.step()
        
        return discriminator_loss

    """
        Generator training
    """
    def train_generator(self, noise):
         # init
        self.generator.zero_grad()

        # batches of fake images
        fake_images = self.generate_image(noise)

        # compute loss
        generator_loss = self.loss_G(fake_images)

        # optimize generator loss
        generator_loss.get("loss_G").backward()
        self.optim_G.step()

        return generator_loss


    """
        noise generation
    """
    def generate_noise(self, images):
        if type(self.generator).__name__ is 'DCGenerator':
            z = autograd.Variable(torch.rand(images.shape[0], self.latent_dim, 1, 1).cuda())
        else:
            z = autograd.Variable(Tensor(np.random.normal(0, 1, (images.shape[0], self.latent_dim))))

        return z

    """
        fake image generation from input noise
    """
    def generate_image(self, noise):
        fake_images = self.generator(noise)
        return fake_images

    def log_training(self, epoch, i, len_batch, batches_done, logging_info: dict) -> None:
        logging_message = "[Epoch: %d/%d] || [Batch: %d/%d] || [batches_done: %d] || " %(epoch, self.epoches, i, len_batch, batches_done)
        for key, value in logging_info.items():
            logging_message += "[{}: {}] || ".format(key, value)
        self.logger.info(logging_message)

    def save_model(self):
        torch.save(self.generator.state_dict(), os.path.join(LOAD_PATH, self.agent, 'generator.pkl'))
        torch.save(self.generator.state_dict(), os.path.join(LOAD_PATH, self.agent, 'discriminator.pkl')) 

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(LOAD_PATH, self.agent, D_model_filename)
        G_model_path = os.path.join(LOAD_PATH, self.agent, G_model_filename)        
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))