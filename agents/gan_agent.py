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

# Saving
from torchvision.utils import save_image
from paths import IMAGE_PATH, LOAD_PATH, LOG_PATH
from pathlib import Path
import os.path

# Tensorboard
from tensorboardX import SummaryWriter

import pdb

"""
    Abstract class for GAN Agent
    universal Generator_loss and Train process
    different Discriminator_loss for each module
"""
class GanAgent(ABC):


    def __init__(self, args, module, env, logger = None):
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

        # tensorboard
        self.tensorboard = args.tensorboard

        # logger
        self.logger = logger
        self.exist_logger = args.logger
        
        # save option
        self.image_save = args.image_save
        
    """
        Loss function for discriminator
        differs by each module
    """

    @abstractmethod
    def loss_D(self, real_imgs, fake_imgs):
        # FIXME: implement loss function for each module in different module
        pass

    """
        Loss function for generator
        universal for all modules
    """

    def loss_G(self, fake_imgs):

        # Can discriminate generated image?
        fake_validity = self.discriminator(fake_imgs)
        loss = -torch.mean(fake_validity)
        return loss

    """
        train process for GAN
        universal for all modules
    """

    def train(self):
        # Path to save images
        # Tensorboard Logging
        if self.tensorboard:
            writer = SummaryWriter()
        dirname = os.path.join(IMAGE_PATH, self.env.env_name, self.agent, type(self.generator).__name__)
        Path(dirname).mkdir(parents = True, exist_ok = True)
        
        # load dataloader
        train_dataloader = self.env.train

        # Setting
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        batches_done = 0
        
        # Loop epoches
        for epoch in range(self.epoches):

            # Loop dataloader
            for i, (images, _) in enumerate(train_dataloader):
                
                # batch image
                real_images = autograd.Variable(images.type(Tensor))

                # Train Discriminator
                self.discriminator.zero_grad()

                # create noise z
                if type(self.generator).__name__ is 'DCGenerator':
                    z = autograd.Variable(torch.rand(images.shape[0], self.latent_dim, 1, 1).cuda())
                else:
                    z = autograd.Variable(Tensor(np.random.normal(0, 1, (images.shape[0], self.latent_dim))))

                # Generate image
                fake_images = self.generator(z)

                # Calculate the loss
                discriminator_loss, wasserstein_distance = self.loss_D(real_images, fake_images)

                # Calculate variance
                # real_std = torch.std(self.discriminator(real_images)).item()
                # fake_std = torch.std(self.discriminator(fake_images)).item()
                # var_discrepency = abs(real_std ** 2 - fake_std ** 2)

                # Optimize discriminator
                discriminator_loss.backward()
                self.optim_D.step()

                # Train generator
                self.generator.zero_grad()

                # train G for every n step
                if i % self.n_critic == 0:

                    if self.exist_logger:
                        self.logger.info("it is %d step. Train Generator" %i)

                    fake_images = self.generator(z)
                    # loss for generator -> can overcome discriminator?
                    generator_loss = self.loss_G(fake_images)

                    generator_loss.backward()
                    self.optim_G.step()

                    if self.image_save and (batches_done % self.sample_interval == 0):
                        # Save fake images
                        filename = os.path.join(dirname, "%d.png" %batches_done)
                        save_image(fake_images.data[:25], str(filename), nrow = 5, normalize = True)

                        if self.exist_logger:
                            self.logger.info("Sucessfully saved fake image")

                    # Logger
                    if self.exist_logger:
                        self.logger.info("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W Dist: %f] [Real Var: %f] [Fake Var: %f] [Var Dis: %f]"
                            % (epoch, 
                            self.epoches, 
                            i, 
                            len(train_dataloader), 
                            discriminator_loss.item(), 
                            generator_loss.item(), 
                            wasserstein_distance.item(),
                            real_std ** 2,
                            fake_std ** 2,
                            var_discrepency))

                    # TensorboardX
                    if self.tensorboard:
                        # Add tensorboard loggings
                        logging_info = {
                            'Loss D': discriminator_loss.item(),
                            'Loss G': generator_loss.item(),
                            'Wasserstein Distance': wasserstein_distance.item(),
                            'Var_dis': var_discrepency
                        }

                        # include in tensorboardX
                        for tag, value in logging_info.items():
                            writer.add_scalar(tag, value, batches_done)

                        # images
                        if batches_done % self.sample_interval == 0:
                            tag = self.env.env_name + " " + self.agent + " " + type(self.generator).__name__ + " %d" %batches_done
                            writer.add_images(tag, fake_images, batches_done)
                        self.logger.info("TensorboardX success")
                        # Save fake images
                        # filename = os.path.join(dirname, "%d.png" %batches_done)
                        # save_image(fake_images.data[:25], str(filename), nrow = 5, normalize = True)
                                            
                    batches_done += self.n_critic
                
                # Implement tensorboardX




    def save_model(self):
        torch.save(self.generator.state_dict(), os.path.join(LOAD_PATH, self.agent, 'generator.pkl'))
        torch.save(self.generator.state_dict(), os.path.join(LOAD_PATH, self.agent, 'discriminator.pkl')) 

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(LOAD_PATH, self.agent, D_model_filename)
        G_model_path = os.path.join(LOAD_PATH, self.agent, G_model_filename)        
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))