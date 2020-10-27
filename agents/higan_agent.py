from gan_agent import GanAgent

import torch 
import numpy as np

import torch.autograd as autograd

NORM = {'L1': 1, 'L2': 2}

class HiGanAgent(GanAgent):

    """
    """
    def __init__(self, args, module, env, logger = None, log_dir = None):
        super().__init__(args, module, env, logger, log_dir)
        assert (module.projection_generator != None), "HiGan need projection generator!"
        self.projection_generator = module.projection_generator
    
    def generate_original(self, noise):
        original_object = self.generator(noise)
        return original_object

    def generate_projection(self, noise):
        generated_projection = self.projection_generator(noise)
        return generated_projection

    def project(self, fake_original, generated_projection):
        projected_object = torch.mm(generated_projection, fake_original)
        return projected_object

    def train(self):
        if self.tensorboard:
            writer = SummaryWriter()

        train_dataloader = self.env.train

        #Setting
        batches_done = 0

        # path for image
        if self.image_save:
            image_path = os.path.join(str(self.log_dir), "images")
            Path(image_path).mkdir(parents=True, exist_ok=True)
        
        # Loop epoches
        for epoch in range(self.epoches):

            # Loop Dataloader
            for i, (images, _) in enumerate(train_dataloader):

                # measure time
                start_time = time.time()

                # real image
                real_images = autograd.Variable(images.type(self.tensor))

                # generate noise
                z = self.generate_noise(images)
                
                # Generate Original dim object
                fake_original_object = self.generate_original(z)

                # Generate Projection
                generated_projection = self.generate_projection(z)

                # Project fake original to fake projected
                fake_projected_object = self.project(fake_original_object, generated_projection)

                discriminator_loss = self.train_discriminator(real_images, fake_projected_object)

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
                        save_image(fake_projected_object[:25], str(image_file), nrow = 5, normalize=True)

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
                            writer.add_images(image_file, fake_projected_object, batches_done)