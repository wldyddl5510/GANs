from gan_agent import GanAgent

import torch
import numpy as np

import torch.autograd as autograd

NORM = {'L1': 1, 'L2': 2}

class WGanGPAgent(GanAgent):

    """
       WGAN_GP requires one more parameters than usual GAN -> lambda for gradient penalty
    """
    def __init__(self, args, module, env, logger = None, log_dir = None):
        super().__init__(args, module, env, logger, log_dir)
        self.lambda_gp = args.lambda_gp

    """
        Loss function for WGAN
        Reference: https://github.com/eriklindernoren/PyTorch-GAN#wasserstein-gan-div
    """
    def loss_D(self, real_imgs, fake_imgs):
        # discriminate images
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs)
        # calculate gradient_penalty
        gp = self.gradient_penalty(real_imgs.data, fake_imgs.data, 'L2')

        # calculate loss for wgan_gp
        # L (Wasserstein Metric) = -E(D(x)) + E(D(G(z))) + (lambda * gp)
        loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gp
        with torch.no_grad():
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity) 
        return {"loss_D": loss, "wasserstein_distance": wasserstein_distance}

    """
        Calculate lipchisz gradient penalty
        metric: L2 for default. Other option available is L1
        Reference: https://github.com/eriklindernoren/PyTorch-GAN#wasserstein-gan-div
    """
    def gradient_penalty(self, real_samples, fake_samples, norm = 'L2'):
        # Set cuda
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Check
        if norm not in [*NORM]:
            raise NotImplementedError

        # alpha: Random weight between real and fake sample
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

        # Get random interpolation between real and fake samples
        # a*real + (1-a)*fake
        interpolation = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        # discriminate interpolation data
        interpolation_discriminated = self.discriminator(interpolation)

        # fake data
        fake = autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad = False)

        # Obtain gradient
        gradient = autograd.grad(
            outputs = interpolation_discriminated,
            inputs = interpolation,
            grad_outputs = fake,
            create_graph = True,
            retain_graph = True,
            only_inputs = True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)

        # L1 norm? L2 norm?
        norm_option = NORM[norm]

        # Calculate GP
        gradient_penalty = ((gradient.norm(norm_option, dim = 1) - 1) ** norm_option).mean()

        # Return
        return gradient_penalty

