from higan_agent import HiGanAgent
from wgangp_agent import WGanGPAgent

import torch
import numpy as np

import torch.autograd as autograd

NORM = {'L1': 1, 'L2': 2}

class HiWGanGPAgent(HiGanAgent, WGanGPAgent):

    """
        HiWGanGP
    """
    def __init__(self, args, module, env, logger = None, log_dir = None):
        super().__init__(args, module, env, logger, log_dir)

    def loss_D(self, real_imgs, fake_projected_object):
        return WGanGPAgent.loss_D(real_imgs, fake_projected_object)
    
    def train(self):
        HiGanAgent.train()