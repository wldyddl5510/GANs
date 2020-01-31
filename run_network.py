import torch
import numpy as np
import logging
import logging.config
import json
import settings

# import loaders
from loader.agent_loader import AgentLoader
from loader.env_loader import EnvLoader
from loader.module_loader import ModuleLoader
"""
with open('logger.json', 'r') as f:
   logger_config = json.load(f)
logging.config.dictConfig(logger_config)
# set logger for root process
root_logger = logging.getLogger("ROOT")
"""
def run(args):
   # logging config

   # Check logger
   if args.logger:
      with open('logger.json', 'r') as f:
         logger_config = json.load(f)
         logging.config.dictConfig(logger_config)
         root_logger = logging.getLogger("ROOT")
         env_logger = logging.getLogger("ENV")
         module_logger = logging.getLogger("MODULE")
         agent_logger = logging.getLogger("AGENT")
      
   # set seed
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)

   # set gpu
   if not torch.cuda.is_available():
      args.cuda = False
      root_logger.info("Cuda is not available. Automatically set Cuda False.")
   if args.logger:
      root_logger.info("Cuda True")
   # Set device
   device = torch.device('cuda' if args.cuda else 'cpu')
   # torch.set_default_tensor_type('torch.Doubletensor')

   # Tuple image shape
   image_shape = (args.channels, args.img_size, args.img_size)

   if args.logger:
      # Env
      env = EnvLoader(args.env, args.batch_size, args.num_workers, image_shape, env_logger)
      # Module
      module = ModuleLoader(args.generator, args.discriminator, image_shape, args.latent_dim, module_logger)
      # Agent
      agent = AgentLoader(args, module, env, agent_logger).agent
   
   else:
      # Env
      env = EnvLoader(args.env, args.batch_size, args.num_workers, image_shape)
      # Module
      module = ModuleLoader(args.generator, args.discriminator, image_shape, args.latent_dim)
      # Agent
      agent = AgentLoader(args, module, env).agent 

   # Train
   agent.train()