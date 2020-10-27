import torch
import numpy as np
import logging
import logging.config
import json
import settings
import paths
import os.path

# tensorboard
from datetime import datetime
from pathlib import Path

# import loaders
from loader.agent_loader import AgentLoader
from loader.env_loader import EnvLoader
from loader.module_loader import ModuleLoader

def run(args):

    # set seed
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)

   # Check logger
   if args.logger:

      # set loggers
      with open('logger.json', 'r') as f:
         logger_config = json.load(f)
         logging.config.dictConfig(logger_config)
         root_logger = logging.getLogger("ROOT")
         env_logger = logging.getLogger("ENV")
         module_logger = logging.getLogger("MODULE")
         agent_logger = logging.getLogger("AGENT")

      # set gpu cuda
      if not torch.cuda.is_available():
         args.cuda = False
         root_logger.info("Cuda is not available. Automatically set Cuda False")
      else:
         if args.cuda:
            device = torch.device('cuda')
            # logging
            root_logger.info("Cuda True")

         else:
            device = torch.device('cpu')
            # logging
            root_logger.info("Cuda False")
      
      # Tuple image shape
      # image_shape = (args.channels, args.img_size, args.img_size)
      # logging
      # root_logger.info("Successfully retrived image shape")

      # Load Env, Module, Agent
      # Env
      env = EnvLoader(args, env_logger)
      # Module
      module = ModuleLoader(args, device, module_logger)
      # Agent
      if args.tensorboard:
         # tfboard log dir
         log_dir = os.path.join(paths.LOG_PATH, args.agent, args.env, args.tag, datetime.now().strftime("%Y%m%d%H%M%S"))
         Path(log_dir).mkdir(parents=True, exist_ok=True)
         
         # config file
         config_save_path = os.path.join(log_dir, 'config.json')
         
         # write json
         with open(config_save_path, 'w') as f:
            json.dump(vars(args), f)
         
         # logging
         root_logger.info("Saving current hyperparameters into {}".format(config_save_path))
         
         agent = AgentLoader(args, module, env, logger = agent_logger, log_dir = log_dir).agent

      else:
         agent = AgentLoader(args, module, env, logger = agent_logger).agent

      # training
      # logging
      root_logger.info("Begin training {} by {}".format(env.env_name, args.agent))
      
      # train
      agent.train()
      
      # logging
      root_logger.info("Finished training {} by {}".format(env.env_name, args.agent))



   # No logger
   else:
      # set gpu cuda
      if not torch.cuda.is_available():
         args.cuda = False
      device = torch.device('cuda' if args.cuda else 'cpu')

      # Tuple image shape
      # image_shape = (args.channels, args.img_size, args.img_size)

      # load Env, Module, Agent
      # Env
      env = EnvLoader(args)
      # Module
      module = ModuleLoader(args, device)

      # tensorboard logging
      if args.tensorboard:
         # tfboard log file
         log_dir = os.path.join(paths.LOG_PATH, args.agent, args.env, args.tag, datetime.now().strftime("%Y%m%d%H%M%S"))
         Path(log_dir).mkdir(parents=True, exist_ok=True)
         
         # config file name
         config_save_path = os.path.join(log_dir, 'config.json')
         
         # write confog.json
         with open(config_save_path, 'w') as f:
            json.dump(vars(args), f)

         # Agent
         agent = AgentLoader(args, module, env, log_dir = log_dir).agent

      else:
         agent = AgentLoader(args, module, env).agent 
      
      
      agent.train()