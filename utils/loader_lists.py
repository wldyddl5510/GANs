# import envs
from envs.mnist import mnist, fashion_mnist
from envs.celeba import celeba
from envs.lsun import lsun_bedroom

# import modules
from modules.dcgan import DCGenerator, DCDiscriminator
from modules.lgan import LGenerator, LDiscriminator
from modules.hilgan import HiLGenerator, HiLDiscriminator, HiProjectionLGenerator


# import agents
from agents.wgangp_agent import WGanGPAgent
from agents.higan_agent import HiGanAgent

# datsets
ENVS = {'mnist': mnist, 'fashion_mnist': fashion_mnist, 'celeba': celeba, 'lsun_bedroom': lsun_bedroom}

# Modules
MODULES_GENERATOR = {'DCGenerator': DCGenerator, 'LGenerator': LGenerator}
MODULES_DISCRIMINATOR = {'DCDiscriminator': DCDiscriminator, 'LDiscriminator': LDiscriminator}

# Higan
HI_MODULES_GENERATOR = {'HiLGenerator': HiLGenerator}
HI_MODULES_DISCRIMINATOR = {'HiLDiscriminator': HiLDiscriminator}
HI_MODULES_PROJECTION_GENERATOR = {'HiProjectionLGenerator': HiProjectionLGenerator}

# Agents
AGENTS = {'wgangp_agent': WGanGPAgent}
