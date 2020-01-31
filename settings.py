# import envs
from envs.mnist import mnist, fashion_mnist

# import modules
from modules.dcgan import DCGenerator, DCDiscriminator
from modules.lgan import LGenerator, LDiscriminator

# import agents
from agents.wgangp_agent import WGanGPAgent
from agents.tgan_agent import TGanAgent

# Computation settings
EPS = 1e-5
INF = 1e5

# datsets
ENVS = {'MNIST': mnist, 'FASHION_MNIST': fashion_mnist}

# Modules
MODULES_GENERATOR = {'DCGAN': DCGenerator, 'LGAN': LGenerator}
MODULES_DISCRIMINATOR = {'DCGAN': DCDiscriminator, 'LGAN': LDiscriminator}

# Agents
AGENTS = {'WGAN_GP': WGanGPAgent, 'TGAN': TGanAgent}
