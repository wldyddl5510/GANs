import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'agents'))

from gan_agent import *
from tgan_agent import *
from wgangp_agent import *
