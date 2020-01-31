import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'loader'))

from agent_loader import *
from env_loader import *
from module_loader import *
