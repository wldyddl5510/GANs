import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'envs'))

from mnist import *
from cifar import *
from imagenet import *