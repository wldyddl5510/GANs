import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'modules'))

from gan import *
from dcgan import *
from lgan import *