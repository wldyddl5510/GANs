from pathlib import Path
import os.path


# General settings
PROJECT_PATH = Path(__file__).resolve().parent.as_posix()
DATA_PATH = os.path.join(PROJECT_PATH, "data")
LOG_PATH = os.path.join(PROJECT_PATH, "logs")
IMAGE_PATH = os.path.join(PROJECT_PATH, 'images')
# DATA_DIR = '../datasets'
LOAD_PATH = os.path.join(PROJECT_PATH, 'checkpoints')
