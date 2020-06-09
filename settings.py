# Path settings
from sys import platform
from os.path import dirname, realpath, basename, exists
import sys
import os


# Path of this project
PROJECT_DIR = dirname(__file__) + '/'
PROJECT_DATA_DIR = dirname(__file__) + '/data/'
UPPER_LAYER_DIR_1 = dirname(dirname(__file__)) + '/'
UPPER_LAYER_DIR_2 = dirname(dirname(dirname(__file__))) + '/'
CURRENT_FULL_DIR = realpath(__file__)
CURRENT_FILE_NAME = basename(__file__)



#Detect and choose platform
# if platform == "darwin":
#     # OS X
#     GoogleD = UPPER_LAYER_DIR_1                 # The project is under google drive
#     Desktop = UPPER_LAYER_DIR_2 + 'Desktop/'
    

# elif platform == "linux" or platform == "linux2":
#     # linux
#     sys.exit('No linux path, please set up in settings.py! ')

# elif any([platform == "win32", platform == "win64"]):
#     # Windows
#     GoogleD = UPPER_LAYER_DIR_1                 # The project is under google drive
#     Desktop = UPPER_LAYER_DIR_2 + 'Desktop/'

# else:
#     sys.exit('Invalid system path, please check out in settings.py!')

# userhome = os.path.expanduser('~')
# desktop = userhome + '/Desktop/'