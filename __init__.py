from pathlib import Path

from lidar import *
from logger import *
from YML_reader import *


if __name__ == '__main__':

    local_yml_reader =  YML_Read(Path().absolute())
    local_yml_reader.power()