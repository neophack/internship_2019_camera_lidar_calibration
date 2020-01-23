from pathlib import Path

from pcap_reader import *
from yaml_reader import *
from logger import *
from smi_calculation import *


if __name__ == '__main__':
    local_yml_reader = YML_Read(Path().absolute())
    local_yml_reader.power()
    local_calibrator = SMI_calculations(Path().absolute()/ 'results')
    local_calibrator.power()
