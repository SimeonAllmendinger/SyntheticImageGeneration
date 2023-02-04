import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

from src.components.utils.logger.MasterLogger import _get_logger_
from src.components.utils.pytorch_cuda.cuda_devices import CudaDevice

PATH_BASE_CONFIG = '/home/stud01/SyntheticImageGeneration/configs/config_base.yaml'

class Opt():
    
    def __init__(self):
        self.base = _get_config_(path=PATH_BASE_CONFIG)
        self.imagen = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_imagen.yaml'))
        self.neptune = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_neptune.yaml'))
        self.param_tuning = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_param_tuning.yaml'))
        self.logger = _get_logger_(path_base_dir=self.base['PATH_BASE_DIR'], verbose=False)
        self.pytorch_cuda = CudaDevice()
        

def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)
        
    return config


if __name__ == '__main__':
    opt = Opt()
    opt.logger.info(opt.imagen)