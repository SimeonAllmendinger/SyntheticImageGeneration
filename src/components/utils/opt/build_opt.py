import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

from src.components.utils.logger.MasterLogger import _get_logger_

class Opt():
    
    def __init__(self):
        self.imagen = _get_config_('configs/config_imagen.yaml')
        self.neptune = _get_config_('configs/config_neptune.yaml')
        self.logger = _get_logger_(verbose=False)
        # TODO: Cuda ...
        

def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)
        
    return config


if __name__ == '__main__':
    opt = Opt()
    opt.logger.info(opt.imagen)