import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import yaml

from src.components.utils.logger.MasterLogger import _get_logger_
from src.components.utils.pytorch_cuda.cuda_devices import CudaDevice
    
class Opt():
    
    def __init__(self):
        #
        path_cwd=get_main_working_directory('SyntheticImageGeneration')
        
        #
        self.base = {'PATH_BASE_DIR': path_cwd}
        
        # Utils
        #self.slurm_jobs = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/utils/config_SLURM_jobs.yaml'))
        self.logger = _get_logger_(path_base_dir=self.base['PATH_BASE_DIR'], verbose=False)
        self.pytorch_cuda = CudaDevice()
        self.logger.debug(f'Master_Logger started')
        self.logger.debug(f'BASE_PATH: {self.base["PATH_BASE_DIR"]}')
        
        # Models
        self.imagen = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/models/config_imagen.yaml'))
        self.elucidated_imagen = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/models/config_elucidated_imagen.yaml'))
        self.dalle2 = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/models/config_dalle2.yaml'))
        self.rendevouz = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/models/config_rendevouz.yaml'))
        self.logger.debug(f'Model configs loaded')
               
        # Visualization tools
        self.neptune = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/visualization/config_neptune.yaml'))
        self.wandb = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/visualization/config_wandb.yaml'))
        
        # Deploying
        self.param_tuning = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_param_tuning.yaml'))
        self.conductor = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_conductor.yaml'))
        self.datasets = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/config_datasets.yaml'))
        self.logger.debug(f'DATA_PATH: {self.datasets["PATH_DATA_DIR"]}')
        self.logger.debug('All options gathered')
        

def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)
        
    return config


def get_main_working_directory(name):
    
    path_base = os.getcwd()
    
    for i in range(len(path_base.split('/'))):

        if path_base.split('/')[-1] == name:
            break
        else:
            path_base = '/'.join(path_base.split('/')[0:-1])
    
    assert len(path_base) > 0, 'Could not find current directory'
    
    return path_base


def main():
    opt = Opt()
    opt.logger.info(opt.imagen)

if __name__ == '__main__':
    main()