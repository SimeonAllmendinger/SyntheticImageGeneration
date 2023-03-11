import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import neptune.new as neptune

from src.components.utils.opt.build_opt import Opt

class Neptune_AI():
    def __init__(self, opt):
        self.configs = opt.neptune
        
        
    def start_neptune_run(self):
        self.run = neptune.init_run(**self.configs)  # credentials
        opt.logger.info('Neptune run started')


    def stop_neptune_run(self, opt):
        self.run.stop()
        opt.logger.info('Neptune run stopped')
        
        
    def upload_neptune_run(self, opt, data_item, neptune_run_save_path :str):
        self.run[neptune_run_save_path].upload(data_item)
        opt.logger.debug('Data Item Uploaded on Neptune Server')
        

    def log_neptune_run(self, opt, data_item, neptune_run_save_path :str):
        self.run[neptune_run_save_path].log(data_item)
        opt.logger.debug('Data Item logged on Neptune Server')
            

    def add_param_neptune_run(self, opt, data_item, neptune_run_save_path :str):
        self.run[neptune_run_save_path] = data_item
        opt.logger.debug(neptune_run_save_path + ' value saved on Neptune Server')
        

if __name__ == "__main__":
    opt = Opt()
    neptune_ai = Neptune_AI(opt=opt)
    neptune_ai.stop_neptune_run(opt=opt)