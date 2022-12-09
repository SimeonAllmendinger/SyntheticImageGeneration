import yaml
import pandas as pd
import neptune.new as neptune

from yaml.loader import SafeLoader

def get_neptune_configs():
    with open('configs/config_neptune.yaml','r') as config_neptune:
        config_neptune_dict = yaml.load(config_neptune, Loader=SafeLoader)
    
    return pd.DataFrame(config_neptune_dict)
     
        
def get_neptune_run(verbose=False):
    
    config_neptune_dict=get_neptune_configs()
    
    run = neptune.init_run(
        project=config_neptune_dict.neptune_project_name,
        api_token=config_neptune_dict.neptune_api_token,
    )  # credentials
    
    return run


def stop_neptune_run(run :neptune.init_run(), verbose=False):
    run.stop()
    
    
def upload_neptune_run(data_item, run :neptune.init_run(), neptune_run_save_path :str, verbose=False):
    run[neptune_run_save_path].upload(data_item)


def log_neptune_run(data_item, run :neptune.init_run(), neptune_run_save_path :str, verbose=False):
    run[neptune_run_save_path].log(data_item)
        

def add_param_neptune_run(data_item, run :neptune.init_run(), neptune_run_save_path :str, verbose=False):
    run[neptune_run_save_path] = data_item