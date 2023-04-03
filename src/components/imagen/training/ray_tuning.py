import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import json
import yaml
import argparse
import numpy as np

from importlib import import_module
from datetime import datetime
import ray
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback


from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.training.train_imagen import train_imagen
from src.components.imagen.training.analyze_tuning import visualize_results

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to uerib@student.kit.edu')

parser.add_argument('--path_data_dir', default='$HOME/SyntheticImageGeneration/',
                    help='PATH to data directory')


def tune_train_params(opt: Opt):

    #
    ray_tuning_params = get_ray_tuning_params(opt=opt)

    #
    run_time = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    custom_dir = os.path.join(opt.base['PATH_BASE_DIR'], opt.param_tuning['tune_params']['local_dir'])
    custom_name = opt.param_tuning['tune_params']['run_name'] + run_time

    #
    scheduler = ASHAScheduler(**opt.param_tuning['tune_params']['scheduler'])

    #
    tuner = tune.Tuner(
        
        #
        tune.with_resources(
            trainable=train_imagen,
            resources=dict(**opt.param_tuning['tune_params']['resources'])
        ),
        
        #
        tune_config=tune.TuneConfig(
            metric=opt.param_tuning['tune_params']['metric'],
            mode=opt.param_tuning['tune_params']['mode'],
            scheduler=scheduler,
            num_samples=opt.param_tuning['tune_params']['num_samples'],
            time_budget_s=opt.param_tuning['tune_params']['time_budget_s'],
            search_alg=OptunaSearch()
        ),
        
        #
        run_config=air.RunConfig(
            local_dir=custom_dir,
            name=custom_name,
            log_to_file=True,
            callbacks=[WandbLoggerCallback(**opt.wandb)]
        ),
        
        #
        param_space=ray_tuning_params,
    )
    
    #
    results = tuner.fit()

    #
    best_result = results.get_best_result(metric=opt.param_tuning['tune_params']['metric'],
                                          mode=opt.param_tuning['tune_params']['mode'])

    #
    opt.logger.info("Best trial config: {}".format(best_result.config))
    opt.logger.info("Best trial final validation loss: {}".format(best_result.metrics["valid_loss"]))

    visualize_results(results=results, experiment_path=os.path.join(custom_dir,custom_name))


def get_ray_tuning_params(opt: Opt):
    
    #
    ray_tuning_params = dict()
    
    for category in opt.param_tuning['search_space'].keys():
        
        ray_tuning_params[category] = {}
        
        #
        for key, value in opt.param_tuning['search_space'][category].items():

            if 'function' in value:

                #
                if type(value['values']) == list:
                    ray_tuning_params[category][key] = load_ray_func(
                        dotpath=value['function'])(value['values'])
                else:
                    ray_tuning_params[category][key] = load_ray_func(
                        dotpath=value['function'])(**value['values'])

            else:
                #
                ray_tuning_params[category][key] = value['values']
        
    
    return ray_tuning_params


def load_ray_func(dotpath : str):
    
    #
    module_, package_, func = dotpath.rsplit(".", maxsplit=2)
    m = import_module(module_)
    p = getattr(m, package_)
    
    return getattr(p, func)
        
        
def main():
    
    #
    with open('configs/config_datasets.yaml') as f:
        config = yaml.safe_load(f)

    args = parser.parse_args()
    config['PATH_DATA_DIR'] = args.path_data_dir

    with open('configs/config_datasets.yaml', 'w') as f:
        yaml.dump(config, f)
    
    opt=Opt()
    tune_train_params(opt=opt)


if __name__ == '__main__':
    main()
