import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from importlib import import_module

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.training.train_imagen import train_imagen


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
            log_to_file=True
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
    opt.logger.info("Best trial final validation loss: {}".format(best_result.metrics["loss"]))

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.log_dir: result.metrics_dataframe for result in results}

    # Plot by epoch
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    #
    for d in dfs.values ():
        if not isinstance(d, type(None)):
            ax = d.loss.plot(ax=ax, legend=False)
            
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
        
    fig.savefig(os.path.join(custom_dir,custom_name,'param_tuning.png'))  # save the figure to file
    plt.close(fig)


def get_ray_tuning_params(opt: Opt):
    
    #
    ray_tuning_params = dict({'imagen': dict(),
                             'data': dict()})
    
    for category in opt.param_tuning['search_space'].keys():
    
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
    opt=Opt()
    tune_train_params(opt=opt)


if __name__ == '__main__':
    main()
