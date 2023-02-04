import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import matplotlib.pyplot as plt

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from importlib import import_module

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.training.train_imagen import train_imagen


def tune_train_params(opt: Opt):

    ray_tuning_params = dict()
    
    for key, value in opt.param_tuning['search_space'].items():
        
        if 'function' in value:
            
            #
            if type(value['values']) == list:
                ray_tuning_params[key] = load_ray_func(
                    dotpath=value['function'])(value['values'])
            else:
                ray_tuning_params[key] = load_ray_func(
                    dotpath=value['function'])(**value['values'])

        else:
            #
            ray_tuning_params[key] = value['values']

    scheduler = ASHAScheduler(**opt.param_tuning['tune_params']['scheduler'])

    tuner = tune.Tuner(
        tune.with_resources(
            trainable=train_imagen,
            resources=dict(**opt.param_tuning['tune_params']['resources'])),
        tune_config=tune.TuneConfig(
            metric=opt.param_tuning['tune_params']['metric'],
            mode=opt.param_tuning['tune_params']['mode'],
            scheduler=scheduler,
            num_samples=opt.param_tuning['tune_params']['num_samples'],
            time_budget_s=opt.param_tuning['tune_params']['time_budget_s'],
            search_alg=OptunaSearch()
        ),
        param_space=ray_tuning_params,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result(metric=opt.param_tuning['tune_params']['metric'],
                                          mode=opt.param_tuning['tune_params']['mode'])

    print("--- Best trial config: {}".format(best_result.config))
    print("--- Best trial final validation loss: {}".format(best_result.metrics["loss"]))

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.log_dir: result.metrics_dataframe for result in results}
    print(dfs)
    print(type(dfs))

    # Plot by epoch
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    for d in dfs.values ():
        ax = d.loss.plot(ax=ax, legend=False)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        
    fig.savefig('./results/param_tuning.png')   # save the figure to file
    plt.close(fig)
    
    
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
