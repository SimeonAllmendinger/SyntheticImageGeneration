import sys
import os
sys.path.append(os.path.abspath(os.curdir))

from ray import tune
from ray.air import Result

import matplotlib.pyplot as plt

from src.components.utils.opt.build_opt import Opt

PATH_EXPERIMENT='/home/stud01/SyntheticImageGeneration/results/tuning/experiment_2023-03-02-10:24:01'


def load_results(opt: Opt, experiment_path: str):
    
    restored_tuner = tune.Tuner.restore(experiment_path)
    results = restored_tuner.get_results()
    
    opt.logger.info(f"Number of results: {len(results)}")
    
    return results


def get_best_results(opt: Opt, experiment_path: str):
    
    results = load_results(opt=opt, experiment_path=experiment_path)
    best_result: Result = results.get_best_result(metric='valid_loss', mode='min')
    
    best_config = best_result.config
    best_checkpoint = best_result.checkpoint
    #best_fid = best_result.metrics['fid']
    best_valid_loss = best_result.metrics['valid_loss']
    
    opt.logger.info(f"Best valid loss: {best_valid_loss}")
    
    return results, best_config, best_result, best_checkpoint


def visualize_results(results, experiment_path):
    
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
        
    fig.savefig(os.path.join(experiment_path,'param_tuning.png'))  # save the figure to file
    plt.close(fig)


def main():
    
    opt=Opt()
    
    results, best_config, best_result, best_checkpoint = get_best_results(opt=opt, experiment_path=PATH_EXPERIMENT)
    
    opt.logger.info("Best trial config: {}".format(best_config))
    opt.logger.info("Best trial final validation loss: {}".format(best_result.metrics["valid_loss"]))
    
    visualize_results(results=results, experiment_path=PATH_EXPERIMENT)


if __name__ == '__main__':
    main()

