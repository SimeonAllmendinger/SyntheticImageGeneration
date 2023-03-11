import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import ray
import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm
from ray.air import session
from ray.air.checkpoint import Checkpoint

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.model.build_imagen import Imagen_Model
from src.components.imagen.data_manager.imagen_dataset import get_train_valid_ds, get_train_valid_dl
from src.components.imagen.testing.test_imagen import test_text2images


def train_imagen(tune_config=None, reporter=None):
    
    #
    opt = Opt()
    
    tqdm_disable=False
    
    if tune_config:
        
        #
        tqdm_disable=True
        
        #
        opt.imagen['imagen'] = tune_config['imagen']
        opt.imagen['trainer'] = opt.param_tuning['trainer']
        opt.imagen['data']['dataset'] = tune_config['data']['dataset']
        opt.imagen['data']['use_existing_data_files'] = tune_config['data']['use_existing_data_files']
        opt.imagen['data']['use_phase_labels'] = tune_config['data']['use_phase_labels']
        
        opt.logger.warning('Override configs with tune params')

    # Build an imagen model
    imagen_model = Imagen_Model(opt=opt)
    unet_number=opt.imagen['trainer']['unet_number']
    
    # 
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt)
    sample_dataset = get_train_valid_ds(opt=opt, testing=True)

    # instantiate train_dataloader and val_dataloader
    train_generator, valid_generator = get_train_valid_dl(opt=opt, 
                                                          train_dataset=train_dataset,
                                                          valid_dataset=valid_dataset)
    
    # Add data size (length of all elements) to imagen config
    opt.imagen['trainer']['train_size'] = train_dataset.__len__()
    opt.imagen['validation']['valid_size'] = valid_dataset.__len__()
    
    # Add dataloaders to imagen trainer (training, validation)
    imagen_model.trainer.add_train_dataloader(dl=train_generator)
    imagen_model.trainer.add_valid_dataloader(dl=valid_generator)
    
    model_checkpoint_file = opt.imagen['trainer']['PATH_MODEL_CHECKPOINT'].replace('.pt', f'-u{unet_number}.pt')
    
    path_run_dir = ''
    
    if opt.imagen['trainer']['param_tuning']:
        
        #
        path_run_dir = reporter.logdir
        model_checkpoint_path = os.path.join(path_run_dir, model_checkpoint_file)
        
    else:
        
        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt).start_neptune_run()
        
        # Upload configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt, 
                                        data_item=opt.imagen,
                                        neptune_run_save_path='configs')
        
        # Make results folder with timestamp for samples
        path_run_dir = opt.imagen['validation']['PATH_TRAINING_SAMPLE'] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" + f"_u{opt.imagen['trainer']['unet_number']}"
        path_run_dir = os.path.join(opt.base['PATH_BASE_DIR'], path_run_dir)
        os.mkdir(path_run_dir)
        
        #
        model_checkpoint_path = os.path.join(path_run_dir, model_checkpoint_file)
            
    opt.logger.info(f'Model Checkpoint: {model_checkpoint_path}')
    
    # Specify model save path
    model_save_path=os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['trainer']['PATH_MODEL_SAVE'])
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')
    
    for epoch in tqdm(iterable=range(1, opt.imagen['trainer']['max_epochs'] + 1), disable=tqdm_disable):

        # Training with text_embeds
        loss = imagen_model.trainer.train_step(unet_number=unet_number)
        
        #
        if opt.imagen['trainer']['early_stopping']['usage']:
            imagen_model.loss_queue.push(loss)

            if imagen_model.loss_queue.stop:
                opt.logger.info('Stop training early')
                break

        # validation
        if not (epoch % opt.imagen['validation']['interval']['valid_loss']):

            #
            valid_loss = imagen_model.trainer.valid_step(unet_number=unet_number)
            opt.logger.debug(f'Epoch validation loss-unet{unet_number}: {valid_loss}')
            
            if opt.imagen['trainer']['param_tuning']:
                
                #
                if 'fid_result' in locals():
                    session.report({"fid": fid_result, "loss": loss, "valid_loss": valid_loss})  # Send the scores to Tune.
                else:
                    session.report({"loss": loss, "valid_loss": valid_loss})  # Send the scores to Tune.
                
                opt.logger.info('report succesful')
                    
            else:
                
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                           data_item=valid_loss,
                                           neptune_run_save_path=f"val/loss_unet-{unet_number}")

        # is_main makes sure this can run in distributed
        if not (epoch % opt.imagen['validation']['interval']['validate_model']) and imagen_model.trainer.is_main:
    
            #
            fid_result = test_text2images(opt=opt,
                                sample_dataset=sample_dataset,
                                imagen_model=imagen_model,
                                unet_number=unet_number,
                                sample_quantity=opt.imagen['validation']['sample_quantity'],
                                save_samples=opt.imagen['validation']['display_samples'],
                                sample_folder=path_run_dir,
                                embed_shape=sample_dataset.__getitem__(index=0)[1].size(),
                                epoch=epoch,
                                seed=opt.imagen['validation']['sample_seed'],
                                max_sampling_batch_size=100)
            
            #
            if not opt.imagen['trainer']['param_tuning']:
                
                # Save model checkpoint
                imagen_model.trainer.save(model_checkpoint_path)
                
                # Upload model checkpoint
                neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_checkpoint_path,
                                  neptune_run_save_path='model')
                
                # Upload epoch loss to neptune_ai
                neptune_ai.log_neptune_run(
                    opt=opt, data_item=fid_result, neptune_run_save_path=f"train/fid-{unet_number}")

        #
        if not opt.imagen['trainer']['param_tuning']:
    
            # Upload epoch loss to neptune_ai
            neptune_ai.log_neptune_run(
                opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_number}")


    opt.logger.info('Imagen Training Finished')

    # Save final model
    imagen_model.trainer.save(model_save_path)
    
    #
    neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_save_path,
                                  neptune_run_save_path='model')
    
    #
    neptune_ai.stop_neptune_run(opt=opt)
    
    
def main():
    train_imagen()


if __name__ == "__main__":
    main()
    