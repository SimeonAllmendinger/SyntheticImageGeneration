import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import ray
import yaml
import argparse
import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm
from ray.air import session
from accelerate.utils import find_executable_batch_size

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.model.build_imagen import Imagen_Model
from src.components.data_manager.dataset_handler import get_train_valid_ds, get_train_valid_dl

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to uerib@student.kit.edu')

parser.add_argument('--path_data_dir',
                    default='/home/stud01/SyntheticImageGeneration/',
                    help='PATH to data directory')

#@find_executable_batch_size(starting_batch_size=Opt().conductor['trainer']['batch_size'])
def train_imagen(tune_config=None, reporter=None):
    
    #
    opt = Opt()
    
    tqdm_disable=False
    
    if tune_config:
        
        #
        tqdm_disable=True
        
        #
        opt.imagen['imagen'] = tune_config['imagen']
        #opt.elucidated_imagen['elucidated_imagen'] = tune_config['elucidated_imagen']
        
        #
        opt.conductor['trainer']['param_tuning'] = True
        opt.conductor['trainer']['early_stopping']['usage'] = False
        
        #
        opt.datasets['data']['dataset'] = tune_config['data']['dataset']
        opt.datasets['data']['use_existing_data_files'] = tune_config['data']['use_existing_data_files']
        opt.datasets['data']['use_phase_labels'] = tune_config['data']['use_phase_labels']
        
        opt.logger.warning('Override configs with tune params')

    # Build an imagen model
    imagen_model = Imagen_Model(opt=opt)
    unet_number=opt.conductor['trainer']['unet_number']
    
    # 
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt)
    sample_dataset = get_train_valid_ds(opt=opt, testing=True, return_text=True)

    # instantiate train_dataloader and val_dataloader
    train_generator, valid_generator = get_train_valid_dl(opt=opt, 
                                                          train_dataset=train_dataset,
                                                          valid_dataset=valid_dataset)
    sample_dataloader = get_train_valid_dl(opt=opt, train_dataset=sample_dataset)
    
    # Add data size (length of all elements) to imagen config
    opt.conductor['trainer']['train_size'] = train_dataset.__len__()
    opt.conductor['validation']['valid_size'] = valid_dataset.__len__()
    
    # Add dataloaders to imagen trainer (training, validation)
    imagen_model.trainer.add_train_dataloader(dl=train_generator)
    imagen_model.trainer.add_valid_dataloader(dl=valid_generator)
    
    model_checkpoint_file = opt.conductor['trainer']['PATH_MODEL_CHECKPOINT'].replace('.pt', f'-u{unet_number}.pt')
    
    path_run_dir = ''
    
    if opt.conductor['trainer']['param_tuning']:
        
        #
        path_run_dir = reporter.logdir
        model_checkpoint_path = os.path.join(path_run_dir, model_checkpoint_file)
        
    else:
        
        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt)
        neptune_ai.start_neptune_run(opt)
        
        if opt.conductor['model']['model_type'] == 'Imagen':
            # Upload model configs to neptune_ai
            neptune_ai.add_param_neptune_run(opt=opt, 
                                            data_item=opt.imagen,
                                            neptune_run_save_path='model_configs')
        
        elif opt.conductor['model']['model_type'] == 'ElucidatedImagen':
            # Upload model configs to neptune_ai
            neptune_ai.add_param_neptune_run(opt=opt, 
                                            data_item=opt.elucidated_imagen,
                                            neptune_run_save_path='model_configs')
        
        # Upload configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt, 
                                        data_item=opt.conductor,
                                        neptune_run_save_path='training_configs')
        
        # Make results folder with timestamp for samples
        path_run_dir = opt.conductor['validation']['PATH_TRAINING_SAMPLE'] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" + f"_u{opt.conductor['trainer']['unet_number']}_" + opt.conductor['model']['model_type']
        path_run_dir = os.path.join('/home/kit/stud/uerib/SyntheticImageGeneration/', path_run_dir)
        os.mkdir(path_run_dir)
        
        #
        model_checkpoint_path = os.path.join(path_run_dir, model_checkpoint_file)
            
    opt.logger.info(f'Model Checkpoint: {model_checkpoint_path}')
    
    # Specify model save path
    model_save_path=os.path.join(opt.base['PATH_BASE_DIR'], opt.conductor['trainer']['PATH_MODEL_SAVE'])
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')
    
    for epoch in tqdm(iterable=range(1, opt.conductor['trainer']['max_epochs'] + 1), disable=tqdm_disable):

        # Training with text_embeds
        loss = imagen_model.trainer.train_step(unet_number=unet_number)
        
        #
        if opt.conductor['trainer']['early_stopping']['usage']:
            imagen_model.loss_queue.push(loss)

            if imagen_model.loss_queue.stop:
                opt.logger.info('Stop training early')
                break

        # validation
        if not (epoch % opt.conductor['validation']['interval']['valid_loss']):

            #
            valid_loss = imagen_model.trainer.valid_step(unet_number=unet_number)
            opt.logger.debug(f'Epoch validation loss-unet{unet_number}: {valid_loss}')
            
            if opt.conductor['trainer']['param_tuning']:
            
                session.report({"loss": loss, "valid_loss": valid_loss})  # Send the scores to Tune.
            
            else:
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                           data_item=valid_loss,
                                           neptune_run_save_path=f"val/loss_unet-{unet_number}")

        # is_main makes sure this can run in distributed
        if not (epoch % opt.conductor['validation']['interval']['validate_model']) and imagen_model.trainer.is_main:
            
            #
            if opt.conductor['validation']['display_samples']:
                n = int(np.ceil(opt.conductor['validation']['sample_quantity'] / opt.conductor['trainer']['batch_size']))
                
                for i in range(n):
                    _, embed_batch, text_batch = next(iter(sample_dataloader))
                    synthetic_images = imagen_model.trainer.sample(text_embeds=embed_batch.cuda(),
                                                                    return_pil_images=True,
                                                                    stop_at_unet_number=unet_number,
                                                                    use_tqdm=not tqdm_disable,
                                                                    cond_scale=opt.conductor['validation']['cond_scale'])
                
                    for j, synthetic_image in enumerate(synthetic_images):
                        image_save_path = os.path.join(path_run_dir, f'e{epoch}-u{unet_number}-{j:05d}-{text_batch[j]}.png')
                        synthetic_image.save(image_save_path)
            
            
            #
            if not opt.conductor['trainer']['param_tuning']:
                
                # Save model checkpoint
                imagen_model.trainer.save(model_checkpoint_path)
                
                # Upload model checkpoint
                neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_checkpoint_path,
                                  neptune_run_save_path='model')

        #
        if not opt.conductor['trainer']['param_tuning']:
    
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
    
    #
    with open('configs/config_datasets.yaml') as f:
        config = yaml.safe_load(f)

    args = parser.parse_args()
    config['PATH_DATA_DIR'] = args.path_data_dir

    with open('configs/config_datasets.yaml', 'w') as f:
        yaml.dump(config, f)
    
    train_imagen()


if __name__ == "__main__":
    main()
    