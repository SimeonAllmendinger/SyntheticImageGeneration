import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import ray
import pandas as pd
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from imagen_pytorch import Imagen
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.model.build_imagen import Imagen_Model
from src.components.imagen.data_manager.imagen_dataset import CholecT45ImagenDataset, CholecSeg8kImagenDataset, ConcatImagenDataset


def train_imagen(tune_config=None):
    
    #
    opt = Opt()
    
    #
    if tune_config:
        
        opt.imagen['imagen'] = tune_config
        
        #
        opt.imagen['trainer']['param_tuning'] = True
        tqdm_disable=True
        
        opt.logger.warning('Override configs with tune params')

    # print(opt.imagen['imagen'])
    # Build an imagen model
    imagen_model = Imagen_Model(opt=opt)
    
    # instantiate train_dataset and val_dataset
    if opt.imagen['data']['dataset'] == 'CholecT45':
        imagen_dataset = CholecT45ImagenDataset(opt=opt)
    
    elif opt.imagen['data']['dataset'] == 'CholecSeg8k':
        imagen_dataset = CholecSeg8kImagenDataset(opt=opt)
    
    elif opt.imagen['data']['dataset'] == 'Both':
        imagen_dataset = ConcatImagenDataset(opt=opt)
    
    train_valid_split=[opt.imagen['trainer']['train_split'], opt.imagen['trainer']['valid_split']]
    train_dataset, valid_dataset = random_split(dataset=imagen_dataset, lengths=train_valid_split)
    
    # instantiate train_dataloader and val_dataloader
    train_generator = DataLoader(dataset=train_dataset, 
                                  batch_size=opt.imagen['trainer']['batch_size'], 
                                  shuffle=opt.imagen['trainer']['shuffle']
                                )
    valid_generator = DataLoader(dataset=valid_dataset, 
                                  batch_size=opt.imagen['trainer']['batch_size'], 
                                  shuffle=opt.imagen['trainer']['shuffle']
                                )
    
    # Add data size (length of all elements) to imagen config
    opt.imagen['trainer']['train_size'] = train_dataset.__len__()
    opt.imagen['validation']['valid_size'] = valid_dataset.__len__()
    
    # Add dataloaders to imagen trainer (training, validation)
    imagen_model.trainer.add_train_dataloader(dl=train_generator)
    imagen_model.trainer.add_valid_dataloader(dl=valid_generator)
    
    if not opt.imagen['trainer']['param_tuning']:
        
        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt)
        
        # Upload configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt, 
                                        data_item=opt.imagen,
                                        neptune_run_save_path='configs')
    
    # Specify model save and checkpoint path
    model_save_path=os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['trainer']['PATH_MODEL_SAVE'])
    model_checkpoint_path=os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['trainer']['PATH_MODEL_CHECKPOINT'])
    
    # Make results folder with timestamp for samples
    path_run_dir = opt.imagen['validation']['PATH_TRAINING_SAMPLE'] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" + f"_u{opt.imagen['trainer']['unet_number']}"
    path_run_dir = os.path.join(opt.base['PATH_BASE_DIR'], path_run_dir)
    os.mkdir(path_run_dir)
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')

    unet_number=opt.imagen['trainer']['unet_number']
    
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
        if not (epoch % 500):

            #
            valid_loss = imagen_model.trainer.valid_step(
                unet_number=unet_number)

            #
            model_checkpoint = path_run_dir + '/' + model_checkpoint_path.replace('.pt', f'-u{unet_number}.pt')
            imagen_model.trainer.save(model_checkpoint)

            #
            opt.logger.debug(
                f'Epoch validation loss-unet{unet_number}: {valid_loss}')

            if opt.imagen['trainer']['param_tuning']:

                #
                checkpoint = Checkpoint.from_directory(model_checkpoint)
                session.report({"loss": loss,
                                "valid_loss": valid_loss},
                               checkpoint=checkpoint)

            else:

                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(
                    opt=opt, data_item=valid_loss, neptune_run_save_path=f"val/loss_unet-{unet_number}")

                #
                neptune_ai.upload_neptune_run(opt=opt,
                                              data_item=model_checkpoint,
                                              neptune_run_save_path='model')

        # is_main makes sure this can run in distributed
        if not (epoch % 1000) and imagen_model.trainer.is_main and opt.imagen['trainer']['display_samples']:
           
            # Get random text sample
            sample_index = np.random.randint(low=0,high=opt.imagen['validation']['valid_size'])
            
            sample_image, sample_text_embeds = valid_dataset.__getitem__(sample_index)
            
            if opt.imagen['data']['Cholec80']['use_phase_labels']:
                triplet = imagen_dataset.df_train['TEXT PROMPT'].values[sample_index]
                phase_label = imagen_dataset.df_train['PHASE LABEL'].values[sample_index]
                
                sample_text = triplet + ' in ' + phase_label
                
            else:
                # Get triplet text
                sample_text = imagen_dataset.df_train['TEXT PROMPT'].values[sample_index]

            # returns List[Image]
            images = imagen_model.trainer.sample(text_embeds=sample_text_embeds[None,:],
                                                    return_pil_images=True,
                                                    stop_at_unet_number=unet_number)  
            
            images[0].save(path_run_dir + f'/e{epoch}-u{unet_number}-{sample_text}.png')

        #
        if opt.imagen['trainer']['param_tuning']:
            
            session.report({"loss": loss})  # Send the score to Tune.
                    
        else:
    
            # Upload epoch loss to neptune_ai
            neptune_ai.log_neptune_run(
                opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_number}")


    opt.logger.info('Imagen Training Finished')

    # Save final model
    imagen_model.trainer.save(model_save_path)
    
    neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_save_path,
                                  neptune_run_save_path='model')
    
    neptune_ai.stop_neptune_run(opt=opt)
    
    
def main():
    train_imagen()


if __name__ == "__main__":
    main()
    