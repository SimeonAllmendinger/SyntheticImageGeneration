import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from imagen_pytorch import Imagen

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.build_imagen import Imagen_Model
from src.components.data_manager.CholecT45.imagen_dataset import CholecT45ImagenDataset


def train_imagen(opt: Opt, imagen_model: Imagen, sample_text: str):

    # Start run with neptune docs
    neptune_ai = Neptune_AI(opt=opt)
    
    # instantiate train_dataset and val_dataset
    train_valid_split=[opt.imagen['trainer']['train_split'], opt.imagen['trainer']['valid_split']]
    imagen_dataset = CholecT45ImagenDataset(opt=opt)
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
    opt.imagen['dataset']['train_size'] = train_generator.__len__() * train_generator.batch_size
    opt.imagen['dataset']['valid_size'] = valid_generator.__len__() * valid_generator.batch_size
    
    # Add dataloaders to imagen trainer (training, validation)
    imagen_model.trainer.add_train_dataloader(dl=train_generator)
    imagen_model.trainer.add_valid_dataloader(dl=valid_generator)
    
    # Upload configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                     data_item=opt.imagen,
                                     neptune_run_save_path='configs')
    
    # Specify model save and checkpoint path
    model_save_path=opt.imagen['trainer']['PATH_MODEL_SAVE']
    model_checkpoint_path=opt.imagen['trainer']['PATH_MODEL_CHECKPOINT']
    
    # Make results folder wiith timestamp
    training_sample_folder = opt.imagen['validation']['PATH_TRAINING_SAMPLE'] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    os.mkdir(training_sample_folder)
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')
            
    if not opt.imagen['dataset']['t5_text_embedding']:
    
        for epoch in range(opt.imagen['trainer']['max_epochs']):
            for unet_k in [1,2]: 
                
                # Training with text
                opt.logger.info(f'Epoch: {epoch+1} | Unet {unet_k}')
                
                for image_batch, triplet_batch in tqdm(train_generator):
                    loss = imagen_model.imagen(images=image_batch,
                                               texts=list(triplet_batch),
                                               unet_number=unet_k)
                    loss.backward()
            
                opt.logger.debug(f'epoch loss-unet{unet_k}: {loss}')
            
                # Upload epoch loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_k}")
            
            imagen_model.trainer.save(model_save_path.replace('.pt', f'_u{unet_k}_e{epoch+1}.pt'))
                    
    else:

        unet_number=opt.imagen['trainer']['unet_number']
        
        for epoch in tqdm(range(opt.imagen['trainer']['max_epochs'])):

            # Training with text_embeds
            loss = imagen_model.trainer.train_step(unet_number=unet_number)

            if not (epoch % 100):
                valid_loss = imagen_model.trainer.valid_step(unet_number = unet_number)
                opt.logger.debug(f'epoch validation loss-unet{unet_number}: {valid_loss}')
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(
                    opt=opt, data_item=valid_loss, neptune_run_save_path=f"val/loss_unet-{unet_number}")

            # is_main makes sure this can run in distributed
            if not (epoch % 1000) and imagen_model.trainer.is_main and epoch != 0:
                
                images = imagen_model.trainer.sample(texts=list(sample_text),
                                                     batch_size=1,
                                                     return_pil_images=True,
                                                     stop_at_unet_number=unet_number)  # returns List[Image]
                images[0].save(training_sample_folder + f'/{sample_text}-e{epoch}-u{unet_number}.png')

                neptune_ai.upload_neptune_run(opt=opt,
                                              data_item=model_checkpoint_path,
                                              neptune_run_save_path='model')

                imagen_model.trainer.save(model_checkpoint_path.replace('.pt',f'_e{epoch}-u{unet_number}.pt'))

            # Upload epoch loss to neptune_ai
            neptune_ai.log_neptune_run(
                opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_number}")

    opt.logger.info('Imagen trained')

    # Save final model
    imagen_model.trainer.save(model_save_path)
    
    neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_save_path,
                                  neptune_run_save_path='model')
    neptune_ai.stop_neptune_run(opt=opt)
    
    
def main():
    opt = Opt()
    
    imagen_model = Imagen_Model(opt=opt)
    
    # Get random text sample
    k = np.random.randint(low=0,high=551)
    sample_text = pd.read_json(opt.imagen['dataset']['PATH_TRIPLETS_DF_FILE'])['triplet_text'].unique().tolist()[k]
        
    train_imagen(opt=opt, imagen_model=imagen_model, sample_text=sample_text)


if __name__ == "__main__":
    main()
    