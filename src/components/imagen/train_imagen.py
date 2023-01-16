import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from imagen_pytorch import Imagen

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI

from src.components.imagen.build_imagen import Imagen_Model
from src.components.data_manager.CholecT45.imagen_dataset import CholecT45ImagenDataset


def train_imagen(opt: Opt, imagen_model: Imagen):

    # Start run with neptune docs
    neptune_ai = Neptune_AI(opt=opt)
    
    # instantiate train_dataset and train_dataloader
    train_dataset = CholecT45ImagenDataset(opt=opt)
    train_generator = DataLoader(dataset=train_dataset, 
                                  batch_size=opt.imagen['trainer']['batch_size'], 
                                  shuffle=opt.imagen['trainer']['shuffle']
                                )
    opt.imagen['dataset']['data_size'] = train_generator.__len__() * train_generator.batch_size
    
    imagen_model.trainer.add_train_dataloader(dl=train_generator)
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')
    
    # Upload configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                     data_item=opt.imagen,
                                     neptune_run_save_path='configs')
    
    # Specify model save path
    model_save_path=opt.imagen['trainer']['PATH_MODEL_SAVE']
            
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

        for unet_k in [2]:
            for epoch in tqdm(range(opt.imagen['trainer']['max_epochs'])):

                # Training with text_embeds
                loss = imagen_model.trainer.train_step(unet_number=unet_k)

                '''if not (epoch % 1000):
                    valid_loss = imagen_model.trainer.valid_step(unet_number = unet_k, max_batch_size = 4)
                    opt.logger.debug(f'epoch validation loss-unet{unet_k}: {valid_loss}')'''

                # is_main makes sure this can run in distributed
                if not (epoch % 1000) and imagen_model.trainer.is_main:
                    images = imagen_model.trainer.sample(texts=['grasper grasp gallbladder'],
                                                         batch_size=1,
                                                         return_pil_images=True,
                                                         stop_at_unet_number=unet_k)  # returns List[Image]
                    images[0].save(f'src/assets/sample-{epoch}.png')

                # Upload epoch loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_k}")
                
                print(imagen_model.trainer.steps)
        
            imagen_model.trainer.save(model_save_path)
            
    opt.logger.info('Imagen trained')
    
    # Save final model
    imagen_model.trainer.save(model_save_path)
    
    neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=model_save_path,
                                  neptune_run_save_path='model')
    neptune_ai.stop_neptune_run(opt=opt)
    

if __name__ == "__main__":
    opt = Opt()
    
    imagen_model = Imagen_Model(opt=opt)
        
    train_imagen(opt=opt, imagen_model=imagen_model)
    