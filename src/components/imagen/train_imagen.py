import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
from torch.utils.data import DataLoader

from imagen_pytorch import Imagen, load_imagen_from_checkpoint

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
                                  shuffle=opt.imagen['trainer']['shuffle'])
    
    # feed images into imagen, training each unet in the cascade
    opt.logger.info('Start training of Imagen Diffusion Model')
    
    # Upload configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                     data_item=opt.imagen,
                                     neptune_run_save_path='configs')
    
    for epoch in range(opt.imagen['trainer']['max_epochs']):
        for unet_k in (1, 2):
            # Training
            for image_batch, triplet_batch in train_generator:
                if not opt.imagen['dataset']['text_embedding']:
                    loss = imagen_model.imagen(images=image_batch,
                                               texts=list(triplet_batch),
                                               unet_number=unet_k)
                else:
                    loss = imagen_model.imagen(images=image_batch,
                                               text_embeds=triplet_batch,
                                               unet_number=unet_k)
                loss.backward()
                opt.logger.debug(f'batch loss: {loss}')
                
                # Upload epoch loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt, data_item=loss, neptune_run_save_path=f"train/loss_unet-{unet_k}")
            
    opt.logger.info('Imagen trained')
    
    save_path=f'./src/assets/imagen/models/imagen_checkpoint.pt'
    imagen_model.trainer.save(save_path)
    
    neptune_ai.upload_neptune_run(opt=opt, 
                                  data_item=save_path,
                                  neptune_run_save_path='model')
    neptune_ai.stop_neptune_run(opt=opt)
    

if __name__ == "__main__":
    opt = Opt()
    if not glob.glob('./src/assets/imagen/models/*.pt'):
        imagen_model = Imagen_Model(opt=opt)
    else:
        opt.logger.info('Load Imagen from Checkpoint')
        imagen_model = load_imagen_from_checkpoint('./src/assets/imagen/models/imagen_checkpoint.pt')
        
    train_imagen(opt=opt, imagen_model=imagen_model)
    