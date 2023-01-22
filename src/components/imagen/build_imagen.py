import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
from imagen_pytorch import Unet, Imagen, ImagenTrainer, ImagenConfig, load_imagen_from_checkpoint
from src.components.utils.opt.build_opt import Opt


class Imagen_Model():

    def __init__(self, opt, validation=False):
        self.unet1, self.unet2 = _get_unets_(opt=opt)
        self.imagen = _get_imagen_(opt=opt, validation=validation)
        self.trainer = _get_trainer_(opt=opt, imagen=self.imagen)


def _get_unets_(opt: Opt):

    # unets for imagen
    unet1 = Unet(**opt.imagen['unet1'])
    unet2 = Unet(**opt.imagen['unet2'])

    return unet1, unet2


def _get_imagen_(opt: Opt, validation: bool):
    
    if validation:
        
        opt.logger.info('Load Imagen model from Checkpoint for Validation')
        imagen = load_imagen_from_checkpoint(opt.imagen['validation']['PATH_MODEL_VALIDATION'])
    
    else:
        if opt.imagen['trainer']['use_existing_model'] or glob.glob(opt.imagen['trainer']['PATH_MODEL_CHECKPOINT']):
            
            opt.logger.info('Load Imagen model from Checkpoint for further Training')
            imagen = load_imagen_from_checkpoint(opt.imagen['trainer']['PATH_MODEL_SAVE'])
        
        else:
            
            opt.logger.info('Create new Imagen model')
            
            if opt.imagen['imagen']['t5_text_embedding']:
                assert opt.imagen['imagen']['text_embed_dim'] == 768, "Text embed dim must be changed to 768"
                assert opt.imagen['imagen']['text_encoder_name'] == 'google/t5-v1_1-base', "Text encoder name must be changed to 'google/t5-v1_1-base'"
            else:
                assert opt.imagen['imagen']['text_embed_dim'] == 29, "Text embed dim must be changed to 29"
                assert opt.imagen['imagen']['text_encoder_name'] == 'Own_Encoder', "Text encoder name must be changed to 'Own_Encoder'"
                
            
            # imagen-config, which contains the unets above (base unet and super resoluting ones)
            # the config class can be safed and loaded afterwards
            imagen = ImagenConfig(unets=[dict(**opt.imagen['unet1']),
                                         dict(**opt.imagen['unet2'])],
                                  **opt.imagen['imagen']
                                  ).create()   
        
    if opt.pytorch_cuda.available:
        imagen = imagen.cuda()
        
    opt.logger.debug('Imagen Model built')

    return imagen


def _get_trainer_(opt: Opt, imagen: Imagen):

    trainer = ImagenTrainer(imagen=imagen,
                            # whether to split the validation dataset from the training
                            split_valid_from_train=opt.imagen['trainer']['split_valid_from_train'],
                            dl_tuple_output_keywords_names = opt.imagen['trainer']['dl_tuple_output_keywords_names']
                            )
        
    if opt.pytorch_cuda.available:
        trainer = trainer.cuda()

    return trainer


def main():
    opt = Opt()
    imagen_model = Imagen_Model(opt=opt)
    opt.logger.debug(f'imagen trainer attributes: {imagen_model.imagen}')


if __name__ == "__main__":
    main()
