import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import torch

from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, load_imagen_from_checkpoint

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.utils.decorators import check_text_encoder, model_starter


def _get_unets_(opt: Opt):

    # unets for imagen
    unet1 = Unet(**opt.imagen['unet1'])
    unet2 = Unet(**opt.imagen['unet2'])

    return unet1, unet2


class Imagen_Model():

    def __init__(self, opt: Opt, validation=False):
        
        self.OPT_IMAGEN_MODEL = dict(**opt.imagen['imagen'])
        self.OPT_IMAGEN_TRAINER = dict(**opt.imagen['trainer'])
        self.OPT_IMAGEN_UNETS = [dict(**opt.imagen['unet1']),
                                  dict(**opt.imagen['unet2'])]
        
        self.unet1, self.unet2 = _get_unets_(opt=opt)
        self.validation = validation
        
        self._set_imagen_()
        self._set_trainer_()

    @model_starter
    @check_text_encoder
    def _set_imagen_(self):
        
        if self.validation:
            
            self.imagen = load_imagen_from_checkpoint(self.OPT_IMAGEN_MODEL['PATH_MODEL_VALIDATION'])
        
        else:
            
            if self.OPT_IMAGEN_TRAINER['use_existing_model'] and glob.glob(self.OPT_IMAGEN_TRAINER['PATH_MODEL_CHECKPOINT']):
                
                self.imagen = load_imagen_from_checkpoint(self.OPT_IMAGEN_TRAINER['PATH_MODEL_SAVE'])
            
            else:

                # imagen-config, which contains the unets above (base unet and super resoluting ones)
                # the config class can be safed and loaded afterwards
                self.imagen = ImagenConfig(unets=self.OPT_IMAGEN_UNETS,
                                      **self.OPT_IMAGEN_MODEL
                                      ).create()

        if torch.cuda.is_available():
            self.imagen = self.imagen.cuda()


    def _set_trainer_(self):

        self.trainer = ImagenTrainer(imagen=self.imagen,
                                # whether to split the validation dataset from the training
                                split_valid_from_train=self.OPT_IMAGEN_TRAINER['split_valid_from_train'],
                                dl_tuple_output_keywords_names = self.OPT_IMAGEN_TRAINER['dl_tuple_output_keywords_names']
                                )
            
        if torch.cuda.is_available():
            self.trainer = self.trainer.cuda()
    

def main():
    opt = Opt()
    
    imagen_model = Imagen_Model(opt=opt)
    opt.logger.debug(f'Imagen Model: {imagen_model.trainer}')


if __name__ == "__main__":
    main()
