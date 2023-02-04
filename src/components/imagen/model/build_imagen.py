import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import torch

from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, load_imagen_from_checkpoint

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.utils.decorators import check_text_encoder, model_starter
from src.components.imagen.utils.early_stopping import EarlyStopping


def _get_unets_(opt: Opt):

    # unets for imagen
    unet1 = Unet(**opt.imagen['unet1'])
    unet2 = Unet(**opt.imagen['unet2'])

    return unet1, unet2


class Imagen_Model():

    def __init__(self, opt: Opt):
        
        self.unet1, self.unet2 = _get_unets_(opt=opt)
        
        self._set_imagen_(opt=opt)
        self._set_trainer_(opt=opt)
        
        if opt.imagen['trainer']['early_stopping']['usage']:
            self.loss_queue = EarlyStopping(opt_early_stopping=opt.imagen['trainer']['early_stopping'])
        

    @model_starter
    @check_text_encoder
    def _set_imagen_(self, opt: Opt):
        
        path_model_save = os.path.join(opt.base['PATH_BASE_DIR'],opt.imagen['trainer']['PATH_MODEL_SAVE'])
            
        if opt.imagen['trainer']['use_existing_model'] and glob.glob(path_model_save):
            
            self.imagen = load_imagen_from_checkpoint(path_model_save)

        else:

            # imagen-config, which contains the unets above (base unet and super resoluting ones)
            # the config class can be safed and loaded afterwards
            self.imagen = ImagenConfig(unets=[dict(**opt.imagen['unet1']),
                                                dict(**opt.imagen['unet2'])],
                                        **opt.imagen['imagen']
                                        ).create()

        if torch.cuda.is_available():
            self.imagen = self.imagen.cuda()


    def _set_trainer_(self, opt: Opt):

        self.trainer = ImagenTrainer(imagen=self.imagen,
                                # whether to split the validation dataset from the training
                                split_valid_from_train=opt.imagen['trainer']['split_valid_from_train'],
                                dl_tuple_output_keywords_names = opt.imagen['trainer']['dl_tuple_output_keywords_names']
                                )
            
        if torch.cuda.is_available():
            self.trainer = self.trainer.cuda()
    

def main():
    opt = Opt()
    
    imagen_model = Imagen_Model(opt=opt)
    opt.logger.debug(f'Imagen Model: {imagen_model.trainer}')


if __name__ == "__main__":
    main()
