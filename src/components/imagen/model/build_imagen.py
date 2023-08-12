import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import torch

from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.utils.decorators import check_text_encoder, model_starter
from src.components.utils.training.early_stopping import EarlyStopping


def _get_unets_(opt: Opt, elucidated: bool):

    # unets for elucidated imagen or basic imagen model
    if elucidated:
        unet1 = Unet(**opt.elucidated_imagen['unet1'])
        unet2 = Unet(**opt.elucidated_imagen['unet2'])
    else:
        unet1 = Unet(**opt.imagen['unet1'])
        unet2 = Unet(**opt.imagen['unet2'])

    return unet1, unet2


class Imagen_Model():

    def __init__(self, opt: Opt, testing=False):

        assert opt.conductor['model']['model_type'] == 'ElucidatedImagen' or opt.conductor['model']['model_type'] == 'Imagen', 'Please choose existing imagen model: Imagen or ElucidatedImagen'
        
        if opt.conductor['model']['model_type'] == 'ElucidatedImagen':
            self.is_elucidated = True
        elif opt.conductor['model']['model_type'] == 'Imagen':
            self.is_elucidated = False
        
        self.testing = testing
        
        self.unet1, self.unet2 = _get_unets_(opt=opt, elucidated=self.is_elucidated)
                
        self._set_imagen_(opt=opt)
        self._set_trainer_(opt=opt)
        
        if opt.conductor['trainer']['early_stopping']['usage']:
            self.loss_queue = EarlyStopping(opt_early_stopping=opt.conductor['trainer']['early_stopping'])
        

    @model_starter
    @check_text_encoder
    def _set_imagen_(self, opt: Opt):
        
        path_model_load = os.path.join(opt.base['PATH_BASE_DIR'],opt.conductor['trainer']['PATH_MODEL_LOAD'])
            
        if self.testing:
            
            #
            test_model_save = os.path.join(opt.base['PATH_BASE_DIR'],opt.conductor['testing']['PATH_MODEL_TESTING'])
            self.imagen_model = load_imagen_from_checkpoint(test_model_save)
            
        elif (opt.conductor['trainer']['use_existing_model'] and glob.glob(path_model_load)):
            
            #
            self.imagen_model = load_imagen_from_checkpoint(path_model_load)

        else:
            
            if self.is_elucidated:
                # elucidated imagen-config, which contains the unets above (base unet and super resoluting ones)
                # the config class can be safed and loaded afterwards
                self.imagen_model = ElucidatedImagenConfig(unets=[dict(**opt.elucidated_imagen['unet1']),
                                                            dict(**opt.elucidated_imagen['unet2'])],
                                                     **opt.elucidated_imagen['elucidated_imagen']
                                                     ).create()
            else:
                # imagen-config, which contains the unets above (base unet and super resoluting ones)
                # the config class can be safed and loaded afterwards
                self.imagen_model = ImagenConfig(unets=[dict(**opt.imagen['unet1']),
                                                  dict(**opt.imagen['unet2'])],
                                           **opt.imagen['imagen']
                                           ).create()

        if torch.cuda.is_available() and not opt.conductor['trainer']['multi_gpu']:
            self.imagen_model = self.imagen_model.cuda()


    def _set_trainer_(self, opt: Opt):

        self.trainer = ImagenTrainer(imagen=self.imagen_model,
                                # whether to split the validation dataset from the training
                                split_valid_from_train=opt.conductor['trainer']['split_valid_from_train'],
                                dl_tuple_output_keywords_names = opt.conductor['trainer']['dl_tuple_output_keywords_names']
                                )
            
        if torch.cuda.is_available() and not opt.conductor['trainer']['multi_gpu']:
            self.trainer = self.trainer.cuda()
    

def main():
    opt = Opt()
    
    imagen_model = Imagen_Model(opt=opt)
    opt.logger.debug(f'Imagen Model: {imagen_model.trainer}')


if __name__ == "__main__":
    main()
