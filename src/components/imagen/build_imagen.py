import sys
import os
sys.path.append(os.path.abspath(os.curdir))

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from src.components.utils.opt.build_opt import Opt


class Imagen_Model():

    def __init__(self, opt):
        self.unet1, self.unet2 = _get_unets_(opt=opt)
        self.imagen = _get_imagen_(opt=opt, unet1=self.unet1, unet2=self.unet2)
        self.trainer = _get_trainer_(opt=opt, imagen=self.imagen)


def _get_unets_(opt: Opt):

    # unets for imagen
    unet1 = Unet(**opt.imagen['unet1'])
    unet2 = Unet(**opt.imagen['unet2'])

    return unet1, unet2


def _get_imagen_(opt: Opt, unet1: Unet, unet2: Unet):

    # imagen, which contains the unets above (base unet and super resoluting ones)
    imagen = Imagen(
        unets=(unet1, unet2),
        image_sizes=opt.imagen['imagen']['image_sizes'],
        timesteps=opt.imagen['imagen']['timesteps'],
        cond_drop_prob=opt.imagen['imagen']['cond_drop_prob']
    )

    opt.logger.debug('imagen built')

    return imagen


def _get_trainer_(opt: Opt, imagen: Imagen):

    trainer = ImagenTrainer(imagen=imagen,
                            # whether to split the validation dataset from the training
                            split_valid_from_train=opt.imagen['trainer']['split_valid_from_train'],
                            dl_tuple_output_keywords_names = opt.imagen['dataset']['dl_tuple_output_keywords_names']
                            )

    return trainer


if __name__ == "__main__":
    opt = Opt()
    imagen = Imagen_Model(opt=opt)
