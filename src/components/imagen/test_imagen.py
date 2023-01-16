import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import matplotlib.pyplot as plt
from imagen_pytorch import ImagenTrainer
from src.components.utils.opt.build_opt import Opt
from src.components.imagen.build_imagen import Imagen_Model
from src.components.data_manager.CholecT45.triplet_coding import get_single_frame_triplet_decoding, get_single_frame_triplet_encoding, _load_text_data


def sample_text2images(opt: Opt, sample_text :str):
    
    imagen_model= Imagen_Model(opt=opt)
    
    # sample an image based on the text embeddings from the cascading ddpm
    images = imagen_model.trainer.sample(texts=list(sample_text),
                                         batch_size=1,
                                         return_pil_images=True)
    
    images[0].save(f"./results/sample-{sample_text.strip().replace(',','_')}.png")
    
    '''for i in range(images.shape[0]):
        plt.imshow(images[i].permute(1, 2, 0))
        plt.show()'''

if __name__ == "__main__":
    opt=Opt()
    
    VIDEO_N=31
    FRAME_N=[1,100,200,600]
    
    triplets_dict = _load_text_data(
        opt=opt, path=opt.imagen['dataset']['PATH_DICT_DIR'] + 'triplet.txt')

    for i in FRAME_N:
        frame_triplet_encoding = get_single_frame_triplet_encoding(
            opt=opt, video_n=VIDEO_N, frame_n=i)
        text = get_single_frame_triplet_decoding(opt=opt,
                                                 frame_triplet_encoding=frame_triplet_encoding,
                                                 triplets_dict=triplets_dict)

        sample_text2images(opt=opt, sample_text=text)
