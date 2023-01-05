import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import matplotlib.pyplot as plt
from imagen_pytorch import ImagenTrainer
from src.components.utils.opt.build_opt import Opt
from src.components.imagen.build_imagen import Imagen_Model
from src.components.data_manager.CholecT45.triplet_coding import get_single_frame_triplet_decoding, get_single_frame_triplet_encoding, _load_text_data


def sample_text2images(opt: Opt, sample_texts :list()):
    
    imagen_model= Imagen_Model(opt=opt)
    trainer = ImagenTrainer(imagen_model.imagen)
    trainer.load('./src/assets/imagen/models/imagen_checkpoint.pt')
    
    # sample an image based on the text embeddings from the cascading ddpm
    images = trainer.sample(texts=sample_texts)
    opt.logger.debug(f'images shape: {images.shape}')
    
    #images[0].save(f'./results/sample-{i // 100}.png')
    
    for i in range(images.shape[0]):
        plt.imshow(images[i].permute(1, 2, 0))
        plt.show()
    
    opt.logger.debug('imagen built')
    

if __name__ == "__main__":
    opt=Opt()
    
    VIDEO_N=1
    FRAME_N=[1,2]
    
    triplets_dict = _load_text_data(opt=opt, path=opt.imagen['dataset']['PATH_DICT_DIR'] + 'triplet.txt')
    texts=list()
    
    for i in FRAME_N:
        frame_triplet_encoding = get_single_frame_triplet_encoding(opt=opt, video_n=VIDEO_N, frame_n=i)
        texts.append(get_single_frame_triplet_decoding(opt=opt, 
                                                       frame_triplet_encoding=frame_triplet_encoding, 
                                                       triplets_dict=triplets_dict))
    
    sample_text2images(opt=opt, sample_texts=texts)