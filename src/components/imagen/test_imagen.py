import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import matplotlib.pyplot as plt
from imagen_pytorch import ImagenTrainer
from src.components.utils.opt.build_opt import Opt
from src.components.imagen.build_imagen import _get_imagen_
from components.data_manager.CholecT45.triplet_coding import get_frame_triplet_decoding, get_frame_triplet_encoding


def sample_text2images(opt: Opt, sample_texts :list()):
    
    imagen=_get_imagen_(opt=opt)
    trainer = ImagenTrainer(imagen = imagen)
    imagen_trained = trainer.load('./path/to/checkpoint.pt')
    
    # sample an image based on the text embeddings from the cascading ddpm
    images = imagen_trained.sample(texts=sample_texts, 
                                   cond_scale = 3.)

    images[0].save(f'./results/sample-{i // 100}.png')
    
    for i in images.shape[0]:
        plt.imshow(images[i])
        plt.show()
    
    opt.logger.debug('imagen built')
    

if __name__ == "__main__":
    opt=Opt()
    
    VIDEO_N=1
    FRAME_N=[1]
    
    texts=list()
    
    for i in FRAME_N:
        frame_triplet_encoding = get_frame_triplet_encoding(video_n=VIDEO_N, frame_n=i, logger=master_logger)
        texts.append(get_frame_triplet_decoding(frame_triplet_encoding=frame_triplet_encoding, logger=master_logger))
    
    sample_text2images(texts=texts, logger=master_logger)