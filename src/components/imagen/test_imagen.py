import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np

from datetime import datetime

from src.components.data_manager.CholecT45.imagen_dataset import CholecT45ImagenDataset
from src.components.utils.opt.build_opt import Opt
from src.components.imagen.build_imagen import Imagen_Model
from src.components.data_manager.CholecT45.triplet_coding import get_single_frame_triplet_decoding, get_single_frame_triplet_encoding


def sample_text2images(opt: Opt, sample_index: int, unet_number: int):
    
    imagen_model= Imagen_Model(opt=opt)
    
    # Make results test folder with timestamp
    test_sample_folder = opt.imagen['testing']['PATH_TEST_SAMPLE'] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    os.mkdir(test_sample_folder)

    ds = CholecT45ImagenDataset(opt=opt)
    sample_text = ds.df_triplets['triplet_text'].values[sample_index]
    sample_image, sample_text_embeds = ds.__getitem__(sample_index)

    # sample an image based on the text embeddings from the cascading ddpm
    images = imagen_model.trainer.sample(text_embeds=sample_text_embeds[None,:],
                                         return_pil_images=True,
                                         stop_at_unet_number=unet_number)  # returns List[Image]
    
    images[0].save(f"{test_sample_folder}/sample-{sample_text.strip().replace(',','_')}.png")


def main():
    opt=Opt()
    
    FRAME_INDICES=opt.imagen['testing']['frame_indices']
    UNET_NUMBER=opt.imagen['testing']['unet_number']

    for i in FRAME_INDICES:
        sample_text2images(opt=opt, sample_index=i, unet_number=UNET_NUMBER)
        
        
if __name__ == "__main__":
    main()

