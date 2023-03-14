import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import numpy as np
import torchvision.transforms as transforms

from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


from src.components.imagen.data_manager.imagen_dataset import CholecT45ImagenDataset
from src.components.utils.opt.build_opt import Opt
from src.components.imagen.model.build_imagen import Imagen_Model
from src.components.imagen.data_manager.imagen_dataset import get_train_valid_ds

def test_text2images(opt: Opt, 
                     sample_dataset, 
                     imagen_model: Imagen_Model, 
                     unet_number: int, 
                     sample_quantity: int,
                     save_samples: bool,
                     sample_folder: str,
                     embed_shape: tuple,
                     epoch: int,
                     seed: int,
                     max_sampling_batch_size=100
                     ):

    #
    fid = FrechetInceptionDistance(**opt.imagen['testing']['FrechetInceptionDistance']).cuda()
    #kid = KernelInceptionDistance().cuda()
    
    # Convert the PIL image to a tensor with data type torch.uint8
    transform = transforms.ToTensor()
    
    #
    sample_images = torch.zeros(sample_quantity,
                                3, # 3 color channels
                                opt.imagen['data']['image_size'],
                                opt.imagen['data']['image_size'])
    
    #
    sample_text_embeds = torch.zeros(sample_quantity,
                                     embed_shape[0],
                                     embed_shape[1])
    
    #
    fid.update((sample_images * 255).to(torch.uint8).cuda(), real=True)
    
    #
    sample_texts = list()
    excluded_indices = set()
    
    #
    np.random.seed(seed)
    
    for k in tqdm(range(sample_quantity)):
        
        #
        sample_index = np.random.randint(low=0, high=sample_dataset.__len__())
        
        while sample_index in excluded_indices:
            sample_index = np.random.randint(low=0, high=sample_dataset.__len__())
        
        excluded_indices.add(sample_index)
        
        #
        sample_image, sample_text_embed, sample_text = sample_dataset.__getitem__(sample_index,
                                                                                   return_text=True)
        sample_images[k, :, :, :] = sample_image
        sample_text_embeds[k, :, :] = sample_text_embed
        sample_texts.append(sample_text)
    
    num_batches = sample_text_embeds.shape[0] // max_sampling_batch_size
    sample_text_embeds_batches = torch.split(sample_text_embeds, max_sampling_batch_size, dim=0)
    
    for i, embed_batch in enumerate(sample_text_embeds_batches):
        
        # sample an image based on the text embeddings from the cascading ddpm
        synthetic_images = imagen_model.trainer.sample(text_embeds=embed_batch,
                                                        return_pil_images=False,
                                                        stop_at_unet_number=unet_number) 
        
        #
        fid.update((synthetic_images * 255).to(torch.uint8).cuda(), real=False)
        
        if save_samples: 
            
            #
            for j, synthetic_image in enumerate(synthetic_images):
                
                # 
                synthetic_image = synthetic_image.cpu().permute(1,2,0)
                synthetic_image = synthetic_image.numpy() * 255
                synthetic_image = Image.fromarray(synthetic_image.astype(np.uint8))
                
                text_index = i*max_sampling_batch_size + j
                sample_save_path = sample_folder + f'/e{epoch}-u{unet_number}-{sample_texts[text_index]}.png'
                synthetic_image.save(sample_save_path)
                
                #
                opt.logger.info(f'Created image for {sample_texts[text_index]} at {sample_save_path}.')
        
        #
        #kid.update(sample_images.to(torch.uint8).cuda(), real=True)
        #kid.update(synthetic_images.to(torch.uint8).cuda()[:-1], real=False)
        
    #
    fid_result = fid.compute()
    #kid_mean, kid_std = kid.compute()
    
    #
    return fid_result#, (kid_mean, kid_std)


def main():
    
    #
    opt=Opt()
    
    #
    sample_quantity=opt.imagen['testing']['sample_quantity']
    unet_number=opt.imagen['testing']['unet_number']
    save_samples=opt.imagen['testing']['save_samples']
    seed=opt.imagen['testing']['sample_seed']
    
    #
    imagen_dataset = get_train_valid_ds(opt=opt, testing=True)
    imagen_model= Imagen_Model(opt=opt, testing=True)
    _, sample_text_embed, _ = imagen_dataset.__getitem__(index=0, 
                                                         return_text=True)

    # Make results test folder with timestamp
    timestamp = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    test_sample_folder = os.path.join(
        opt.base['PATH_BASE_DIR'], opt.imagen['testing']['PATH_TEST_SAMPLE'] + timestamp)
    os.mkdir(test_sample_folder)

    #
    fid_result = test_text2images(opt=opt,
                                  sample_dataset=imagen_dataset,
                                  imagen_model=imagen_model,
                                  unet_number=unet_number,
                                  sample_quantity=sample_quantity,
                                  save_samples=save_samples,
                                  sample_folder=test_sample_folder,
                                  embed_shape=sample_text_embed.size(),
                                  epoch=0,
                                  seed=seed,
                                  max_sampling_batch_size=100
                                  )
    
    #
    opt.logger.info(f'FRECHET INCEPTION DISTANCE (FID): {fid_result}')
    #opt.logger.info(f'KERNEL INCEPTION DISTANCE (FID): {kid_result}')
        
        
if __name__ == "__main__":
    main()

