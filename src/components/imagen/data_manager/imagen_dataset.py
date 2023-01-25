import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from imagen_pytorch.data import Dataset
from PIL import Image

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.data_manager.triplet_coding import get_df_triplets
from src.components.imagen.data_manager.triplet_embedding import get_triplet_ohe_embedding, get_triplet_t5_embedding
from src.components.imagen.utils.decorators import check_dataset_name, check_text_encoder


class ImagenDataset(Dataset):
    
    def __init__(self, dataset_name: str, opt: Opt):
        
        #
        self.OPT_DATA = dict(**opt.imagen['data'])
        self.TEXT_ENCODER_NAME = opt.imagen['imagen']['text_encoder_name']
        self.DATASET = dataset_name
        
        #
        self._set_df_triplets_(opt=opt)
        self._set_triplet_embeds_(opt=opt)
        
        #
        super().__init__(folder=self.OPT_DATA[self.DATASET]['PATH_VIDEO_DIR'],
                         image_size=self.OPT_DATA['image_size'])
        
        
    @check_text_encoder
    def _set_triplet_embeds_(self, opt: Opt):
        
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            # Create OHE embedding
            self.triplet_embeds = get_triplet_ohe_embedding(triplet_dict_indices=self.df_triplets['FRAME TRIPLET DICT INDICES'], 
                                                       opt=opt)

        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':

            # Create T5 embedding
            self.triplets_unique_list = self.df_triplets['FRAME TRIPLET TEXT'].unique().tolist()
            self.triplet_embeds = get_triplet_t5_embedding(triplets_unique_list=self.triplets_unique_list, 
                                                      opt=opt)


        opt.logger.debug('Text embedding created')
        opt.logger.debug('Text embedding shape: ' + str(self.triplet_embeds.size()))
    
    
    @check_dataset_name
    def _set_df_triplets_(self, opt: Opt):
        self.df_triplets=get_df_triplets(dataset_name=self.DATASET, opt=opt)
    
    
    def __len__(self):
        return self.df_triplets.shape[0]
    
    
    def __getitem__(self, index):
        
        # Get Image
        path = os.path.join(self.folder, self.df_triplets['FRAME PATH'].values[index])
        image = Image.open(path)
        
        # Get triplet text
        triplet_text = self.df_triplets['FRAME TRIPLET TEXT'].values[index]

        # Transform Image
        image = self.transform(image)
        
        # Get triplet embedding
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            triplet_embedding = self.triplet_embeds[index]
        
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':
            
            index_unique = self.triplets_unique_list.index(triplet_text)
            triplet_embedding = self.triplet_embeds[index_unique]

        # Check gpu availability    
        if torch.cuda.is_available():
            image = image.cuda()
            triplet_embedding = triplet_embedding.cuda()
        
        return image, triplet_embedding


class CholecT45ImagenDataset(ImagenDataset):
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecT45', opt=opt)
        

class CholecSeg8kImagenDataset(ImagenDataset):
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecSeg8k', opt=opt)


def main():
    opt=Opt()

    if opt.imagen['data']['dataset'] == 'CholecT45':
        dataset = CholecT45ImagenDataset(opt=opt)
    elif opt.imagen['data']['dataset'] == 'CholecSeg8k':
        dataset = CholecSeg8kImagenDataset(opt=opt)
    
    triplets_list = pd.read_json(opt.imagen['data'][opt.imagen['data']['dataset']]['PATH_TRIPLETS_DF_FILE'])['FRAME TRIPLET TEXT'].values
    
    # Plot
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    
    for i in range(2):
        # Example image with random index
        index = np.random.randint(low=0, high=200)
        image, embed = dataset.__getitem__(index=index)
        
        opt.logger.info(f'Embedding Shape: {embed.size()}')

        # Display example
        triplet = triplets_list[index]
        opt.logger.debug('Triplets string: ' + triplet)
        ax[i].set_title(triplet)
            
        if opt.pytorch_cuda.available:
            image = image.cpu()
        
        ax[i].imshow(image.permute(1, 2, 0))
        
    fig.savefig('./results/imagen_data_item.png')   # save the figure to file
    plt.close(fig)
 
    
if __name__ == "__main__":
    main()
    