import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import exists as file_exists
from imagen_pytorch.data import Dataset
from PIL import Image

from src.components.utils.opt.build_opt import Opt
from src.components.data_manager.CholecT45.triplet_coding import get_df_triplets
from src.components.data_manager.CholecT45.triplet_embedding import get_triplet_ohe_embedding, get_triplet_t5_embedding

class CholecT45ImagenDataset(Dataset):
    def __init__(self, opt: Opt):
        
        super().__init__(folder=opt.imagen['dataset']['PATH_VIDEO_DIR'],
                    image_size=opt.imagen['dataset']['image_size'])
                
        self.df_triplets = get_df_triplets(opt=opt)
        self.t5_embedding = opt.imagen['dataset']['t5_text_embedding']
        
        if self.t5_embedding:
            
            self.triplets_unique_list = self.df_triplets['triplet_text'].unique().tolist()
            self.triplet_embeds = get_triplet_t5_embedding(opt=opt, 
                                                           triplets_unique_list=self.triplets_unique_list)
            
        else:
            
            self.triplet_embeds = get_triplet_ohe_embedding(opt=opt, 
                                                            triplet_dict_indices=self.df_triplets['triplet_dict_indices'])
        
        opt.logger.debug('Text embedding created')
        opt.logger.debug('Text embedding shape: ' + str(self.triplet_embeds.size()))

    def __getitem__(self, index):
        # Get Image
        path = os.path.join(self.folder, self.df_triplets.iloc[index, 0])
        image = Image.open(path)
        
        # Get triplet text
        triplet_text = self.df_triplets['triplet_text'].values[index]

        # Transform Image
        image = self.transform(image)
        
        if self.t5_embedding:
            
            # Get triplet embedding
            index_unique = self.triplets_unique_list.index(triplet_text)
            triplet_embedding = self.triplet_embeds[index_unique]
        
        else:
            
            triplet_embedding = self.triplet_embeds[index]
            
        if torch.cuda.is_available():
            image = image.cuda()
            triplet_embedding = triplet_embedding.cuda()
        
        return image, triplet_embedding
        

def main():
    opt=Opt()
    
    cholecT45_dataset = CholecT45ImagenDataset(opt=opt)
    triplets_list = pd.read_json(opt.imagen['dataset']['PATH_TRIPLETS_DF_FILE'])['triplet_text'].values
    
    # Plot
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    
    for i in range(2):
        # Example image with random index
        index = np.random.randint(low=10000, high=20000)
        image, embed = cholecT45_dataset.__getitem__(index=index)
        
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
    