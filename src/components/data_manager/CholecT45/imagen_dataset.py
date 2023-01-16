import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import exists as file_exists
from imagen_pytorch.data import Dataset
from imagen_pytorch.t5 import t5_encode_text
from PIL import Image

from src.components.utils.opt.build_opt import Opt
from src.components.data_manager.CholecT45.triplet_coding import get_df_triplets

class CholecT45ImagenDataset(Dataset):
    def __init__(self, opt: Opt):
        
        self.img_triplets = get_df_triplets(opt=opt)
        self.t5_embedding = opt.imagen['dataset']['t5_text_embedding']
        
        super().__init__(folder=opt.imagen['dataset']['PATH_VIDEO_DIR'],
                         image_size=opt.imagen['dataset']['image_size'])

        if self.t5_embedding:
            
            self.triplets_unique_list = self.img_triplets['triplet_text'].unique().tolist()
            
            if not file_exists(f"{opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE']}"):

                opt.logger.debug('Create text embedding')

                self.triplet_embeds = t5_encode_text(texts=self.triplets_unique_list)
                    
                # Save embedding of unique triplets
                embed_save_path = opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE']
                torch.save(self.triplet_embeds, f=embed_save_path)
        
            else:
            
                self.triplet_embeds = torch.load(opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE'])
            
            opt.logger.debug('Text embedding created')
            opt.logger.debug('Text embedding shape: ' + str(self.triplet_embeds.size()))
            

    def __getitem__(self, index):
        # Get Image
        path = os.path.join(self.folder, self.img_triplets.iloc[index, 0])
        image = Image.open(path)
        
        # Get triplet text
        triplet_text = self.img_triplets['triplet_text'].values[index]
        
        # Transform Image
        image = self.transform(image)
        
        if not self.t5_embedding:
            
            if torch.cuda.is_available():
                image = image.cuda()
            
            return image, triplet_text
        
        else:
            
            # Get triplet embedding
            index_unique = self.triplets_unique_list.index(triplet_text)
            triplet_embedding = self.triplet_embeds[index_unique]
            
            if torch.cuda.is_available():
                image = image.cuda()
                triplet_embedding = triplet_embedding.cuda()

            return image, triplet_embedding
        

if __name__ == "__main__":
    opt=Opt()
    
    cholecT45_dataset = CholecT45ImagenDataset(opt=opt)
    
    # Plot
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    
    for i in range(2):
        # Example image with random index
        index = np.random.randint(low=1, high=2000)
        image, triplet = cholecT45_dataset.__getitem__(index=index)

        # Display example
        if not opt.imagen['dataset']['t5_text_embedding']:
            opt.logger.debug('Triplets string: ' + triplet)
            ax[i].set_title(triplet)
        
        if opt.pytorch_cuda.available:
            image = image.cpu()
        
        ax[i].imshow(image.permute(1, 2, 0))
        
    plt.show()
    