import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import torch
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
        self.t5_embedding = opt.imagen['dataset']['text_embedding']
        
        super().__init__(folder=opt.imagen['dataset']['PATH_VIDEO_DIR'],
                         image_size=opt.imagen['dataset']['image_size'])
        
        
        if self.t5_embedding:
            if file_exists(opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE']):
                opt.logger.debug('Load text embedding from file')
                
                self.triplet_embeds = torch.load(opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE'])
            
            else:
                opt.logger.debug('Create text embedding')
                
                self.triplet_embeds = t5_encode_text(texts=self.img_triplets['triplet_text'].to_list())
                torch.save(self.triplet_embeds, opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE'])

    def __getitem__(self, index):
        # Get Image
        path = os.path.join(self.folder, self.img_triplets.iloc[index, 0])
        image = Image.open(path)
        
        # Transform Image
        image = self.transform(image)
        
        if not self.t5_embedding:
            
            # Get triplet text
            triplet_text = self.img_triplets['triplet_text'].values[index]

            return image, triplet_text
        
        else:
            # Get triplet text
            triplet_embedding = self.triplet_embeds[index]
            
            return image, triplet_embedding
        

if __name__ == "__main__":
    opt=Opt()
    
    cholecT45_dataset = CholecT45ImagenDataset(opt=opt)
    
    # Plot
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    
    for i in range(2):
        # Example image with random index
        index = np.random.randint(low=1, high=200)
        image, triplet = cholecT45_dataset.__getitem__(index=index)

        # Display example
        if not opt.imagen['dataset']['text_embedding']:
            opt.logger.debug('Triplets string: ' + triplet)
            ax[i].set_title(triplet)
            
        ax[i].imshow(image.permute(1, 2, 0))
        
    plt.show()
    