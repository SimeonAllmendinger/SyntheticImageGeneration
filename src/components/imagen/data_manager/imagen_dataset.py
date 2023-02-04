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
from src.components.imagen.data_manager.segment_coding import get_seg8k_df_train
from src.components.imagen.data_manager.text_embedding import get_text_ohe_embedding, get_text_t5_embedding
from src.components.imagen.utils.decorators import check_dataset_name, check_text_encoder
from src.components.imagen.data_manager.phase_label_coding import get_phase_labels_for_videos_as_df


class BaseImagenDataset(Dataset):
    
    def __init__(self, dataset_name: str, opt: Opt):
        
        #
        self.TEXT_ENCODER_NAME = opt.imagen['imagen']['text_encoder_name']
        self.DATASET = dataset_name
        self.use_phase_labels = opt.imagen['data']['Cholec80']['use_phase_labels']
        
        #
        super().__init__(folder=os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['data'][self.DATASET]['PATH_VIDEO_DIR']),
                         image_size=opt.imagen['data']['image_size'])
        
    @check_text_encoder
    def _set_text_embeds_(self, opt: Opt):
        
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            # Create OHE embedding
            self.text_embeds = get_text_ohe_embedding(triplet_dict_indices=self.df_train['FRAME TRIPLET DICT INDICES'], 
                                                      phase_label_encoding=self.df_train['PHASE LABEL ENCODING'],
                                                      opt=opt)
            
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':

            if opt.imagen['data']['Cholec80']['use_phase_labels']:
                triplets = self.df_train['TEXT PROMPT'].values
                phase_labels = self.df_train['PHASE LABEL'].values
                
                self.text_unique_list=np.unique(["{} in {}".format(a, b) for a, b in zip(triplets, 
                                                                                         phase_labels)]).tolist()

            else:
                # Create T5 embedding
                self.text_unique_list = self.df_train['TEXT PROMPT'].unique().tolist()
            
            self.text_embeds = get_text_t5_embedding(opt=opt,
                                                     dataset_name=self.DATASET,
                                                     triplets_unique_list=self.text_unique_list)


        opt.logger.debug('Text embedding created')
        opt.logger.debug('Text embedding shape: ' + str(self.text_embeds.size()))
    
    
    def __len__(self):
        return self.df_train.shape[0]
    
    
    def __getitem__(self, index):
        
        # Get Image
        path = os.path.join(self.folder, self.df_train['FRAME PATH'].values[index])
        image = Image.open(path)

        # Transform Image
        image = self.transform(image)
        
        # Get triplet embedding
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            text_embedding = self.text_embeds[index]
        
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':
            
            if self.use_phase_labels:
                triplet = self.df_train['TEXT PROMPT'].values[index]
                phase_label = self.df_train['PHASE LABEL'].values[index]
                
                text = triplet + ' in ' + phase_label
                
            else:
                # Get triplet text
                text = self.df_train['TEXT PROMPT'].values[index]
                
            index_unique = self.text_unique_list.index(text)
            text_embedding = self.text_embeds[index_unique]

        # Check gpu availability    
        if torch.cuda.is_available():
            image = image.cuda()
            text_embedding = text_embedding.cuda()
        
        return image, text_embedding


class CholecT45ImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecT45', opt=opt)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
                
        if opt.imagen['data']['Cholec80']['use_phase_labels']:
            
            #
            df_triplets = get_df_triplets(opt=opt)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_triplets['VIDEO NUMBER'].to_list(),
                                                                frames=df_triplets['FRAME NUMBER'].to_list(),
                                                                fps=opt.imagen['data']['Cholec80']['fps'])
            #
            self.df_train= pd.concat((df_triplets, df_phase_labels[['PHASE LABEL', 'PHASE LABEL ENCODING']]), axis=1)
            
        else: 
            
            self.df_train=get_df_triplets(opt=opt)
        

class CholecSeg8kImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecSeg8k', opt=opt)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
        
        if opt.imagen['data']['Cholec80']['use_phase_labels']:
            
            #
            df_train = get_seg8k_df_train(opt=opt, folder=self.folder)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_train['VIDEO NUMBER'].to_list(),
                                                                frames=df_train['FRAME NUMBER'].to_list(),
                                                                fps=opt.imagen['data']['Cholec80']['fps'])
            #
            self.df_train= pd.concat((df_train, df_phase_labels[['PHASE LABEL', 'PHASE LABEL ENCODING']]), axis=1)
            
        else: 
            self.df_train=get_seg8k_df_train(opt=opt, folder=self.folder)


class ConcatImagenDataset(torch.utils.data.ConcatDataset):
    
    def __init__(self, opt: Opt):
        #
        cholecT45_ds = CholecT45ImagenDataset(opt=opt)
        cholecSeg8k_ds = CholecSeg8kImagenDataset(opt=opt)
        
        self.df_train = pd.concat((cholecT45_ds.df_train, cholecSeg8k_ds.df_train), ignore_index=True)

        if cholecT45_ds.text_embeds.size()[1] != cholecSeg8k_ds.text_embeds.size()[2]:
            
            # create a tensor with shape [100, X, 768] filled with zeros
            new_embeds = torch.zeros(cholecSeg8k_ds.text_embeds.size()[0],
                                     cholecT45_ds.text_embeds.size()[1],
                                     cholecT45_ds.text_embeds.size()[2])
            
            # copy the values from the existing text embeds tensor to the new embeds tensor
            new_embeds[:, :6, :] = cholecSeg8k_ds.text_embeds
            
            # Renew the attribute text_embeds
            cholecSeg8k_ds.text_embeds = new_embeds
            
        #
        super().__init__([cholecT45_ds, cholecSeg8k_ds])
        
        
    def __getitem__(self, index):
        return super().__getitem__(idx=index)
        
          
def main():
    opt=Opt()

    if opt.imagen['data']['dataset'] == 'CholecT45':
        dataset = CholecT45ImagenDataset(opt=opt)
    elif opt.imagen['data']['dataset'] == 'CholecSeg8k':
        dataset = CholecSeg8kImagenDataset(opt=opt)
    elif opt.imagen['data']['dataset'] == 'Both':
        dataset = ConcatImagenDataset(opt=opt)
    
    path_train_df_file = os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['data']['CholecT45']['PATH_TRAIN_DF_FILE'])
    triplets_list = pd.read_json(path_train_df_file)['TEXT PROMPT'].values
    
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
        
    fig.savefig(os.path.join(opt.base['PATH_BASE_DIR'], './results/imagen_data_item.png'))   # save the figure to file
    plt.close(fig)
 
    
if __name__ == "__main__":
    main()
    