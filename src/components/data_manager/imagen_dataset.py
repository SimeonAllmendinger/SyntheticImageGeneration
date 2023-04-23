import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import bisect

from imagen_pytorch.data import Dataset
from torch.utils.data import ConcatDataset
from PIL import Image

from src.components.utils.opt.build_opt import Opt
from src.components.data_manager.preprocessing.triplet_coding import get_df_triplets
from src.components.data_manager.preprocessing.segment_coding import get_seg8k_df_train
from src.components.data_manager.preprocessing.text_embedding import get_text_ohe_embedding, get_text_t5_embedding
from src.components.data_manager.preprocessing.phase_label_coding import get_phase_labels_for_videos_as_df
from src.components.imagen.utils.decorators import check_dataset_name, check_text_encoder


class BaseImagenDataset(Dataset):
    
    def __init__(self, dataset_name: str, opt: Opt, return_text=False):
        
        #
        self.TEXT_ENCODER_NAME = opt.imagen['imagen']['text_encoder_name']
        self.DATASET = dataset_name
        self.use_phase_labels = opt.datasets['data']['Cholec80']['use_phase_labels']
        self.return_text = return_text
        self.multi_gpu = opt.conductor['trainer']['multi_gpu']
        
        #
        super().__init__(folder=os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data'][self.DATASET]['PATH_VIDEO_DIR']),
                         image_size=opt.datasets['data']['image_size'])
        
    @check_text_encoder
    def _set_text_embeds_(self, opt: Opt):
        
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            # Create OHE embedding
            self.text_embeds = get_text_ohe_embedding(triplet_dict_indices=self.df_train['FRAME TRIPLET DICT INDICES'], 
                                                      phase_label_encoding=self.df_train['PHASE LABEL ENCODING'],
                                                      opt=opt)
        
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':

            self.text_unique_list = self.df_train['TEXT PROMPT'].unique().tolist()
            
            # Create T5 embedding
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
            
            # Get triplet text
            text = self.df_train['TEXT PROMPT'].values[index]
                
            index_unique = self.text_unique_list.index(text)
            text_embedding = self.text_embeds[index_unique]
        
        if self.return_text:
            return image, text_embedding, text
        
        else:
            return image, text_embedding


class CholecT45ImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt, return_text=False):
        super().__init__(dataset_name='CholecT45', opt=opt, return_text=return_text)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
                
        if opt.datasets['data']['Cholec80']['use_phase_labels']:
            
            #
            df_triplets = get_df_triplets(opt=opt)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_triplets['VIDEO NUMBER'].to_list(),
                                                                frames=df_triplets['FRAME NUMBER'].to_list(),
                                                                fps=opt.datasets['data']['CholecT45']['fps'])
            #
            self.df_train= pd.concat((df_triplets, df_phase_labels[['PHASE LABEL TEXT', 'PHASE LABEL']]), axis=1)
            
            # add the two columns using the custom function and append the resulting strings row-wise
            self.df_train['TEXT PROMPT'] = self.df_train.apply(lambda row: concatenate_strings(row['TEXT PROMPT'],row['PHASE LABEL TEXT']), axis=1)

        else: 
            
            self.df_train=get_df_triplets(opt=opt)
        

class CholecSeg8kImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt, return_text=False):
        super().__init__(dataset_name='CholecSeg8k', opt=opt, return_text=return_text)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
        
        if opt.datasets['data']['Cholec80']['use_phase_labels']:
            
            #
            df_train = get_seg8k_df_train(opt=opt, folder=self.folder)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_train['VIDEO NUMBER'].to_list(),
                                                                frames=df_train['FRAME NUMBER'].to_list(),
                                                                fps=opt.datasets['data']['CholecSeg8k']['fps'])
            #
            self.df_train= pd.concat((df_train, df_phase_labels[['PHASE LABEL TEXT', 'PHASE LABEL']]), axis=1)
            
        else: 
            self.df_train=get_seg8k_df_train(opt=opt, folder=self.folder)


class ConcatImagenDataset(ConcatDataset):
    
    def __init__(self, opt: Opt, return_text=False):
        
        #
        cholecT45_ds = CholecT45ImagenDataset(opt=opt, return_text=return_text)
        cholecSeg8k_ds = CholecSeg8kImagenDataset(opt=opt, return_text=return_text)
        
        #
        self.DATASET = 'Both'
        
        #
        self.df_train = pd.concat((cholecT45_ds.df_train, cholecSeg8k_ds.df_train), ignore_index=True)
        
        #
        opt.logger.debug(f'Text embed size: CholecT45={cholecT45_ds.text_embeds.size()} | CholecSeg8k={cholecSeg8k_ds.text_embeds.size()}')

        #
        if cholecT45_ds.text_embeds.size()[1] != cholecSeg8k_ds.text_embeds.size()[2]:
            
            # create a tensor with shape [A, B, 768] filled with zeros
            new_embeds = torch.zeros(cholecSeg8k_ds.text_embeds.size()[0],
                                     cholecT45_ds.text_embeds.size()[1],
                                     cholecT45_ds.text_embeds.size()[2])
            
            # copy the values from the existing text embeds tensor to the new embeds tensor
            new_embeds[:, :cholecSeg8k_ds.text_embeds.size()[1], :] = cholecSeg8k_ds.text_embeds
            
            # Renew the attribute text_embeds
            cholecSeg8k_ds.text_embeds = new_embeds
            
        #
        super().__init__([cholecT45_ds, cholecSeg8k_ds])
        

# define a function to concatenate two strings with a space in between
def concatenate_strings(s1, s2):
    return s1 + ' in ' + s2

def main():
    opt=Opt()

    #
    cholcet45_imagen_dataset = CholecT45ImagenDataset(opt=opt)
    
    #
    text_embed, image, text = cholcet45_imagen_dataset.__getitem__(index=0, 
                                                                   return_text=True)
    
    #
    opt.logger.info(f'TEXT: {text} \n IMAGE: {image} \n TEXT EMBED: {text_embed}')
    
if __name__ == "__main__":
    main()
    