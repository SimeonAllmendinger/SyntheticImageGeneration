import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import bisect

from torch.utils.data import Dataset, ConcatDataset
from x_clip import CLIP, TextTransformer
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt
from src.components.data_manager.preprocessing.triplet_coding import get_df_triplets
from src.components.data_manager.preprocessing.segment_coding import get_seg8k_df_train
from src.components.data_manager.preprocessing.phase_label_coding import get_phase_labels_for_videos_as_df
from src.components.imagen.utils.decorators import check_dataset_name


class BaseDalle2Dataset(Dataset):
    
    def __init__(self, dataset_name: str, opt: Opt):
    
        #
        self.DATASET = dataset_name
        self.use_phase_labels = opt.datasets['data']['Cholec80']['use_phase_labels']
        
        #
        self.folder=os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][self.DATASET]['PATH_VIDEO_DIR'])
        self.image_size=opt.datasets['data']['image_size']
        
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])
        
           
    def __len__(self):
        return self.df_train.shape[0]
      
      
    def __getitem__(self, index, return_text=False):
        
        # Get Image
        path = os.path.join(self.folder, self.df_train['FRAME PATH'].values[index])
        image = Image.open(path)

        # Transform Image
        image = self.transform(image)
        
        # Get triplet text
        text = self.df_train['TEXT PROMPT'].values[index]

        # Check gpu availability    
        if torch.cuda.is_available():
            image = image.cuda()
        
        if return_text:
            
            return image, text, text
        
        else:
             return image, text
    

class CholecT45Dalle2Dataset(BaseDalle2Dataset):
    
    def __init__(self, opt: Opt, clip_embedding=False):
        super().__init__(dataset_name='CholecT45', opt=opt)
        
        #
        self._set_df_train_(opt=opt)
    
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
            self.df_train['TEXT PROMPT'] = self.df_train.apply(lambda row: concatenate_strings(row['TEXT PROMPT'], row['PHASE LABEL TEXT']), axis=1)

        else: 
            
            self.df_train=get_df_triplets(opt=opt)
        
        
class CholecSeg8kDalle2Dataset(BaseDalle2Dataset):
    
    def __init__(self, opt: Opt, clip_embedding=False):
        super().__init__(dataset_name='CholecSeg8k', opt=opt)
        
        self._set_df_train_(opt=opt)    
        
    
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
            
            # add the two columns using the custom function and append the resulting strings row-wise
            self.df_train['TEXT PROMPT'] = self.df_train.apply(lambda row: concatenate_strings(row['TEXT PROMPT'], row['PHASE LABEL TEXT']), axis=1)

        else: 
            self.df_train=get_seg8k_df_train(opt=opt, folder=self.folder)


class ConcatDalle2Dataset(ConcatDataset):
    
    def __init__(self, opt: Opt):
        
        #
        cholecT45_ds = CholecT45Dalle2Dataset(opt=opt)
        cholecSeg8k_ds = CholecSeg8kDalle2Dataset(opt=opt)
        
        #
        self.DATASET = 'Both'
        
        #
        self.df_train = pd.concat((cholecT45_ds.df_train, cholecSeg8k_ds.df_train), ignore_index=True)
        
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
        
        
    def __getitem__(self, index, return_text=False):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].__getitem__(index=sample_idx, 
                                                      return_text=return_text)


# define a function to concatenate two strings with a space in between
def concatenate_strings(s1, s2):
    return s1 + ' in ' + s2


def get_text_tensor(opt: Opt, text_batch, tokenizer):

    # tokenize the texts
    tokens = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")

    # concatenate the input_ids, attention_mask, and token_type_ids tensors
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    token_type_ids = torch.zeros_like(input_ids)  # set token_type_ids to 0 for all tokens
    text_tensor = torch.cat((input_ids, attention_mask, token_type_ids), dim=1).cuda()
        
    return text_tensor