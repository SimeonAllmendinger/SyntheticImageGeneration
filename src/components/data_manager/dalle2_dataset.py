import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import glob
import bisect
import open_clip

from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T
from os.path import exists as file_exists
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
        self.batch_size = opt.conductor['trainer']['batch_size']
        
        #
        self.folder=os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][self.DATASET]['PATH_VIDEO_DIR'])
        self.image_size=opt.datasets['data']['image_size']
        
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(self.image_size)
        ])

        #
        *_, self.preprocess = open_clip.create_model_and_transforms(opt.dalle2['clip']['model_name'], 
                                                               pretrained=opt.dalle2['clip']['pretrained'])
        self.tokenizer = open_clip.get_tokenizer(model_name=opt.dalle2['clip']['model_name'])
        self._set_embed_paths_(opt=opt)
        
        
    def _set_embed_paths_(self, opt: Opt):
        
        self.image_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][self.DATASET]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
        self.text_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][self.DATASET]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR'])
        
    
    def _set_tokens_(self, opt: Opt):
        
        if file_exists(opt.datasets['data'][self.DATASET]['clip']['PATH_CLIP_IMAGE_ENCODING_FILE']) and opt.datasets['data'][self.DATASET]['clip']['use_existing_encodings']: 
            self.encoded_texts = torch.load(opt.datasets['data'][self.DATASET]['clip']['PATH_CLIP_IMAGE_ENCODING_FILE'])
            opt.logger.debug(f'Encoded texts loaded with size: {self.encoded_texts.size()}')
        else:
            self.encoded_texts = self.tokenizer(texts=self.df_train['TEXT PROMPT'].tolist(),
                                context_length=opt.dalle2['clip']['context_length'])
        
            torch.save(self.encoded_texts, opt.datasets['data'][self.DATASET]['clip']['PATH_CLIP_IMAGE_ENCODING_FILE'])
            opt.logger.debug(f'Encoded texts created with size: {self.encoded_texts.size()}')


    def __len__(self):
        return self.df_train.shape[0]
      
      
    def __getitem__(self, index, return_embeds=True, return_text=False):
        
        # Get Image
        path = os.path.join(self.folder, self.df_train['FRAME PATH'].values[index])
        image = Image.open(path)

        # Transform Image
        #image=self.transform(image)
        image = self.preprocess(image)
        
        # Get triplet text
        text = self.df_train['TEXT PROMPT'].values[index]
        encoded_text = self.encoded_texts[index]              
    
        # Check gpu availability
        if torch.cuda.is_available():
            image = image.cuda()
            encoded_text = encoded_text.cuda()
            
        if return_embeds:
            
            batch_index = int(index / self.batch_size)
            embed_image_batch = torch.load(self.image_embeds_save_dir_path + f'image_embeds_{batch_index:05d}.pt')
            embed_text_batch = torch.load(self.text_embeds_save_dir_path + f'text_embeds_{batch_index:05d}.pt')
            
            item_index = int(index % self.batch_size)
            embed_image=embed_image_batch[item_index] 
            embed_text=embed_text_batch[item_index]
            
            if torch.cuda.is_available():
                embed_image = embed_image.cuda()
                embed_text = embed_text.cuda()
                
            return image, encoded_text, embed_image, embed_text
        
        elif return_text:

            return image, encoded_text, embed_image, embed_text, text
        
        else:
                
            return image, encoded_text
    

class CholecT45Dalle2Dataset(BaseDalle2Dataset):
    
    def __init__(self, opt: Opt, clip_embedding=False):
        super().__init__(dataset_name='CholecT45', opt=opt)
        
        #
        self._set_df_train_(opt=opt)
        self._set_tokens_(opt=opt)
    
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
        self._set_tokens_(opt=opt)
    
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


