import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import re
import torch
from os.path import exists as file_exists
from tqdm import tqdm
import numpy as np
from imagen_pytorch.t5 import t5_encode_text

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.data_manager.triplet_coding import _load_text_data_, get_df_triplets


def get_text_ohe_embedding(triplet_dict_indices: list, phase_label_encoding: list, opt: Opt):
    
    """
    Generates one-hot-encoding (OHE) embedding of triplet data and phase labels.
    
    Parameters:
        triplet_dict_indices (list): List of indices of triplet data.
        phase_label_encoding (list): List of one-hot-encoding of phase labels.
        opt (Opt): Instance of Opt class to store configuration options.
        
    Returns:
        triplet_ohe_embeds (torch.Tensor): Tensor of shape (len(triplet_dict_indices), 3, 36 or 29) 
            representing OHE embeddings of triplets and phase labels.
    """
    
    path_ohe_embedding_file = os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['data'][opt.imagen['data']['dataset']]['PATH_OHE_EMBEDDING_FILE'])
    
    if opt.imagen['data']['use_existing_data_files'] and file_exists(path_ohe_embedding_file) :
        
        triplet_ohe_embeds = torch.load(path_ohe_embedding_file)
        
    else:
        
        # Get dictionary .txt file of triplet mapping
        path_dict_maps=os.path.join(opt.base['PATH_BASE_DIR'],opt.imagen['data']['CholecT45']['PATH_DICT_DIR'] + 'maps.txt')
        map_dict = _load_text_data_(opt=opt, path=path_dict_maps)
        
        if opt.imagen['data']['Cholec80']['use_phase_labels']:
            triplet_ohe_embeds = np.zeros((len(triplet_dict_indices),3,29+7)) # add 7 surgical phases
        else:
            triplet_ohe_embeds = np.zeros((len(triplet_dict_indices),3,29)) #
        
        #
        for img, indices in tqdm(enumerate(triplet_dict_indices)):
            
            for k_tools, index in enumerate(indices):
                line = map_dict[index+1]
                numbers = [int(x) for x in re.findall(r'\d+', line)]
                triplet_map = numbers[1:4]

                for j, digit in enumerate(triplet_map):
                    
                    # instrument
                    if j==0:
                        triplet_ohe_embeds[img, k_tools, digit-1] = 1
                    # Verb
                    elif j==1:
                        if digit==9:
                            continue
                        else:
                            triplet_ohe_embeds[img, k_tools, digit+5] = 1
                    # Target
                    elif j==2:
                        if digit==14:
                            continue
                        else:
                            triplet_ohe_embeds[img, k_tools, digit+13] = 1
                
                
                if opt.imagen['data']['Cholec80']['use_phase_labels']:
                    
                    # add phase labels as one-hot-encoding
                    triplet_ohe_embeds[img, k_tools, 29:] = phase_label_encoding[img]
    
        triplet_ohe_embeds = torch.from_numpy(triplet_ohe_embeds).to(torch.float32)
        
        # Save embedding of unique triplets
        torch.save(triplet_ohe_embeds, f=path_ohe_embedding_file)
        
    return triplet_ohe_embeds


def get_text_t5_embedding(opt: Opt, dataset_name: str, triplets_unique_list: list, ):
    
    path_t5_embedding_file = os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['data'][dataset_name]['PATH_T5_EMBEDDING_FILE'])
          
    if opt.imagen['data']['use_existing_data_files'] and file_exists(path_t5_embedding_file) :
        
        triplet_embeds = torch.load(path_t5_embedding_file)

    else:
        
        opt.logger.debug('Create text embedding')

        triplet_embeds = t5_encode_text(texts=triplets_unique_list)
            
        # Save embedding of unique triplets
        torch.save(triplet_embeds, f=path_t5_embedding_file)
    
    return triplet_embeds

   
def main():
    opt = Opt()
    
    df_triplets = get_df_triplets(opt=opt)
    triplet_ohe_embeds=get_text_ohe_embedding(triplet_dict_indices=df_triplets['FRAME TRIPLET DICT INDICES'].values[0:10],
                                                 opt=opt)

    opt.logger.info('---------- TEST OUTPUT ----------')
    opt.logger.info(f'triplet_ohe_embeds shape: {triplet_ohe_embeds.shape}' )
    opt.logger.info(f'triplet_ohe_embeds: {triplet_ohe_embeds}' )

    
if __name__ == "__main__":
    main()