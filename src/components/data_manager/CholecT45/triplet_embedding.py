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
from src.components.data_manager.CholecT45.triplet_coding import _load_text_data_, get_df_triplets


def get_triplet_ohe_embedding(opt: Opt, triplet_dict_indices: list):
    
    if opt.imagen['dataset']['use_existing_data_files'] and file_exists(f"{opt.imagen['dataset']['PATH_OHE_EMBEDDING_FILE']}") :
        
        triplet_ohe_embeds = torch.load(opt.imagen['dataset']['PATH_OHE_EMBEDDING_FILE'])
        
    else:
        
        # Get dictionary .txt file of triplet mapping
        map_dict = _load_text_data_(opt=opt, path=opt.imagen['dataset']['PATH_DICT_DIR'] + 'maps.txt')
        
        triplet_ohe_embeds = np.zeros((len(triplet_dict_indices),3,29))
        
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
    
        triplet_ohe_embeds = torch.from_numpy(triplet_ohe_embeds).to(torch.float32)
        
        # Save embedding of unique triplets
        embed_save_path = opt.imagen['dataset']['PATH_OHE_EMBEDDING_FILE']
        torch.save(triplet_ohe_embeds, f=embed_save_path)
        
    return triplet_ohe_embeds


def get_triplet_t5_embedding(opt: Opt, triplets_unique_list: list):
          
    if opt.imagen['dataset']['use_existing_data_files'] and file_exists(f"{opt.imagen['dataset']['PATH_T5_EMBEDDING_FILE']}") :
        
        triplet_embeds = torch.load(opt.imagen['dataset']['PATH_T5_EMBEDDING_FILE'])

    else:
        
        opt.logger.debug('Create text embedding')

        triplet_embeds = t5_encode_text(texts=triplets_unique_list)
            
        # Save embedding of unique triplets
        embed_save_path = opt.imagen['dataset']['PATH_T5_EMBEDDING_FILE']
        torch.save(triplet_embeds, f=embed_save_path)
    
    return triplet_embeds

   
def main(): 
    opt = Opt()
    
    df_triplets = get_df_triplets(opt=opt)
    triplet_ohe_embeds=get_triplet_ohe_embedding(opt=opt, triplet_dict_indices=df_triplets['triplet_dict_indices'].values[0:10])

    opt.logger.info('---------- TEST OUTPUT ----------')
    opt.logger.info(f'triplet_ohe_embeds shape: {triplet_ohe_embeds.shape}' )
    opt.logger.info(f'triplet_ohe_embeds: {triplet_ohe_embeds}' )

    
if __name__ == "__main__":
    main()