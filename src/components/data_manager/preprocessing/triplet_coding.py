import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from os.path import exists as file_exists

from src.components.utils.opt.build_opt import Opt


def get_df_triplets(opt: Opt):

    OPT_DATA = dict(**opt.datasets['data'])
    
    path_train_df_file = os.path.join(opt.base['PATH_BASE_DIR'], OPT_DATA['CholecT45']['PATH_TRAIN_DF_FILE'])
    
    if OPT_DATA['use_existing_data_files'] and file_exists(path_train_df_file):
        
        # Load df_triplets
        df_triplets = pd.read_json(path_train_df_file)
        
    else:
        # Get file paths of triplets .txt files
        triplets_files_paths = get_triplet_file_paths_in_dir_as_list(opt=opt)
        
        # Get dictionary triplet.txt file of triplet text mapping
        path_triplet_dict = os.path.join(opt.base['PATH_BASE_DIR'], OPT_DATA['CholecT45']['PATH_DICT_DIR'] + 'triplet.txt')
        triplets_dict = _load_text_data_(opt=opt, path=path_triplet_dict)
        
        # Initialize triplets_text and triplets_embed_path as list
        frame_paths=list()
        frame_numbers=list()
        video_numbers=list()
        frame_encodings=list()
        triplets_text_list = list()
        triplet_dict_indices_list = list()

        for i, triplets_file_path in enumerate(triplets_files_paths):

            # Get video number k
            video_k = int(triplets_file_path.strip('.txt')[-2:])
                
            # read triplet .txt file of video k
            triplets_data = pd.read_csv(triplets_file_path, sep=',', header=None)


            opt.logger.debug('Gather Triplet Data of Video ' + str(video_k))
            
            #
            frame_paths += ['VID' + f'{(video_k):02d}' + '/' + f'{frame_number:06d}' + '.png'
                            for frame_number in triplets_data.iloc[:, 0].to_list()]

            #
            frame_numbers += triplets_data.iloc[:, 0].to_list()
            video_numbers += [video_k] * len(triplets_data.iloc[:, 0].to_list())

            #
            [frame_encodings.append(triplets_data.iloc[frame_number, 1:].astype(int).to_list())
                for frame_number in triplets_data.iloc[:, 0].to_list()]


        # add paths and frame numbers of video to the DataFrame
        df_triplets = pd.DataFrame({'FRAME PATH': frame_paths,
                                    'VIDEO NUMBER': video_numbers,
                                    'FRAME NUMBER': frame_numbers,
                                    })

        # Get corresponding triplet encoding
        for triplet_encoding in frame_encodings:
            
            triplet_text, triplet_dict_indices = get_single_frame_triplet_decoding(frame_triplet_encoding=triplet_encoding,
                                                                                   triplets_dict=triplets_dict, 
                                                                                   opt=opt)
            
            # Append triplets text for each frame
            triplets_text_list.append(triplet_text)
            triplet_dict_indices_list.append(triplet_dict_indices)

        #
        df_triplets['TEXT PROMPT'] = triplets_text_list
        df_triplets['FRAME TRIPLET DICT INDICES'] = triplet_dict_indices_list
        
        #
        df_triplets.to_json(path_train_df_file)
        
    opt.logger.info('df_CholecT45_shape: ' + str(df_triplets.shape))
    opt.logger.debug('df_triplet_variations: ' + str(df_triplets['TEXT PROMPT'].unique().shape[0]))
    opt.logger.debug('df_triplets: \n' + str(df_triplets['TEXT PROMPT']))
    
    return df_triplets


def get_triplet_file_paths_in_dir_as_list(opt: Opt):

    # Fill file_paths list with all paths of the triplets.txt files
    glob_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data']['CholecT45']['PATH_TRIPLETS_DIR'] + "*.txt")
    file_paths = sorted(glob.glob(glob_path))
    
    opt.logger.debug('file_paths:' + str(file_paths))
    
    return file_paths

    
def get_single_frame_triplet_encoding(video_n :int, frame_n :int, opt: Opt):
    
    # Define load path of triplet encodings of video n
    load_path = os.path.join(opt.base['PATH_BASE_DIR'],opt.datasets['data']['CholecT45']['PATH_TRIPLETS_DIR'] + 'VID' + f'{video_n:02d}' + '.txt')
    
    # load .txt data as list
    lines = _load_text_data_(path=load_path, opt=opt)
    
    # get triplet of determined frame as encoded list
    frame_triplet_encoding = np.array(list(map(int, lines[frame_n].split(','))))[1:]
    
    opt.logger.debug('frame_triplet_encoding created')
    
    return frame_triplet_encoding
            
        
def get_single_frame_triplet_decoding(frame_triplet_encoding: list, triplets_dict: list, opt: Opt):
    
    # get triplet of determined frame as decoded prompt (list or string)
    if np.sum(frame_triplet_encoding) == 0:
        frame_triplet_decoding = np.array(triplets_dict)[-1:].tolist()
        triplet_dict_indices = [len(triplets_dict)-1]
    else:
        triplet_dict_indices = np.where(np.array(frame_triplet_encoding) == 1)[0]
        frame_triplet_decoding = np.array(triplets_dict)[triplet_dict_indices.astype(int)].tolist()
    
    frame_triplet_decoding_list=list()
    for i, triplet in enumerate(frame_triplet_decoding):
        frame_triplet_decoding_list.append([word.replace(':',',').strip('\n').split(',') for word in frame_triplet_decoding][i][1:])
        
    triplet_string=''
    for i, triplet in enumerate(frame_triplet_decoding_list):
        if i > 0:
            triplet_string+=' and '
        triplet_string +=' '.join(triplet)
    
    #
    for sub_string in ['null_verb','null_target','null_instrument']:
        triplet_string = remove_substring_and_whitespace(string=triplet_string,
                                                         substring=sub_string)
    
    #
    triplet_string = triplet_string.replace('_', ' ')
    triplet_string = triplet_string.replace('  ', ' ')

    return triplet_string, triplet_dict_indices
    
    
def _load_text_data_(path, opt: Opt):
    
    with open(path, 'r') as file:
        lines=file.readlines()
        
    opt.logger.debug(f'Data loaded from path: {path}')
    
    return lines


def remove_substring_and_whitespace(string, substring):
    
    # Find the index of the substring in the string
    index = string.find(substring)
    
    if index == -1:
        
        # Return the original string if the substring is not found
        return string
    
    # Get the index of the first character in the substring
    start = index + len(substring)
    
    # Get the index of the first whitespace character after the substring
    end = start
    while end < len(string) and string[end] != ' ':
        end += 1
    
    # Remove the substring and preceding whitespace from the string
    return string[:index] + string[end:]
 
 
def main():
    global opt
    opt = Opt()
    
    # TODO
    df_triplets=get_df_triplets(opt=opt)    
    
    
if __name__ == "__main__":
    main()