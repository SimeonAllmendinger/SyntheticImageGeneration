import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from src.components.utils.opt.build_opt import Opt


def get_df_triplets(opt: Opt):
    
    triplets_files_paths = get_triplet_file_paths_in_dir_as_list(opt=opt)
    triplets_dict = _load_text_data(opt=opt, path=opt.imagen['dataset']['PATH_DICT_DIR'] + 'triplet.txt')

    triplets_text = list()

    for i, triplets_file_path in enumerate(triplets_files_paths):
        if i == 0:
            df_img_triplets = pd.read_csv(triplets_file_path, sep=',', header=None)
            df_img_triplets.iloc[:, 0] = ['VID' + f'{(i+1):02d}' + '/' + f'{j:06d}' + '.png'
                                          for j in df_img_triplets.iloc[:, 0].to_list()]
            
        else:
            img_triplets = pd.read_csv(triplets_file_path, sep=',', header=None)
            img_triplets.iloc[:, 0] = ['VID' + f'{(i+1):02d}' + '/' + f'{j:06d}' + '.png'
                                       for j in img_triplets.iloc[:, 0].to_list()]
            df_img_triplets = pd.concat([df_img_triplets, img_triplets],
                                        ignore_index=True)
    
    # Get corresponding triplet encoding
    for i in range(df_img_triplets.shape[0]):
        triplet_encoding = df_img_triplets.iloc[i, 1:].astype(int)
        triplet_text = get_single_frame_triplet_decoding(frame_triplet_encoding=triplet_encoding,
                                                               triplets_dict=triplets_dict)
        triplets_text.append(triplet_text)
        

    df_img_triplets['triplet_text'] = triplets_text
    df_triplets = pd.concat((df_img_triplets.iloc[:, 0], df_img_triplets['triplet_text']), axis=1)
    
    opt.logger.info('df_triplets_shape: ' + str(df_triplets.shape))
    opt.logger.debug('df_triplets: \n' + str(df_triplets.head(5)))
    
    return df_triplets


def get_triplet_file_paths_in_dir_as_list(opt: Opt):
    
    file_paths=list()

    for file_path in glob.glob(opt.imagen['dataset']['PATH_TRIPLETS_DIR'] + "*.txt"):
        file_paths.append(file_path)
    
    opt.logger.debug(file_paths)
    
    return file_paths

    
def get_single_frame_triplet_encoding(opt :Opt, video_n :int, frame_n :int):
    
    # Define load path of triplet encodings of video n
    load_path = opt.imagen['dataset']['PATH_TRIPLETS_DIR'] + 'VID' + f'{video_n:02d}' + '.txt'
    
    # load .txt data as list
    lines = _load_text_data(opt=opt, path=load_path)
    
    # get triplet of determined frame as encoded list
    frame_triplet_encoding = np.array(list(map(int, lines[frame_n].split(','))))
    
    opt.logger.debug('frame_triplet_encoding created')
    
    return frame_triplet_encoding
            
        
def get_single_frame_triplet_decoding(frame_triplet_encoding :list(), triplets_dict):
    
    # get triplet of determined frame as decoded prompt (list or string)
    if frame_triplet_encoding.sum() == 0:
        triplet_string = np.array(triplets_dict)[-1].tolist()
    else:
        triplet_dict_indices = np.where(frame_triplet_encoding[:] == 1)[0]
        frame_triplet_decoding = np.array(triplets_dict)[triplet_dict_indices.astype(int)].tolist()
    
        frame_triplet_decoding_list=list()
        for i, triplet in enumerate(frame_triplet_decoding):
            frame_triplet_decoding_list.append([word.replace(':',',').strip('\n').split(',') for word in frame_triplet_decoding][i][1:])
        
        triplet_string=''
        for i, triplet in enumerate(frame_triplet_decoding_list):
            if i > 0:
                triplet_string+=', '
            triplet_string +=' '.join(triplet)

    return triplet_string
    
    
def _load_text_data(opt :Opt, path):
    
    with open(path, 'r') as file:
        lines=file.readlines()
        
    opt.logger.debug(f'Data loaded from path: {path}')
    
    return lines
   
 
if __name__ == "__main__":
    opt = Opt()
    # Define load path of triplet encodings of video n
    load_path = opt.imagen['dataset']['PATH_DICT_DIR'] + 'triplet.txt'
    
    # load .txt data as list
    triplets_dict = _load_text_data(opt=opt, path=load_path)
    
    frame_triplet_encoding=get_frame_triplet_encoding(opt=opt, 
                                                      video_n=1, 
                                                      frame_n=10)
    frame_triplet_decoding=get_frame_triplet_decoding(opt=opt, 
                                                      frame_triplet_encoding=
                                                      frame_triplet_encoding[1:], 
                                                      triplets_dict=triplets_dict)    
    
    print('---------- TEST OUTPUT ----------')
    print('frame_triplet_encoding shape: ', frame_triplet_encoding.shape)
    print('frame_triplet_decoding: ', frame_triplet_decoding)