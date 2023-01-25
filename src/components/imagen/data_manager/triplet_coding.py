import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from os.path import exists as file_exists

from src.components.utils.opt.build_opt import Opt


def get_df_triplets(dataset_name: str, opt: Opt):

    OPT_DATA = dict(**opt.imagen['data'])
    
    if file_exists(OPT_DATA[dataset_name]['PATH_TRIPLETS_DF_FILE']) and OPT_DATA['use_existing_data_files']:
        
        # Load df_triplets
        df_triplets = pd.read_json(OPT_DATA[dataset_name]['PATH_TRIPLETS_DF_FILE'])
        
    else:
        # Get file paths of triplets .txt files
        triplets_files_paths = get_triplet_file_paths_in_dir_as_list(dataset_name=dataset_name, opt=opt)
        
        # Get dictionary triplet.txt file of triplet text mapping
        triplets_dict = _load_text_data_(path=OPT_DATA['CholecT45']['PATH_DICT_DIR'] + 'triplet.txt', opt=opt)
        
        # Initialize triplets_text and triplets_embed_path as list
        frame_paths=list()
        frame_numbers=list()
        frame_encodings=list()
        triplets_text_list = list()
        triplet_dict_indices_list = list()

        for i, triplets_file_path in enumerate(triplets_files_paths):

            # Get video number k
            k = int(triplets_file_path.strip('.txt')[-2:])
                
            # read triplet .txt file of video k
            triplets_data = pd.read_csv(triplets_file_path, sep=',', header=None)

            if dataset_name == 'CholecT45':

                opt.logger.debug('Gather Triplet Data of Video ' + str(k))
                
                #
                frame_paths += ['VID' + f'{(k):02d}' + '/' + f'{frame_number:06d}' + '.png'
                                for frame_number in triplets_data.iloc[:, 0].to_list()]

                #
                frame_numbers += triplets_data.iloc[:, 0].to_list()

                #
                [frame_encodings.append(triplets_data.iloc[frame_number, 1:].astype(int).to_list())
                 for frame_number in triplets_data.iloc[:, 0].to_list()]

            elif dataset_name == 'CholecSeg8k':
                
                opt.logger.debug('Gather Triplet Data of Video ' + str(k))
                
                # Get frame numbers, fps and paths
                seg8k_frame_numbers, seg8k_frame_paths=get_cholecSeg8k_frame_numbers_and_paths_as_list(video_number=k, opt=opt)
                seg8k_fps=OPT_DATA['CholecSeg8k']['fps']
                
                for j, frame_number in enumerate(seg8k_frame_numbers):
                    if frame_number % seg8k_fps == 0 and frame_number/seg8k_fps < triplets_data.shape[0]:
                        
                        #
                        path='/'.join(seg8k_frame_paths[j].strip('.png').split('/')[-3:]) + '_color_mask.png'
                        frame_paths.append(path)

                        #
                        n = int(frame_number/seg8k_fps)
                        frame_numbers.append(n)

                        #
                        frame_encodings.append(triplets_data.iloc[n, 1:].astype(int).to_list())

        # add paths and frame numbers of video to the DataFrame
        df_triplets = pd.DataFrame({'FRAME PATH': frame_paths,
                                    'FRAME NUMBER': frame_numbers,
                                    'FRAME ENCODING': frame_encodings,
                                    })

        # Get corresponding triplet encoding
        for triplet_encoding in df_triplets['FRAME ENCODING']:
            
            triplet_text, triplet_dict_indices = get_single_frame_triplet_decoding(frame_triplet_encoding=triplet_encoding,
                                                                                   triplets_dict=triplets_dict, 
                                                                                   opt=opt)
            
            # Append triplets text for each frame
            triplets_text_list.append(triplet_text)
            triplet_dict_indices_list.append(triplet_dict_indices)

        df_triplets['FRAME TRIPLET TEXT'] = triplets_text_list
        df_triplets['FRAME TRIPLET DICT INDICES'] = triplet_dict_indices_list
        
        df_triplets.to_json(OPT_DATA[dataset_name]['PATH_TRIPLETS_DF_FILE'])
        
    opt.logger.info('df_triplets_shape: ' + str(df_triplets.shape))
    opt.logger.debug('df_triplet_variations: ' + str(df_triplets['FRAME TRIPLET TEXT'].unique().shape[0]))
    opt.logger.debug('df_triplets: \n' + str(df_triplets['FRAME TRIPLET TEXT']))
    
    return df_triplets


def get_triplet_file_paths_in_dir_as_list(dataset_name: str, opt: Opt):

    # Fill file_paths list with all paths of the triplets.txt files
    file_paths = sorted(glob.glob(opt.imagen['data']['CholecT45']['PATH_TRIPLETS_DIR'] + "*.txt"))
     
    if dataset_name == 'CholecSeg8k':
        video_numbers=get_cholecSeg8k_video_numbers_as_list(opt=opt)
        
        for file_path in file_paths:
            video_number=int(file_path.strip('.txt')[-2:])

            if video_number not in video_numbers:
                file_paths.remove(file_path)
    
    opt.logger.debug('file_paths:' + str(file_paths))
    
    return file_paths

    
def get_single_frame_triplet_encoding(video_n :int, frame_n :int, opt: Opt):
    
    # Define load path of triplet encodings of video n
    load_path = opt.imagen['data']['CholecT45']['PATH_TRIPLETS_DIR'] + 'VID' + f'{video_n:02d}' + '.txt'
    
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
            triplet_string+=', '
        triplet_string +=' '.join(triplet)

    return triplet_string, triplet_dict_indices


def get_cholecSeg8k_video_numbers_as_list(opt: Opt):
    
    video_numbers=list()
    video_paths=sorted(os.listdir(opt.imagen['data']['CholecSeg8k']['PATH_VIDEO_DIR']))
    
    for file_path in video_paths:
        video_numbers.append(int(file_path.strip('video')))
    
    return video_numbers
    

def get_cholecSeg8k_frame_numbers_and_paths_as_list(video_number: int, opt: Opt):

    video_dir_path = opt.imagen['data']['CholecSeg8k']['PATH_VIDEO_DIR']
    
    frame_numbers = list()
    frame_paths = list()


    frame_paths = sorted(glob.glob(video_dir_path + f'video{video_number:02d}/*/*_endo.png'))
    
    for frame_path in frame_paths:
        
        # Comment
        frame_numbers.append(int(frame_path.split('_')[-2]))

    return frame_numbers, frame_paths
    
    
def _load_text_data_(path, opt: Opt):
    
    with open(path, 'r') as file:
        lines=file.readlines()
        
    opt.logger.debug(f'Data loaded from path: {path}')
    
    return lines
 
 
def main():
    global opt
    opt = Opt()
    
    # 
    DATASET = opt.imagen['data']['dataset'] 
    df_triplets=get_df_triplets(dataset_name=DATASET, opt=opt)    
    
    
if __name__ == "__main__":
    main()