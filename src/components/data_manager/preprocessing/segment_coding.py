import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from os.path import exists as file_exists

from src.components.utils.opt.build_opt import Opt


def get_seg8k_df_train(opt: Opt, folder: str):
    """
    Get the training DataFrame for the CholecSeg8k dataset.

    Args:
        opt (Opt): Options object.
        folder (str): Folder path.

    Returns:
        pd.DataFrame: Training DataFrame.

    """
    OPT_DATA = dict(**opt.datasets['data'])

    # Path to the train DataFrame file
    path_train_df_file = os.path.join(opt.datasets['PATH_DATA_DIR'], OPT_DATA['CholecSeg8k']['PATH_TRAIN_DF_FILE'])

    if file_exists(path_train_df_file) and OPT_DATA['use_existing_data_files']:
        # Load existing train DataFrame
        df_train = pd.read_json(path_train_df_file)
    else:
        # Get sorted video numbers for CholecSeg8k
        seg8k_video_numbers = sorted(get_video_numbers(opt=opt, folder=folder))

        # Initialize lists to store frame paths, frame numbers, video numbers, text prompts, and indices
        frame_paths_list = list()
        frame_numbers_list = list()
        video_numbers_list = list()
        text_list = list()
        indices_list = list()

        for video_k in seg8k_video_numbers:
            # Get frame numbers and paths for the current video
            seg8k_frame_numbers, seg8k_frame_paths = get_frame_numbers_and_paths(opt=opt, folder=folder,
                                                                                 video_number=video_k)

            for j, frame_number in enumerate(seg8k_frame_numbers):
                if opt.datasets['data']['CholecSeg8k']['single_classes']:
                    # Process single classes
                    for key, value in opt.datasets['data']['CholecSeg8k']['classes'].items():
                        path = '/'.join(seg8k_frame_paths[j].replace('.png', '').split('/')[-3:]) + f'_{key}.png'

                        if file_exists(folder + path):
                            # Add frame path, frame number, video number, text prompt, and index to the lists
                            frame_paths_list.append(path)
                            frame_numbers_list.append(frame_number)
                            video_numbers_list.append(video_k)
                            text_list.append(key)
                            indices_list.append(0)

                if opt.datasets['data']['CholecSeg8k']['multi_classes']:
                    # Process multi classes
                    for key, value in opt.datasets['data']['CholecSeg8k']['multi_classes'].items():
                        path = '/'.join(seg8k_frame_paths[j].replace('.png', '').split('/')[-3:]) + f'_{key}.png'

                        if file_exists(folder + path):
                            # Add frame path, frame number, video number, text prompt, and index to the lists
                            frame_paths_list.append(path)
                            frame_numbers_list.append(frame_number)
                            video_numbers_list.append(video_k)
                            text_list.append(key)
                            indices_list.append(0)

        # Create a DataFrame with the collected information
        df_train = pd.DataFrame({'FRAME PATH': frame_paths_list,
                                 'VIDEO NUMBER': video_numbers_list,
                                 'FRAME NUMBER': frame_numbers_list,
                                 'TEXT PROMPT': text_list,
                                 'FRAME TRIPLET DICT INDICES': indices_list})

        # Save the DataFrame to a JSON file
        df_train.to_json(path_train_df_file)

    opt.logger.info('df_CholecSeg8k_shape: ' + str(df_train.shape))

    return df_train


def get_video_numbers(opt: Opt, folder):
    """
    Get the list of video numbers from the folder.

    Args:
        opt (Opt): Options object.
        folder (str): Folder path.

    Returns:
        list: List of video numbers.

    """
    video_numbers = list()
    
    # Get the list of file paths in the folder
    file_paths = os.listdir(folder)
    
    opt.logger.debug(f'file_paths: {file_paths}')
    
    # Iterate over the file paths
    for file_path in file_paths:
        # Extract the video number from the file path
        video_numbers.append(int(file_path.strip('video')))
    
    return video_numbers


def get_frame_numbers_and_paths(opt: Opt, folder: str, video_number: int):
    """
    Get the frame numbers and paths for a specific video.

    Args:
        opt (Opt): Options object.
        folder (str): Folder path.
        video_number (int): Video number.

    Returns:
        tuple: A tuple containing the list of frame numbers and the list of frame paths.

    """
    frame_numbers = list()
    frame_paths = list()

    # Retrieve the frame paths for the specified video number
    frame_paths = sorted(glob.glob(folder + f'video{video_number:02d}/*/*_endo.png'))
    
    # Iterate over the frame paths
    for frame_path in frame_paths:

        frame_numbers.append(int(frame_path.split('_')[-2]))

    return frame_numbers, frame_paths