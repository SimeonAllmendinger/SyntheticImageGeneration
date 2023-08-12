import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from os.path import exists as file_exists

from src.components.utils.opt.build_opt import Opt


def get_df_triplets(opt: Opt):
    """
    Get the DataFrame of triplets.

    Args:
        opt (Opt): Options object.

    Returns:
        pd.DataFrame: DataFrame containing the triplets.

    """
    OPT_DATA = dict(**opt.datasets['data'])
    
    # Set the path to the train DataFrame file
    path_train_df_file = os.path.join(opt.datasets['PATH_DATA_DIR'], OPT_DATA['CholecT45']['PATH_TRAIN_DF_FILE'])
    
    opt.logger.debug(path_train_df_file)
    
    if OPT_DATA['use_existing_data_files'] and file_exists(path_train_df_file):
        # Load the existing train DataFrame
        df_triplets = pd.read_json(path_train_df_file)
        
    else:
        # Get file paths of triplet .txt files
        triplets_files_paths = get_triplet_file_paths_in_dir_as_list(opt=opt)
        
        # Get the dictionary triplet.txt file of triplet text mapping
        path_triplet_dict = os.path.join(opt.datasets['PATH_DATA_DIR'], OPT_DATA['CholecT45']['PATH_DICT_DIR'] + 'triplet.txt')
        triplets_dict = _load_text_data_(opt=opt, path=path_triplet_dict)
        
        # Initialize lists for frame paths, frame numbers, video numbers, frame encodings, triplets text, and triplet dictionary indices
        frame_paths = list()
        frame_numbers = list()
        video_numbers = list()
        frame_encodings = list()
        triplets_text_list = list()
        triplet_dict_indices_list = list()

        for i, triplets_file_path in enumerate(triplets_files_paths):

            # Get the video number k
            video_k = int(triplets_file_path.strip('.txt')[-2:])
                
            # Read the triplet .txt file of video k
            triplets_data = pd.read_csv(triplets_file_path, sep=',', header=None)

            opt.logger.debug('Gather Triplet Data of Video ' + str(video_k))
            
            # Append frame paths
            frame_paths += ['VID' + f'{(video_k):02d}' + '/' + f'{frame_number:06d}' + '.png'
                            for frame_number in triplets_data.iloc[:, 0].to_list()]

            # Append frame numbers
            frame_numbers += triplets_data.iloc[:, 0].to_list()
            
            # Append video numbers
            video_numbers += [video_k] * len(triplets_data.iloc[:, 0].to_list())

            # Append frame encodings
            [frame_encodings.append(triplets_data.iloc[frame_number, 1:].astype(int).to_list())
                for frame_number in triplets_data.iloc[:, 0].to_list()]


        # Create the DataFrame with frame paths, video numbers, and frame numbers
        df_triplets = pd.DataFrame({'FRAME PATH': frame_paths,
                                    'VIDEO NUMBER': video_numbers,
                                    'FRAME NUMBER': frame_numbers,
                                    })

        # Get corresponding triplet encoding
        for triplet_encoding in frame_encodings:
            
            # Get the triplet text and triplet dictionary indices
            triplet_text, triplet_dict_indices = get_single_frame_triplet_decoding(frame_triplet_encoding=triplet_encoding,
                                                                                   triplets_dict=triplets_dict, 
                                                                                   opt=opt)
            
            # Append triplets text for each frame
            triplets_text_list.append(triplet_text)
            triplet_dict_indices_list.append(triplet_dict_indices)

        # Add triplets text and triplet dictionary indices to the DataFrame
        df_triplets['TEXT PROMPT'] = triplets_text_list
        df_triplets['FRAME TRIPLET DICT INDICES'] = triplet_dict_indices_list
        
        #
        df_triplets.to_json(path_train_df_file)
        
    opt.logger.info('df_CholecT45_shape: ' + str(df_triplets.shape))
    opt.logger.debug('df_triplet_variations: ' + str(df_triplets['TEXT PROMPT'].unique().shape[0]))
    opt.logger.debug('df_triplets: \n' + str(df_triplets['TEXT PROMPT']))
    
    return df_triplets


def get_triplet_file_paths_in_dir_as_list(opt: Opt):
    """
    Get a list of file paths for triplet .txt files in a directory.

    Args:
        opt (Opt): Options object.

    Returns:
        list: List of file paths.

    """
    # Fill file_paths list with all paths of the triplets.txt files
    glob_path = os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data']['CholecT45']['PATH_TRIPLETS_DIR'] + "*.txt")
    file_paths = sorted(glob.glob(glob_path))
    
    opt.logger.debug('file_paths:' + str(file_paths))
    
    return file_paths


def get_single_frame_triplet_encoding(video_n: int, frame_n: int, opt: Opt):
    """
    Get the triplet encoding of a single frame in a video.

    Args:
        video_n (int): Video number.
        frame_n (int): Frame number.
        opt (Opt): Options object.

    Returns:
        numpy.ndarray: Triplet encoding of the frame.

    """
    # Define load path of triplet encodings of video n
    load_path = os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data']['CholecT45']['PATH_TRIPLETS_DIR'] + 'VID' + f'{video_n:02d}' + '.txt')
    
    # load .txt data as list
    lines = _load_text_data_(path=load_path, opt=opt)
    
    # get triplet of determined frame as encoded list
    frame_triplet_encoding = np.array(list(map(int, lines[frame_n].split(','))))[1:]
    
    opt.logger.debug('frame_triplet_encoding created')
    
    return frame_triplet_encoding

        
def get_single_frame_triplet_decoding(frame_triplet_encoding: list, triplets_dict: list, opt: Opt):
    """
    Decode the triplet encoding of a single frame into a readable prompt.

    Args:
        frame_triplet_encoding (list): Triplet encoding of the frame.
        triplets_dict (list): Dictionary of triplet mappings.
        opt (Opt): Options object.

    Returns:
        tuple: A tuple containing the decoded triplet prompt and the dictionary indices.

    """
    # get triplet of determined frame as decoded prompt (list or string)
    if np.sum(frame_triplet_encoding) == 0:
        frame_triplet_decoding = np.array(triplets_dict)[-1:].tolist()
        triplet_dict_indices = [len(triplets_dict)-1]
    else:
        triplet_dict_indices = np.where(np.array(frame_triplet_encoding) == 1)[0]
        frame_triplet_decoding = np.array(triplets_dict)[triplet_dict_indices.astype(int)].tolist()
    
    frame_triplet_decoding_list = list()
    for i, triplet in enumerate(frame_triplet_decoding):
        frame_triplet_decoding_list.append([word.replace(':',',').strip('\n').split(',') for word in frame_triplet_decoding][i][1:])
        
    triplet_string = ''
    for i, triplet in enumerate(frame_triplet_decoding_list):
        if i > 0:
            triplet_string += ' and '
        triplet_string += ' '.join(triplet)
    
    # Remove null words from the triplet string
    for sub_string in ['null_verb', 'null_target', 'null_instrument']:
        triplet_string = remove_substring_and_whitespace(string=triplet_string, substring=sub_string)
    
    # Remove underscores and extra whitespaces
    triplet_string = triplet_string.replace('_', ' ')
    triplet_string = triplet_string.replace('  ', ' ')

    return triplet_string, triplet_dict_indices
    
    
def load_text_data(path, opt):
    """
    Load text data from a file.

    Args:
        path (str): The file path of the text data.
        opt (Opt): An object containing options and configurations.

    Returns:
        list[str] or None: A list of strings representing the lines of the text data,
                           or None if an error occurred.

    Raises:
        FileNotFoundError: If the specified file is not found.
        Exception: If an error occurs while loading the data.
    """
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        opt.logger.error(f"File not found: {path}")
        return None
    except Exception as e:
        opt.logger.error(f"Error loading data from {path}: {str(e)}")
        return None

    return lines


def remove_substring_and_whitespace(string, substring):
    """
    Remove a substring and any preceding whitespace from a string.

    Args:
        string (str): The original string.
        substring (str): The substring to be removed.

    Returns:
        str: The modified string with the substring and preceding whitespace removed.
    """
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