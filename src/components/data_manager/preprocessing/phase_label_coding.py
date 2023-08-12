import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
import pandas as pd
from os.path import exists as file_exists

from src.components.utils.opt.build_opt import Opt

def get_phase_labels_for_videos_as_df(opt: Opt, videos: list, frames: list, fps: int):
    """
    Get phase labels for videos as a DataFrame.

    Parameters:
        opt (Opt): Options object containing dataset configurations.
        videos (list): List of video numbers.
        frames (list): List of frame numbers.
        fps (int): Frames per second.

    Returns:
        df_phase_labels (DataFrame): DataFrame containing video number, frame number,
                                     phase label, and phase label text.

    """
    df_phase_labels = pd.DataFrame()
    phase = list()
    phase_text = list()
    phase_dict = dict(**opt.datasets['data']['Cholec80']['classes'])
    
    # Iterate over unique video numbers
    for video_k in np.unique(videos):
        
        # Construct the path for phase labels file
        video_label_path = os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data']['Cholec80']['PATH_PHASE_LABELS'], f'video{video_k:02d}-phase.txt')
        
        # Get the frame indices corresponding to the current video number
        frame_indices = [frames[i] for i in np.argwhere(np.array(videos) == video_k)[:, 0]]
        
        # Calculate the interval based on the frame rate
        assert fps != 0, 'fps is zero! Please change...'
        interval = int(opt.datasets['data']['Cholec80']['fps'] / fps)
        
        with open(video_label_path, 'r') as file:
            lines = np.array(file.readlines())
            
            # Handle special case for video 56
            if video_k == 56:
                try:
                    lines = [lines[(i * interval) + 1] for i in frame_indices]
                except IndexError:
                    lines = [lines[(i * interval) + 1] for i in frame_indices[:-2]]
                    lines += lines[-2:]
            else:
                lines = [lines[(i * interval) + 1] for i in frame_indices]
        
        for line in lines:
            phase.append(line.split('\t')[1].split('\n')[0])
            phase_text.append(phase_dict[phase[-1]])
        
    # Create columns in the DataFrame for video number, frame number, phase label, and phase label text
    df_phase_labels[f'VIDEO NUMBER'] = videos
    df_phase_labels[f'FRAME NUMBER'] = frames
    df_phase_labels[f'PHASE LABEL'] = phase
    df_phase_labels[f'PHASE LABEL TEXT'] = phase_text
    
    opt.logger.debug(f'Phase labels gathered')
    
    return df_phase_labels


def main():
    """
    Main function for testing.

    """
    opt = Opt()

    # Generate random video number (k) and frame number (f)
    k = np.random.randint(1, 80)
    f = np.random.randint(1, 30000)
    
    # Get phase labels for the specified video and frame
    phase_labels = get_phase_labels_for_videos_as_df(opt=opt, videos=[k], frames=[f], fps=25)

    # Print test output
    opt.logger.info('---------- TEST OUTPUT ----------')
    opt.logger.info(f'Phase Labels shape: {len(phase_labels)}')
    opt.logger.info(f'Phase Labels:\n {phase_labels.head()}')

    
if __name__ == "__main__":
    main()