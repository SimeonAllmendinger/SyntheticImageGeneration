import pandas as pd
import numpy as np
from tqdm import tqdm
from google_access import authenticate_google_access

cholecSeg8k_id = '16-yrIKy1ZyTG21ZSBJkM9RHETIvzss8d'
cholecT45_id = '1--i6vGXr0Or-VKzZww0S3dKTd1yYSU8e'

def get_google_drive_filenames(drive_id :str, drive):

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(drive_id)}).GetList()
    ''' for file in video_list:
        print('title: %s, id: %s' % (file['title'], file['id']))'''
    return file_list


def get_frame_list_seg8k(frames):
    frame_list = list()
    for frame in frames['title'].values:
        frame_list.append(frame[-5:])
        
    return frame_list


def get_frame_list_t45(frames):
    frame_list = list()
    for frame in frames['title'].values:
        frame_list.append(frame[-9:-4])
        
    return frame_list


def main():
    drive=authenticate_google_access()
    cholecSeg8k_video_list=pd.DataFrame(get_google_drive_filenames(drive_id=cholecSeg8k_id,
                        drive=drive))
    cholecT45_video_list=pd.DataFrame(get_google_drive_filenames(drive_id=cholecT45_id,
                    drive=drive))
    
    cholecSeg8k_array = np.zeros((1,2))
    cholecT45_array = np.zeros((1,2))

    for i in tqdm(range(1,81)):
        cholecSeg8k_video_id = cholecSeg8k_video_list[cholecSeg8k_video_list['title'] == 'video' + f"{i:02d}"].id
        cholecT45_video_id = cholecT45_video_list[cholecT45_video_list['title'] == 'VID' + f"{i:02d}"].id
        if len(cholecSeg8k_video_id.values) > 0:
            frames=pd.DataFrame(get_google_drive_filenames(drive=drive,
                                              drive_id=str(cholecSeg8k_video_id.values[0])))
        
            frame_list=get_frame_list_seg8k(frames=frames)
            #print('Number of Seg8k files:', len(frame_list))
            
            for frame_number in frame_list:
                for j in range(0,80):
                    frame_no=int(frame_number)+j
                    frame_row=np.array(['Video' + f"{i:02d}", f"{frame_no:05d}"])
                    cholecSeg8k_array = np.vstack([cholecSeg8k_array, frame_row])
                    
        if len(cholecT45_video_id.values) > 0:
            frames=pd.DataFrame(get_google_drive_filenames(drive=drive,
                                              drive_id=str(cholecT45_video_id.values[0])))
        
            frame_list=get_frame_list_t45(frames=frames)
            #print('Number of T45 files:',len(frame_list))
            
            for frame_number in frame_list:
                frame_row=np.array(['Video' + f"{i:02d}", frame_number])
                cholecT45_array = np.vstack([cholecT45_array, frame_row])
            
    
    np.save('cholecSeg8k_array.npy', cholecSeg8k_array[1:]) 
    np.save('cholecT45_array.npy', cholecT45_array[1:])      
                       

if __name__ == '__main__':
    main()
    