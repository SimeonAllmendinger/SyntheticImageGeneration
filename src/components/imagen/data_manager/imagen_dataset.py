import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import bisect

from torch.utils.data import DataLoader, random_split
from collections import Counter
from imagen_pytorch.data import Dataset
from PIL import Image
from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt
from src.components.imagen.data_manager.preprocessing.triplet_coding import get_df_triplets
from src.components.imagen.data_manager.preprocessing.segment_coding import get_seg8k_df_train
from src.components.imagen.data_manager.preprocessing.text_embedding import get_text_ohe_embedding, get_text_t5_embedding
from src.components.imagen.utils.decorators import check_dataset_name, check_text_encoder
from src.components.imagen.data_manager.preprocessing.phase_label_coding import get_phase_labels_for_videos_as_df


class BaseImagenDataset(Dataset):
    
    def __init__(self, dataset_name: str, opt: Opt):
        
        #
        self.TEXT_ENCODER_NAME = opt.imagen['imagen']['text_encoder_name']
        self.DATASET = dataset_name
        self.use_phase_labels = opt.imagen['data']['Cholec80']['use_phase_labels']
        
        #
        super().__init__(folder=os.path.join(opt.base['PATH_BASE_DIR'], opt.imagen['data'][self.DATASET]['PATH_VIDEO_DIR']),
                         image_size=opt.imagen['data']['image_size'])
        
    @check_text_encoder
    def _set_text_embeds_(self, opt: Opt):
        
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            # Create OHE embedding
            self.text_embeds = get_text_ohe_embedding(triplet_dict_indices=self.df_train['FRAME TRIPLET DICT INDICES'], 
                                                      phase_label_encoding=self.df_train['PHASE LABEL ENCODING'],
                                                      opt=opt)
        
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':

            if opt.imagen['data']['Cholec80']['use_phase_labels']:
                triplets = self.df_train['TEXT PROMPT'].values
                phase_labels = self.df_train['PHASE LABEL TEXT'].values
                
                self.text_unique_list=np.unique(["{} in {}".format(a, b) for a, b in zip(triplets, 
                                                                                         phase_labels)]).tolist()

            else:
                
                #
                self.text_unique_list = self.df_train['TEXT PROMPT'].unique().tolist()
            
            # Create T5 embedding
            self.text_embeds = get_text_t5_embedding(opt=opt,
                                                     dataset_name=self.DATASET,
                                                     triplets_unique_list=self.text_unique_list)


        opt.logger.debug('Text embedding created')
        opt.logger.debug('Text embedding shape: ' + str(self.text_embeds.size()))
    
    
    def __len__(self):
        return self.df_train.shape[0]
    
    
    def __getitem__(self, index, return_text=False):
        
        # Get Image
        path = os.path.join(self.folder, self.df_train['FRAME PATH'].values[index])
        image = Image.open(path)

        # Transform Image
        image = self.transform(image)
        
        # Get triplet embedding
        if self.TEXT_ENCODER_NAME == 'ohe_encoder':
            
            text_embedding = self.text_embeds[index]
        
        elif self.TEXT_ENCODER_NAME == 'google/t5-v1_1-base':
            
            if self.use_phase_labels:
                triplet = self.df_train['TEXT PROMPT'].values[index]
                phase_label = self.df_train['PHASE LABEL TEXT'].values[index]
                
                text = triplet + ' in ' + phase_label
                
            else:
                # Get triplet text
                text = self.df_train['TEXT PROMPT'].values[index]
                
            index_unique = self.text_unique_list.index(text)
            text_embedding = self.text_embeds[index_unique]

        # Check gpu availability    
        if torch.cuda.is_available():
            image = image.cuda()
            text_embedding = text_embedding.cuda()
        
        if return_text:
            return image, text_embedding, text
        
        else:
            return image, text_embedding


class CholecT45ImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecT45', opt=opt)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
                
        if opt.imagen['data']['Cholec80']['use_phase_labels']:
            
            #
            df_triplets = get_df_triplets(opt=opt)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_triplets['VIDEO NUMBER'].to_list(),
                                                                frames=df_triplets['FRAME NUMBER'].to_list(),
                                                                fps=opt.imagen['data']['CholecT45']['fps'])
            #
            self.df_train= pd.concat((df_triplets, df_phase_labels[['PHASE LABEL TEXT', 'PHASE LABEL']]), axis=1)
            
        else: 
            
            self.df_train=get_df_triplets(opt=opt)
        

class CholecSeg8kImagenDataset(BaseImagenDataset):
    
    def __init__(self, opt: Opt):
        super().__init__(dataset_name='CholecSeg8k', opt=opt)
        
        #
        self._set_df_train_(opt=opt)                          
        self._set_text_embeds_(opt=opt)
    
    
    @check_dataset_name
    def _set_df_train_(self, opt: Opt):
        
        if opt.imagen['data']['Cholec80']['use_phase_labels']:
            
            #
            df_train = get_seg8k_df_train(opt=opt, folder=self.folder)

            #
            df_phase_labels = get_phase_labels_for_videos_as_df(opt=opt,
                                                                videos=df_train['VIDEO NUMBER'].to_list(),
                                                                frames=df_train['FRAME NUMBER'].to_list(),
                                                                fps=opt.imagen['data']['CholecSeg8k']['fps'])
            #
            self.df_train= pd.concat((df_train, df_phase_labels[['PHASE LABEL TEXT', 'PHASE LABEL']]), axis=1)
            
        else: 
            self.df_train=get_seg8k_df_train(opt=opt, folder=self.folder)


class ConcatImagenDataset(torch.utils.data.ConcatDataset):
    
    def __init__(self, opt: Opt):
        
        #
        cholecT45_ds = CholecT45ImagenDataset(opt=opt)
        cholecSeg8k_ds = CholecSeg8kImagenDataset(opt=opt)
        
        #
        self.DATASET = 'Both'
        
        #
        self.df_train = pd.concat((cholecT45_ds.df_train, cholecSeg8k_ds.df_train), ignore_index=True)

        #
        if cholecT45_ds.text_embeds.size()[1] != cholecSeg8k_ds.text_embeds.size()[2]:
            
            # create a tensor with shape [100, X, 768] filled with zeros
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
        

def get_train_valid_ds(opt: Opt, testing=False):
    """
    Returns the training and validation datasets based on the specified options.

    Parameters
    ----------
    opt : Opt
        An object containing various options or configurations.

    Returns
    -------
    train_dataset : Dataset
        The training dataset.
    valid_dataset : Dataset
        The validation dataset.

    """
    
    # Check which dataset is specified in the opt object
    if opt.imagen['data']['dataset'] == 'CholecT45' or (testing and opt.imagen['testing']['only_triplets']):
        imagen_dataset = CholecT45ImagenDataset(opt=opt)
    
    elif opt.imagen['data']['dataset'] == 'CholecSeg8k':
        imagen_dataset = CholecSeg8kImagenDataset(opt=opt)
    
    elif opt.imagen['data']['dataset'] == 'Both':
        imagen_dataset = ConcatImagenDataset(opt=opt)
    
    if testing:
        
        return imagen_dataset
    
    else:
        
        # Split the instantiated dataset into training and validation datasets
        train_valid_split=[opt.imagen['trainer']['train_split'], opt.imagen['trainer']['valid_split']]
        train_dataset, valid_dataset = random_split(dataset=imagen_dataset, lengths=train_valid_split)
        
        # Return the training and validation datasets
        return train_dataset, valid_dataset
       

def get_train_valid_dl(opt: Opt, train_dataset, valid_dataset):
    
    train_generator = DataLoader(dataset=train_dataset, 
                                  batch_size=opt.imagen['trainer']['batch_size'], 
                                  shuffle=opt.imagen['trainer']['shuffle']
                                )
    valid_generator = DataLoader(dataset=valid_dataset, 
                                  batch_size=opt.imagen['trainer']['batch_size'], 
                                  shuffle=opt.imagen['trainer']['shuffle']
                                )
    
    return train_generator, valid_generator


def visualize_class_representation(opt: Opt, dataset, quantity=25):
    
    labels=list()
    
    # Get the labels for all instances in the dataset
    for i in tqdm(range(dataset.__len__())):
        image, embed, text = dataset.__getitem__(index=i, return_text=True)
        labels.append(text)
        
    # Count the number of instances in each class
    class_counts = Counter(labels)
    
    # create a list of tuples by pairing the elements of the two lists
    zipped_lists = list(zip(class_counts.keys(), class_counts.values()))

    # sort the list of tuples based on the values in the first element of each tuple
    zipped_lists.sort(key=lambda x: x[1], reverse=True)

    # unzip the sorted list of tuples back into two separate lists
    keys, values = zip(*zipped_lists)
    
    # Plot the proportions
    fig, ax = plt.subplots(1,1, figsize=(20,15))
    
    ax.bar(keys[:quantity], values[:quantity], align='center')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(keys[:quantity], rotation=20, ha='right')
    ax.set_title('Class distribution in the dataset')
    
    # save the figure to file
    fig.savefig(f'./results/{dataset.DATASET}_class_representation_first.png')   
    plt.close(fig)
    
    # Plot the proportions
    fig, ax = plt.subplots(1,1, figsize=(20,15))
    
    ax.bar(keys[-quantity:], values[-quantity:], align='center')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(keys[-quantity:], rotation=20, ha='right')
    ax.set_title('Class distribution in the dataset')
    
    # save the figure to file
    fig.savefig(f'./results/{dataset.DATASET}_class_representation_last.png')   
    plt.close(fig)


def visualize_data_items(opt: Opt, dataset, quantity=2):
    
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    
    # Plot images
    for i in range(quantity):
        
        # Example image with random index
        index = np.random.randint(low=0, high=40000)
        image, embed, text = dataset.__getitem__(index=index, return_text=True)
        
        opt.logger.info(f'Embedding Shape: {embed.size()}')

        # Display example
        opt.logger.info('Text: ' + text)
        ax[i].set_title(text)
            
        if opt.pytorch_cuda.available:
            image = image.cpu()
        
        ax[i].imshow(image.permute(1, 2, 0))
    
    file_path=os.path.join(opt.base['PATH_BASE_DIR'], f'./results/imagen_data_item_{dataset.DATASET}.png')
    
    # save the figure to file
    fig.savefig(file_path)   
    plt.close(fig)
    

def main():
    opt=Opt()

    #
    dataset = get_train_valid_ds(opt=opt, testing=True)
    
    #
    visualize_data_items(opt=opt, dataset=dataset, quantity=2)
    
    #
    visualize_class_representation(opt=opt, dataset=dataset)
    
    
    
    
 
    
if __name__ == "__main__":
    main()
    