import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split
from collections import Counter
from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt
from src.components.data_manager.imagen_dataset import CholecSeg8kImagenDataset, CholecT45ImagenDataset, ConcatImagenDataset
from src.components.data_manager.dalle2_dataset import CholecSeg8kDalle2Dataset, CholecT45Dalle2Dataset, ConcatDalle2Dataset

class BaseDataLoader(DataLoader):
    
    def __init__(self, opt: Opt, dataset):
        
        super().__init__(dataset,
                         batch_size=opt.conductor['trainer']['batch_size'], 
                         shuffle=opt.conductor['trainer']['shuffle'])
        
    
def get_cholect45_dataset(opt: Opt):
        
        if opt.conductor['model']['model_type'] == 'Imagen':
            return CholecT45ImagenDataset(opt=opt)
        elif opt.conductor['model']['model_type'] == 'ElucidatedImagen':
            return CholecT45ImagenDataset(opt=opt)
        elif opt.conductor['model']['model_type'] == 'Dalle2':
            return CholecT45Dalle2Dataset(opt=opt)


def get_cholecseg8k_dataset(opt: Opt):
        
    if opt.conductor['model']['model_type'] == 'Imagen':
        return CholecSeg8kImagenDataset(opt=opt)
    elif opt.conductor['model']['model_type'] == 'ElucidatedImagen':
        return CholecSeg8kImagenDataset(opt=opt)
    elif opt.conductor['model']['model_type'] == 'Dalle2':
       return CholecSeg8kDalle2Dataset(opt=opt)


def get_concat_dataset(opt: Opt):
        
    if opt.conductor['model']['model_type'] == 'Imagen':
        return ConcatImagenDataset(opt=opt)
    elif opt.conductor['model']['model_type'] == 'ElucidatedImagen':
        return ConcatImagenDataset(opt=opt)
    elif opt.conductor['model']['model_type'] == 'Dalle2':
        return ConcatDalle2Dataset(opt=opt)
        

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
    if opt.datasets['data']['dataset'] == 'CholecT45' or (testing and opt.conductor['testing']['only_triplets']):
        imagen_dataset = get_cholect45_dataset(opt=opt)
    
    elif opt.datasets['data']['dataset'] == 'CholecSeg8k':
        imagen_dataset = get_cholecseg8k_dataset(opt=opt)
    
    elif opt.datasets['data']['dataset'] == 'Both':
        imagen_dataset = get_concat_dataset(opt=opt)
    
    if testing:
        
        return imagen_dataset
    
    else:
        
        # Split the instantiated dataset into training and validation datasets
        train_valid_split=[opt.conductor['trainer']['train_split'], opt.conductor['trainer']['valid_split']]
        train_dataset, valid_dataset = random_split(dataset=imagen_dataset, lengths=train_valid_split)
        
        # Return the training and validation datasets
        return train_dataset, valid_dataset
       

def get_train_valid_dl(opt: Opt, train_dataset, valid_dataset):
    
    train_generator = BaseDataLoader(opt=opt, dataset=train_dataset)
    valid_generator = BaseDataLoader(opt=opt, dataset=valid_dataset)
    
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
    
    fig, ax = plt.subplots(quantity,1, figsize=(6, quantity*4))
    
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
