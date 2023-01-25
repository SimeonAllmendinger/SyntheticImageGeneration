import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt

def cosine_similarity(A, B):
    input1 = A
    input2 = B
    output = torch.nn.functional.cosine_similarity(input1, input2)
    #print('output:', output)
    
    return output

def sum_abs_cosine_similarity(A, B):
    
    abs_sum = 0

    cos_sim = cosine_similarity(A=A, B=B)

    for i in cos_sim:
        abs_sum += abs(i)

    #print('abs_sum:',abs_sum)
    return abs_sum

def main():
    opt = Opt()

    triplets_unique_list = pd.read_json(opt.imagen['dataset']['PATH_TRIPLETS_DF_FILE'])['triplet_text'].unique().tolist()
    # Appnd domain-strange texts
    # Get t5 nlp embeddings
    triplet_embeds = torch.load(opt.imagen['dataset']['PATH_TEXT_EMBEDDING_FILE']) #.reshape((551,35*768))
    
    n = 5
    
    #(551, 35, 768) --> (35, 768)
    # triplet_embeds[i].flatten() --> (26880,1)
    
    similarity_array = np.zeros((n,n))
    
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                 a = sum_abs_cosine_similarity(A=triplet_embeds[i], B=triplet_embeds[j])
                 similarity_array[i][j] = a
    
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(25,20))  # create figure & 1 axis
    ax.imshow(similarity_array, cmap='Blues', interpolation='None')
    ax.set_xticks(np.arange(n), labels=triplets_unique_list[:n])
    ax.set_yticks(np.arange(n), labels=triplets_unique_list[:n])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    fig.savefig('./results/similarity_1.png')   # save the figure to file
    plt.close(fig)    # close the figure window
    
if __name__ == '__main__':
    main()