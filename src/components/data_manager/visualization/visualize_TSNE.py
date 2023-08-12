import numpy as np
import scienceplots
import torch
import glob
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

plt.style.use(['science'])

length=400
cond_scale_list=[3,5]
model_type='imagen'
text_list = ['grasper retract liver in gallbladder dissection', 'hook dissect gallbladder in gallbladder dissection', 'grasper retract gallbladder in calot triangle dissection']
#
def get_vector(image_tensor):

    my_embedding = torch.zeros((2048))

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)

    img=scaler(image_tensor).unsqueeze(0)
    model(img)

    h.remove()

    return my_embedding

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

scaler = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

fig,ax = plt.subplots(1,2,figsize=(12,6))

for k in range(2):
    
    cond_scale=cond_scale_list[k]
    embed_batch = torch.zeros((6*length,2048))  
    
    for j in range(3):
        print(k,j)
        text = text_list[j]
        real_path_list = glob.glob(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/testing/imagen/real_images/cond_scale_3_dtp95_p2/01/evaluation/{text}/*.png')
        synthetic_path_list = glob.glob(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/testing/{model_type}/synthetic_images/cond_scale_{cond_scale}_dtp95_p2/01/evaluation/{text}/*.png')

        for i, (real_path, synthetic_path) in enumerate(tqdm(zip(real_path_list,synthetic_path_list),
                                                            total=length)):
            real_image = Image.open(real_path)
            synthetic_image = Image.open(synthetic_path)
            
            embed_batch[i+j*length,:] = get_vector(real_image)
            embed_batch[i+(j+3)*length,:] = get_vector(synthetic_image)
            
            if i == length-1:
                break

    image_embedded = TSNE(n_components=2, learning_rate='auto',
                                    init='random', perplexity=30).fit_transform(embed_batch)

    for h in range(3):
        ax[k].scatter(image_embedded[h*length:(h+1)*length,0], image_embedded[h*length:(h+1)*length,1],
                marker='.',
                c='C'+str(h), 
                label=f'Real Images text prompt {h+1}')

        ax[k].scatter(image_embedded[(h+3)*length:(h+4)*length,0], image_embedded[(h+3)*length:(h+4)*length,1],
                marker='+',
                c='C'+str(h),
                label=f'Samples text prompt {h+1}')

    ax[k].set_title(f'TSNE Viusalization of ResNet50 Embeddings (cond_scale: {cond_scale})')
    handles, labels = ax[k].get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center',ncols=3)   
#fig.savefig(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/TSNE/TSNE_{model_type}_{cond_scale}-{text}.png')
fig.savefig(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/TSNE/TSNE_{model_type}_{cond_scale}_10082023.png')