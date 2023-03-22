import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import clip
import numpy as np
import open_clip
from x_clip import CLIP
from tqdm import tqdm
from dalle2_pytorch import OpenClipAdapter, DiffusionPrior, DiffusionPriorNetwork, DiffusionPriorTrainer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPTokenizer
from src.components.data_manager.dataset_handler import get_train_valid_ds, get_train_valid_dl
from src.components.utils.opt.build_opt import Opt

opt=Opt()

class OpenClipAdapterWithContextLength(OpenClipAdapter):
    def __init__(self, opt: Opt, name: str, pretrained: str):
        self.context_length = opt.dalle2['clip']['context_length']
        super().__init__(name=name, pretrained=pretrained)
        
    @property
    def max_text_len(self):
        return self.context_length

clip_model = OpenClipAdapterWithContextLength(opt=opt,
                                              name=opt.dalle2['clip']['model_name'], 
                                              pretrained=opt.dalle2['clip']['pretrained'])

t_ds = get_train_valid_ds(opt=opt, testing=True)
dl = get_train_valid_dl(opt=opt, train_dataset=t_ds)

diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

diffusion_prior = DiffusionPrior(net=diffusion_prior_network,
                                 clip=clip_model,
                                 **opt.dalle2['diffusion_prior']['params']
                                 ).cuda()

diffusion_prior_trainer = DiffusionPriorTrainer(diffusion_prior=diffusion_prior,
                                                **opt.dalle2['diffusion_prior_trainer']['params']
                                                )

image_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
text_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR'])
        
for i, (image_batch, text_enc_batch, *_) in enumerate(tqdm(dl, disable=False)):
    #
    clip_image_embeds = diffusion_prior.clip.embed_image(image_batch).image_embed
    torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_embeds_{i:05d}.pt')

    #
    clip_text_embeds = diffusion_prior.clip.embed_text(text_enc_batch).text_embed
    torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_embeds_{i:05d}.pt')
    




#text = torch.randint(0, 49408, (4, 256)).cuda()
#images = torch.randn(4, 3, 256, 256).cuda()

# precompute the text and image embeddings
# here using the diffusion prior class, but could be done with CLIP alone

#clip_image_embeds = diffusion_prior.clip.embed_image(images).image_embed
#lip_text_embeds = diffusion_prior.clip.embed_text(text).text_embed

# feed text and images into diffusion prior network

'''loss = diffusion_prior(
    text_embed = clip_text_embeds,
    image_embed = clip_image_embeds
)

loss.backward()'''