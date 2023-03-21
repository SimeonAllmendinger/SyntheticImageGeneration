import torch
import numpy as np
import clip
from x_clip import CLIP
from tqdm import tqdm
from dalle2_pytorch import OpenAIClipAdapter, DiffusionPrior, DiffusionPriorNetwork
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPTokenizer

a=torch.load('src/assets/data/CholecT45/clip_embeds/image_embeds/image_batch_00001.pt')
b=torch.load('src/assets/data/CholecT45/clip_embeds/text_embeds/text_batch_00001.pt')


clip = OpenAIClipAdapter()

# mock data
prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

# diffusion prior network, which contains the CLIP and network (with transformer) above

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip=clip,
    timesteps = 100,
    cond_drop_prob = 0.2,
    condition_on_text_encodings = True  # this probably should be true, but just to get Laion started
).cuda()

text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# precompute the text and image embeddings
# here using the diffusion prior class, but could be done with CLIP alone

clip_image_embeds = diffusion_prior.clip.embed_image(images).image_embed
clip_text_embeds = diffusion_prior.clip.embed_text(text).text_embed

# feed text and images into diffusion prior network

loss = diffusion_prior(
    text_embed = clip_text_embeds,
    image_embed = clip_image_embeds
)

loss.backward()