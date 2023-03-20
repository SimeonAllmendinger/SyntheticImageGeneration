import torch
import numpy as np
import clip
from x_clip import CLIP
from tqdm import tqdm
from dalle2_pytorch import OpenAIClipAdapter
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPTokenizer

'''MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
#tokenizer = CLIPTokenizer(merges_path=MERGES_FILE, encoder_json_path=ENCODER_FILE)

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8,
    visual_patch_dropout = 0.5,             # patch dropout probability, used in Kaiming He's FLIP to save compute and improve end results - 0.5 is good value, 0.75 on high end is tolerable
    use_all_token_embeds = False,           # whether to use fine-grained contrastive learning (FILIP)
    decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
    extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_visual_ssl = True,                  # whether to do self supervised learning on iages
    use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
    text_ssl_loss_weight = 0.05,            # weight for text MLM loss
    image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
)'''

# mock data
print('hello')
clip = OpenAIClipAdapter('ViT-B/32').cuda()
tokenizer=CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

images = torch.randn(4, 3, 256, 256).cuda()
text=["grasper grasp gallbladder", "grasper retract gallbladder", 'jdjdj', 'jshsgsg']
tokens = tokenizer(text=text, 
              return_tensors="pt", 
              padding='max_length', 
              max_length=256, 
              truncation=True)

text = tokens.input_ids.cuda()
print(text)
print(text.size())
print(text.min(), text.max())


# train

for i in tqdm(range(10)):
    loss = clip(
        text,
        images,
        #return_loss = True              # needs to be set to True to return contrastive loss
    )

    loss.backward()
    print(loss)