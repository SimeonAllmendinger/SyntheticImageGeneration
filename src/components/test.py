import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import argparse
import torch

from imagen_pytorch import load_imagen_from_checkpoint

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to simeon.allmendinger@fim-rc.de')

parser.add_argument('--model',
                    default='ElucidatedImagen',
                    help='Please choose: Imagen or ElucidatedImagen.')
parser.add_argument('--text',
                    default='grasper grasp gallbladder in callot triangle dissection',
                    help='Please write a text prompt.')
parser.add_argument('--cond_scale',
                    default=3,
                    help='Please insert a conditioning scale between 1 and 10.')

args = parser.parse_args()

if args.model == 'ElucidatedImagen':
    model = load_imagen_from_checkpoint('src/assets/elucidated_imagen/models/elucidated_imagen_model_u2_p2_dtp95_T45.pt')
    
elif args.model == 'Imagen':
    model = load_imagen_from_checkpoint('src/assets/imagen/models/imagen_model_u2_p2_dtp95_T45.pt')
    
if torch.cuda.is_available():
    model=model.cuda()
    
images = model.sample(texts = [args.text], cond_scale = int(args.cond_scale), return_pil_images=True)
images[0].save(f'./test_image_{args.text}.png')