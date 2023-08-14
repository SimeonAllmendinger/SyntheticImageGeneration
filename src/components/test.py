import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import argparse

from imagen_pytorch import load_imagen_from_checkpoint

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to simeon.allmendinger@fim-rc.de')

parser.add_argument('--model',
                    default='ElucidatedImagen',
                    help='Please choose: Imagen or ElucidatedImagen')
parser.add_argument('--text',
                    default='grasper grasp gallbladder in callot triangle dissection',
                    help='Please write a text prompt')

args = parser.parse_args()

if args.model == 'ElucidatedImagen':
    model = load_imagen_from_checkpoint('src/assets/elucidated_imagen/models/elucidated_imagen_model_u2_p2_dtp95_T45.pt')
    
elif args.model == 'Imagen':
    model = load_imagen_from_checkpoint('src/assets/imagen/models/imagen_model_u2_p2_dtp95_T45.pt')
    
images = model.sample(texts = [args.text], cond_scale = 3., return_pil_images=True)
images[0].save(f'./test_image_{args.text}.png')