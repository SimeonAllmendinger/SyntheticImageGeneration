import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt

def create_segment_frames(opt: Opt, mask_path: str, image_path: str):
        
    # Open the image
    color_mask = Image.open(mask_path)
    image = np.array(Image.open(image_path))
    
    # Get the pixels in the image
    color_mask_pixels = color_mask.load()
    
    # Create a set to store unique color values
    colors = set()
    
    for key, value in opt.imagen['data']['CholecSeg8k']['classes'].items():
        
        mask, colors = get_mask(opt=opt, 
                                image_mask=color_mask, 
                                image_pixels=color_mask_pixels, 
                                segment_hex_codes=value, 
                                segment_class=key,
                                colors=colors)
        
        if np.any(mask != [0, 0, 0]) and key not in ['white frame', 'black background']:
            segment_image = image * mask
            segment_image=Image.fromarray(segment_image)
            segment_image.save(image_path.split('.png')[0] + f'_{key}.png')
                
                
def get_mask(opt: Opt, image_mask: Image, image_pixels, segment_hex_codes: list, segment_class: str, colors: set):
    
    
    # Create a new image with the same dimensions as the original image
    mask = Image.new("RGB", (image_mask.width, image_mask.height), (0, 0, 0))
    mask_pixels = mask.load()
        
    for hex_code in segment_hex_codes:
        # Convert the hexadecimal color code to RGB values
        r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))
    
        # Loop through each pixel in the image
        for i in range(image_mask.width):
            for j in range(image_mask.height):
                
                # Get the RGB value of the pixel
                if len(image_pixels[i, j]) > 3:
                    p_r, p_g, p_b = image_pixels[i, j][:3]
                
                else:
                    p_r, p_g, p_b = image_pixels[i, j]
                
                colors.add((p_r, p_g, p_b))
                
                # Check if the RGB value of the pixel matches the given hexadecimal color code
                if (p_r, p_g, p_b) == (r, g, b):
                    mask_pixels[i, j] = (r, g, b)
                    
    mask = np.array(mask.convert('L'))
    mask = np.where(mask>0,1,mask).reshape(*mask.shape, 1)
    mask_array = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(3):
        mask_array[:,:,i] = mask[:, :, 0]
    
    return mask_array.astype('uint8'), colors


def main():
    opt=Opt()
    
    image_paths = sorted(glob.glob('./data/CholecSeg8k/*/*/*_endo.png'))
    mask_paths = sorted(glob.glob('./data/CholecSeg8k/*/*/*_color_mask.png'))
    
    for i in tqdm(range(2637, len(mask_paths))):
        create_segment_frames(opt=opt, mask_path= mask_paths[i], image_path=image_paths[i])
    
    
if __name__ == '__main__':
    main()