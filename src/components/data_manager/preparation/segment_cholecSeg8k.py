import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from src.components.utils.opt.build_opt import Opt

def create_single_segment_frames(opt: Opt, mask_path: str, image_path: str, crop_transformer: T.Compose):
    """
    Creates segment frames for a given image and its mask image. The segment frames are created for the classes
    defined in the `opt` argument and saved to disk as PNG files.

    Parameters
        opt: Opt
            An object containing configuration information, including the classes to create segments for.
        mask_path: str
            The file path to the mask image.
        image_path: str
            The file path to the original image.
        crop_transformer: T.Compose
            torchvision transformer for resizing and cropping

    Returns
        classes: set
            A set of unique class names for which segment frames were created.
    """
    
    # Open the images
    color_mask = Image.open(mask_path)
    image = Image.open(image_path)
    
    # Crop images to image size
    color_mask = crop_transformer(color_mask)
    image = crop_transformer(image)
    
    # Convert images to numpy arraysv
    color_mask = np.array(color_mask)[:,:,:3]
    image = np.array(image)[:,:,:3]
    
    # Create a set to store unique classes e.g. liver, gallbladder etc.
    classes = set()

    # Loop through each class
    for key, value in opt.datasets['data']['CholecSeg8k']['classes'].items():
        
        # Skip the white frame and black background classes
        if key in ['white frame', 'black background']:
            continue
        
        # Create a mask for the current class
        mask = get_mask(color_mask=color_mask,
                        segment_hex_codes=value)

        # Create a file path for the current class segment frame
        file_path = image_path.split('.png')[0] + f'_{key}.png'

        # Check if the created mask contains any values greater than zero
        if mask.any() > 0:
            
            # Create the segment frame by applying the mask to the original image
            segment_image = image * mask
            segment_image = Image.fromarray(segment_image)

            # Save the segment frame to disk as a PNG file
            segment_image.save(file_path)

            # Add the current class name to the set of unique class names
            classes.add(key)

            # Log a message indicating that the segment frame was created successfully
            opt.logger.debug(f'Segment image created for {key} at {file_path}')
            
        else:
            
            # If the segment frame file exists but does not contain any data, remove it
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Return the set of unique class names for which segment frames were created
    return classes
                

def create_multi_segment_frames(opt: Opt, image_path: str, segments: list, segment_name: str):

    #
    image_size=opt.datasets['data']['image_size']
    image = np.zeros((image_size, image_size, 3))
    
    #
    for segment in segments:
        
        #
        segment_path = image_path.replace('.png',f'_{segment}.png')
        segment_image = np.array(Image.open(segment_path))
        
        # Create the segment frame by applying the mask to the original image
        image += segment_image
    
    #
    image = Image.fromarray(image.astype('uint8'))

    # Save the segment frame to disk as a PNG file
    image_path=image_path.replace('.png',f'_{segment_name}.png')
    image.save(image_path)

            
def get_mask(color_mask: np.array, segment_hex_codes: list):
    """
    Creates a mask for the given image based on the given hexadecimal color codes.

    Parameters:
        image_mask (np.array): The original image.
        segment_hex_codes (list): A list of hexadecimal color codes for each segment.
       
    Returns:
        tuple: The mask as a 3-channel Numpy array and the set of all RGB color codes in the image.
    """

    # Split the hexadecimal color codes into separate arrays for R, G, and B codes
    r_codes, g_codes, b_codes = zip(*[tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5)) for hex_code in segment_hex_codes])

    # Create a new image using the mask and the
    mask = np.zeros_like(color_mask)
    
    for i in range(len(segment_hex_codes)):
        target_rgb = np.array([r_codes[i], g_codes[i], b_codes[i]],dtype=np.uint8)
        
        # Create a boolean mask that is True where the image equals the target RGB value
        indices = np.all(color_mask == target_rgb, axis=-1)
        
        # target RGB value
        mask[indices] = np.array([1,1,1])
    
    return mask.astype('uint8')


def main():
    """
    Main function of the script.
    """
    
    # Initialize the Opt class
    opt=Opt()
    
    # Get all image paths with the pattern './data/CholecSeg8k/*/*/*_endo.png' and sort them
    image_paths = sorted(glob.glob('./data/CholecSeg8k/*/*/*_endo.png'))
    
    # Get all mask paths with the pattern './data/CholecSeg8k/*/*/*_color_mask.png' and sort them
    mask_paths = sorted(glob.glob('./data/CholecSeg8k/*/*/*_color_mask.png'))
    
    # Create a set to store all unique classes or segments
    all_single_classes=set()
    all_multi_classes=set()
    
    # Create CenterCrop transformer with image ssize
    image_size=opt.datasets['data']['image_size']
    transformer = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
        ])
    
    # Loop over the mask paths
    for i in tqdm(range(0, len(mask_paths))):

        # Call the create_single_segment_frames function and get the set of classes for the current image
        single_classes = create_single_segment_frames(opt=opt, mask_path=mask_paths[i],
                                                      image_path=image_paths[i],
                                                      crop_transformer=transformer)

        for key, value in opt.datasets['data']['CholecSeg8k']['multi_classes'].items():
            if set(value).issubset(single_classes):

                # Call the create_multi_segment_frames function
                create_multi_segment_frames(opt=opt,
                                            image_path=image_paths[i],
                                            segments=value,
                                            segment_name=key)
                
                all_multi_classes.add(key)

        # Update the set of all classes with the classes of the current image
        all_single_classes.update(single_classes)

    # Log the number of detected classes or segments
    opt.logger.info(f'Number of detected single classes or segments: {len(all_single_classes)}')
    opt.logger.info(f'Number of detected multi classes or segments: {len(all_multi_classes)}')
    
    # Log all detected classes or segments 
    opt.logger.debug(f'All detected single classes or segments: {all_single_classes}')
    opt.logger.debug(f'All detected multi classes or segments: {all_multi_classes}')


if __name__ == '__main__':
    main()