import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import glob
import yaml
import json
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from os.path import exists as file_exists
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from cleanfid import fid as clean_fid

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.imagen.model.build_imagen import Imagen_Model
from src.components.data_manager.dataset_handler import get_train_valid_ds, get_train_valid_dl
from src.components.dalle2.model.build_dalle2 import Dalle2_Model

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to uerib@student.kit.edu')

parser.add_argument('--path_data_dir',
                    default='/home/kit/stud/uerib/SyntheticImageGeneration',
                    help='PATH to data directory')


def test_text2images(opt: Opt, 
                     sample_dataloader, 
                     model, 
                     unet_number: int, 
                     sample_quantity: int,
                     save_samples: bool,
                     save_image_tensors: bool,
                     sample_folder: str,
                     embed_shape: tuple,
                     tqdm_disable=False,
                     text_list=None
                     ):
        
    # Define the folder path to save synthetic and real images as .pt and .png
    cond_scale = opt.conductor['testing']['cond_scale']
    seed = opt.conductor['testing']['sample_seed']
    loss_weighting = opt.conductor['testing']['loss_weighting']
    
    #
    model_type = opt.conductor['model']['model_type']
    
    #
    if model_type == 'Imagen':
        real_image_save_path = sample_folder + f"imagen/real_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        synthetic_image_save_path = sample_folder + f"imagen/synthetic_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        imagen_model = model
    
    elif model_type == 'ElucidatedImagen':
        real_image_save_path = sample_folder + f"elucidated_imagen/real_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        synthetic_image_save_path = sample_folder + f"elucidated_imagen/synthetic_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        imagen_model = model
        
    elif model_type == 'Dalle2':
        real_image_save_path = sample_folder + f"dalle2/real_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        synthetic_image_save_path = sample_folder + f"dalle2/synthetic_images/cond_scale_{cond_scale}_dtp95_{loss_weighting}/{seed:02d}/"
        dalle2_model = model
    
    #
    lower_batch = opt.conductor['testing']['lower_batch']
    upper_batch = opt.conductor['testing']['upper_batch']
    
    #
    if text_list is None:
        if save_image_tensors or save_samples:
        
            #
            opt.logger.info('Start Sampling Loop')
            for i, (image_batch, embed_batch, text_batch) in enumerate(tqdm(sample_dataloader, disable=tqdm_disable)):
                
                if i < lower_batch or i > upper_batch:
                    continue
                
                if i >= int(sample_quantity / opt.conductor['trainer']['batch_size']):
                    break
                
                if model_type == 'Imagen' or model_type == 'ElucidatedImagen':
        
                    # sample an image based on the text embeddings from the cascading ddpm
                    synthetic_image_batch = imagen_model.trainer.sample(text_embeds=embed_batch.cuda(),
                                                                        return_pil_images=False,
                                                                        stop_at_unet_number=unet_number,
                                                                        # use_tqdm=not tqdm_disable,
                                                                        cond_scale=opt.conductor['testing']['cond_scale'])
                elif model_type == 'Dalle2':
                    synthetic_image_batch = dalle2_model(text_batch,
                                                        # classifier free guidance strength (> 1 would strengthen the condition)
                                                        cond_scale=opt.conductor['testing']['cond_scale'],
                                                        return_pil_images=False
                                                        )

                #
                opt.logger.info('Clamp Tensors')
                synthetic_image_batch = torch.clamp(synthetic_image_batch.cuda(), min=0, max=1)
                real_image_batch = torch.clamp(image_batch.cuda(), min=0, max=1)
                #embed_batch = torch.clamp(embed_batch, min=0, max=1)

                if save_image_tensors:
                    opt.logger.info('Save Tensors')
                    # Save image and embed tensors
                    torch.save(synthetic_image_batch, synthetic_image_save_path + f'{i:05d}_synthetic_image_batch.pt')
                    torch.save(real_image_batch, real_image_save_path + f'{i:05d}_real_image_batch.pt')
                    #torch.save(embed_batch, sample_folder + f'{i:05d}_image_batch.pt')
                
                    # Save texts
                    with open(real_image_save_path + f'{i:05d}_text_batch.json', "w") as f:
                        json.dump(text_batch, f)
                
                if save_samples:
                    # Save real images
                    save_images(opt=opt, 
                                images=real_image_batch, 
                                text_batch=text_batch, 
                                batch_size=embed_shape[0], 
                                unet_number=unet_number, 
                                sample_folder=real_image_save_path, 
                                i=i)
                    
                    save_images(opt=opt, 
                                images=synthetic_image_batch, 
                                text_batch=text_batch, 
                                batch_size=embed_shape[0], 
                                unet_number=unet_number, 
                                sample_folder=synthetic_image_save_path,
                                i=i)
            
            #
            opt.logger.debug(f'real_image_batch_size: {real_image_batch.size()}')
            opt.logger.debug(f'real_image_batch min | max: {real_image_batch.min()} | {real_image_batch.max()}')
            
            #
            opt.logger.debug(f'synthetic_image_batch_size: {synthetic_image_batch.size()}')
            opt.logger.debug(f'synthetic_image_batch: min | max {synthetic_image_batch.min()} | {synthetic_image_batch.max()}')

        #
        if opt.conductor['testing']['FrechetInceptionDistance']['usage']:
            fid = FrechetInceptionDistance(**opt.conductor['testing']['FrechetInceptionDistance']['params']).cuda()
            opt.logger.info('FID initialized')
        
        #
        if opt.conductor['testing']['KernelInceptionDistance']['usage']:
            kid = KernelInceptionDistance(**opt.conductor['testing']['KernelInceptionDistance']['params']).cuda()
            opt.logger.info('KID initialized')
        
        #
        if opt.conductor['testing']['LearnedPerceptualImagePatchSimilarity']['usage']:
            lpips = LearnedPerceptualImagePatchSimilarity(**opt.conductor['testing']['LearnedPerceptualImagePatchSimilarity']['params']).cuda()
            lpips_batch_score_list = []
            opt.logger.info('lpips initialized')
            
        #
        if opt.conductor['testing']['FrechetInceptionDistance']['usage'] or opt.conductor['testing']['KernelInceptionDistance']['usage'] or opt.conductor['testing']['LearnedPerceptualImagePatchSimilarity']['usage']:
            
            opt.logger.info('Start updating FID / KID')
            
            real_image_path_list = glob.glob(real_image_save_path + '*_real_image_batch.pt')
            synthetic_image_path_list = glob.glob(synthetic_image_save_path + '*_synthetic_image_batch.pt')
            
            for real_path, synthetic_path in tqdm(zip(real_image_path_list, synthetic_image_path_list), 
                                                total=len(real_image_path_list),
                                                disable=tqdm_disable):
                
                real_image_batch = torch.load(real_path).cuda()
                synthetic_image_batch = torch.load(synthetic_path).cuda()

                # Update FID
                if opt.conductor['testing']['FrechetInceptionDistance']['usage']:
                    fid.update(real_image_batch, real=True)
                    fid.update(synthetic_image_batch, real=False)
                
                # Update KID
                if opt.conductor['testing']['KernelInceptionDistance']['usage']:
                    kid.update(real_image_batch, real=True)
                    kid.update(synthetic_image_batch, real=False)
                    
                # Update LPIPS
                if opt.conductor['testing']['LearnedPerceptualImagePatchSimilarity']['usage']:
                    real_image_batch = rescale_tensor(real_image_batch)
                    synthetic_image_batch = rescale_tensor(synthetic_image_batch)
                    lpips_batch_score_list.append(lpips(real_image_batch, synthetic_image_batch).tolist())
            
        
        # Compute FID
        if opt.conductor['testing']['FrechetInceptionDistance']['usage']:
            fid_result = fid.compute().cpu().item()
            opt.logger.info(f'fid_result: {fid_result}')
        else:
            fid_result = None
        
        # Compute KID
        if opt.conductor['testing']['KernelInceptionDistance']['usage']:
            kid_mean, kid_std = kid.compute()
            kid_mean, kid_std = (kid_mean.cpu().item(), kid_std.cpu().item())
            opt.logger.info(f'kid_result: {kid_mean}, {kid_std}')
        else:
            kid_mean, kid_std = [None, None]
        
        #
        if opt.conductor['testing']['CleanFID']['usage']:
            fdir1 = real_image_save_path + 'images/'
            fdir2 = synthetic_image_save_path + 'images/'
            clean_fid_score = clean_fid.compute_fid(fdir1=fdir1,
                                                    fdir2=fdir2,
                                                    **opt.conductor['testing']['CleanFID']['params'])
            opt.logger.info(f'clean_fid_score: {clean_fid_score}')
        else:
            clean_fid_score = None

        # FCD
        if opt.conductor['testing']['FrechetCLIPDistance']['usage']:
            fdir1 = real_image_save_path + 'images/'
            fdir2 = synthetic_image_save_path + 'images/'
            fcd_score = clean_fid.compute_fid(fdir1=fdir1,
                                            fdir2=fdir2,
                                            **opt.conductor['testing']['FrechetCLIPDistance']['params'])
            opt.logger.info(f'fcd_score: {fcd_score}')
        else:
            fcd_score = None
        
        # LPIPS
        if opt.conductor['testing']['LearnedPerceptualImagePatchSimilarity']['usage']:
            lpips_mean = np.mean(lpips_batch_score_list)
            opt.logger.info(f'lpips_mean: {lpips_mean}')
            pd.DataFrame({'cond_scale': cond_scale,
                          'Dataset': opt.datasets['data']['dataset'],
                          'Model': model_type,
                          'lpips': lpips_batch_score_list}).to_json(
                synthetic_image_save_path + timestamp + '_lpips.json')
        else:
            lpips_mean = None
            
        #
        results = pd.DataFrame({'cond_scale': cond_scale,
                                'Dataset': opt.datasets['data']['dataset'],
                                'Model': model_type,
                                'FID': [fid_result],
                                'KID_mean': [kid_mean],
                                'KID_std': [kid_std],
                                'Clean_FID': [clean_fid_score],
                                'FCD': [fcd_score],
                                'LPIPS': [lpips_mean]
                                })

        #
        timestamp = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        results.to_csv(synthetic_image_save_path + timestamp + '_results.csv')

        return results
        
    #
    else:
        
        batch_size = opt.conductor['trainer']['batch_size']
        
        if opt.conductor['testing']['cond_scale'] == 3:
            save_real_images(opt=opt, real_image_save_path=real_image_save_path)
        
        for i in range(int(np.ceil(len(text_list)/100))):
            
            if i < lower_batch or i > upper_batch:
                continue
            
            if (i+1)*batch_size >= len(text_list):
                text_list_batch = text_list[i*batch_size:]
            else:
                text_list_batch = text_list[i*batch_size:(i+1)*batch_size]
                
            if model_type == 'Imagen' or model_type == 'ElucidatedImagen':
                # sample an image based on the text embeddings from the cascading ddpm
                synthetic_images = imagen_model.trainer.sample(texts=text_list_batch,
                                                            return_pil_images=True,
                                                            stop_at_unet_number=unet_number,
                                                            use_tqdm=not tqdm_disable,
                                                            cond_scale=opt.conductor['testing']['cond_scale'])
            
            elif model_type == 'Dalle2':
                #
                synthetic_image_batch = dalle2_model(text_list_batch,
                                                    # classifier free guidance strength (> 1 would strengthen the condition)
                                                    cond_scale=opt.conductor['testing']['cond_scale'],
                                                    return_pil_images=True
                                                    )

            for j, synthetic_image in enumerate(synthetic_images):
                image_index = batch_size * i + j
                image_save_path_dir = synthetic_image_save_path + f'evaluation/{text_list[image_index]}/'
                
                if not file_exists(image_save_path_dir):
                    os.mkdir(image_save_path_dir)
                
                image_save_path = os.path.join(image_save_path_dir, f"{image_index:05d}-{opt.conductor['model']['model_type']}-{text_list[image_index]}.png")
                synthetic_image.save(image_save_path)
            
                
def save_real_images(opt: Opt, real_image_save_path: str):
    
    text_dict_index = opt.conductor['testing']['text_dict_index']
    text_dict = np.zeros(100)
    text_dict[text_dict_index] = 1
    
    folder=os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['PATH_VIDEO_DIR'])
    transform = T.Compose([
            T.Resize(opt.datasets['data']['image_size']),
            T.CenterCrop(opt.datasets['data']['image_size'])
        ])
    
    n = 0
    
    for video_index in tqdm(range(1,81)):
        try:
            triplet_labels = np.loadtxt(f'/home/kit/stud/uerib/SyntheticImageGeneration/data/CholecT45/triplet/VID{video_index:02d}.txt', dtype=np.int32, delimiter=',')
            phase_labels = pd.read_csv(f'/home/kit/stud/uerib/SyntheticImageGeneration/data/Cholec80/phase_annotations/video{video_index:02d}-phase.txt', sep='\t')
        except FileNotFoundError:
            continue
        
        triplet_indices = np.where(np.all(triplet_labels[:, 1:]==text_dict,axis=1))[0]
        
        for image_index in triplet_indices:
            if phase_labels.iloc[image_index*25,1] != opt.conductor['testing']['phase_name']:
                continue
            
            path = os.path.join(folder, f'VID{video_index:02d}/{int(image_index):06d}.png')
            image = Image.open(path)

            # Transform Image
            image = transform(image)
            
            # Save Image
            image_save_path_dir = real_image_save_path + f"evaluation/{opt.conductor['testing']['text']}/"
            
            if not file_exists(image_save_path_dir):
                os.mkdir(image_save_path_dir)
            
            image_save_path = os.path.join(image_save_path_dir, f"{n:05d}-{opt.conductor['model']['model_type']}-{opt.conductor['testing']['text']}.png")
            image.save(image_save_path)
            
            # Update Counter
            n+=1               
            if n >= opt.conductor['testing']['sample_quantity']:
                break
        
        # Update Counter                
        if n >= 1000:
            break
            
        
def save_images(opt: Opt, 
                images, 
                text_batch: list,
                i: int, 
                batch_size: int, 
                unet_number: int,
                sample_folder: str):           
    
    
    for j, image in tqdm(enumerate(images)):
        
        # 
        image = image.cpu().permute(1,2,0)
        image = image.numpy() * 255
        image = Image.fromarray(image.astype(np.uint8))
        
        image_index = i*batch_size + j
        image_save_path = sample_folder + f'images/{image_index:05d}_u{unet_number}-{text_batch[j]}.png'
        image.save(image_save_path)
        
        #
        opt.logger.debug(f'Created image for {text_batch[j]} at {image_save_path}.') 
        

def rescale_tensor(x):
    y = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)) - 1
    return y


def main():
    
    #
    with open('configs/config_datasets.yaml') as f:
        config = yaml.safe_load(f)

    args = parser.parse_args()
    config['PATH_DATA_DIR'] = args.path_data_dir

    with open('configs/config_datasets.yaml', 'w') as f:
        yaml.dump(config, f)
        
    #
    opt=Opt()
    
    # Start run with neptune docs
    neptune_ai = Neptune_AI(opt=opt)
    neptune_ai.start_neptune_run(opt)
    #
    sample_quantity=opt.conductor['testing']['sample_quantity']
    unet_number=opt.conductor['testing']['unet_number']
    save_samples=opt.conductor['testing']['save_samples']
    save_image_tensors=opt.conductor['testing']['save_image_tensors']
    model_type=opt.conductor['model']['model_type']
    
    if opt.conductor['testing']['text'] != '':
        text = opt.conductor['testing']['text']
        text_list=[text] * sample_quantity
    else:
        text_list=None
    
    #
    imagen_dataset = get_train_valid_ds(opt=opt, testing=True, return_text=True)
    imagen_dataloader = get_train_valid_dl(opt=opt, train_dataset=imagen_dataset)
    _, sample_text_embed, _ = imagen_dataset.__getitem__(index=0)
    
    
    if model_type == 'Imagen' or model_type == 'ElucidatedImagen':
        model= Imagen_Model(opt=opt, testing=True)
    elif model_type == 'Dalle2':
        model= Dalle2_Model(opt=opt, testing=True)
        
    # Make results test folder with timestamp
    test_sample_folder = os.path.join(
        '/home/kit/stud/uerib/SyntheticImageGeneration/', opt.conductor['testing']['PATH_TEST_SAMPLE'])

    #
    results = test_text2images(opt=opt,
                               sample_dataloader=imagen_dataloader,
                               model=model,
                               unet_number=unet_number,
                               sample_quantity=sample_quantity,
                               save_samples=save_samples,
                               save_image_tensors=save_image_tensors,
                               sample_folder=test_sample_folder,
                               embed_shape=sample_text_embed.shape,
                               tqdm_disable=False,
                               text_list=text_list
                               )

    #
    opt.logger.info(f'FRECHET INCEPTION DISTANCE (FID): {results["FID"]}')
    opt.logger.info(f'KERNEL INCEPTION DISTANCE (KID): mean {results["KID_mean"]} | std {results["KID_std"]}')
    opt.logger.info(f'CLEAN - FRECHET INCEPTION DISTANCE (clean-fid): {results["Clean_FID"]}')
    opt.logger.info(f'FRECHET CLIP DISTANCE (FCD): {results["FCD"]}')  

    neptune_ai.stop_neptune_run(opt=opt)

if __name__ == "__main__":
    main()

