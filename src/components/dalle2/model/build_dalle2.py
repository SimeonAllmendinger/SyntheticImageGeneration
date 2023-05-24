import sys
import os

sys.path.append(os.path.abspath(os.curdir))

import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from os.path import exists as file_exists
from dalle2_pytorch import (
    DALLE2,
    DiffusionPriorNetwork,
    DiffusionPrior,
    Decoder,
    CLIP,
    Unet,
    OpenClipAdapter,
    DecoderTrainer,
    DiffusionPriorTrainer,
)

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds
from src.components.dalle2.training.train_dalle2 import _train_decoder_
from src.components.utils.training.early_stopping import EarlyStopping

# Create an argument parser
parser = argparse.ArgumentParser(
    prog='SyntheticImageGeneration',
    description='Magic with Text2Image',
    epilog='For help refer to uerib@student.kit.edu'
)

# Add an argument for the path to the data directory
parser.add_argument(
    '--path_data_dir',
    default='/home/kit/stud/uerib/SyntheticImageGeneration/',
    help='PATH to data directory'
)

# Define a class that extends OpenClipAdapter to include context length
class OpenClipAdapterWithContextLength(OpenClipAdapter):
    
    def __init__(self, opt: Opt):
        self.context_length = opt.dalle2['clip']['context_length']
        super().__init__(
            name=opt.dalle2['clip']['model_name'],
            pretrained=opt.dalle2['clip']['pretrained']
        )

    @property
    def max_text_len(self):
        return self.context_length


# Define a Prior_Model class
class Prior_Model():
    
    def __init__(self, opt: Opt, testing=False):
        self.testing = testing

        if opt.conductor['trainer']['early_stopping']['usage']:
            self.loss_queue = EarlyStopping(opt_early_stopping=opt.conductor['trainer']['early_stopping'])

        self._init_clip_(opt=opt)
        self._create_diffusion_prior_(opt=opt)


    def _init_clip_(self, opt: Opt):
        self.clip = OpenClipAdapterWithContextLength(opt=opt).cuda()


    def _create_clip_embeds(
        self,
        opt: Opt,
        image_embeds_save_dir_path: str,
        text_embeds_save_dir_path: str
    ):
        opt.conductor['trainer']['shuffle'] = False
        dataset = get_train_valid_ds(opt=opt, return_text=False, testing=True, return_embeds=False)
        dataloader = get_train_valid_dl(opt=opt, train_dataset=dataset)

        opt.logger.info('Create text embeddings...')

        for i, (image_batch, text_enc_batch, *_) in enumerate(tqdm(dataloader, disable=False)):
            # Compute clip_image_embeds and save to file
            clip_image_embeds = self.diffusion_prior.clip.embed_image(image_batch.cuda()).image_embed
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_embeds_{i:05d}.pt')

            # Compute clip_text_embeds and save to file
            clip_text_embeds = self.diffusion_prior.clip.embed_text(text_enc_batch.cuda()).text_embed
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_embeds_{i:05d}.pt')

        opt.logger.debug('Training embeds created')
        
     
    def _create_diffusion_prior_(self, opt: Opt):
        # Create and initialize DiffusionPriorNetwork
        self.diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

        # Create DiffusionPrior using DiffusionPriorNetwork and CLIP
        self.diffusion_prior = DiffusionPrior(
            net=self.diffusion_prior_network,
            clip=self.clip,
            **opt.dalle2['diffusion_prior']['params']
        ).cuda()

        # Create DiffusionPriorTrainer
        self.diffusion_prior_trainer = DiffusionPriorTrainer(
            diffusion_prior=self.diffusion_prior,
            **opt.dalle2['diffusion_prior_trainer']['params']
        )

        # Load existing model if specified and available
        if file_exists(opt.dalle2['diffusion_prior_trainer']['model_save_path']) and opt.dalle2['diffusion_prior']['use_existing_model']:
            self.diffusion_prior_trainer.load(opt.dalle2['diffusion_prior_trainer']['model_save_path'])

        if not self.testing:
            # Define the save paths for image and text embeddings
            image_embeds_save_dir_path = os.path.join(
                opt.datasets['PATH_DATA_DIR'],
                opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR']
            )
            text_embeds_save_dir_path = os.path.join(
                opt.datasets['PATH_DATA_DIR'],
                opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR']
            )

            # Create clip embeddings if they don't exist already
            if not opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['use_existing_embeds']:
                self._create_clip_embeds(
                    opt=opt,
                    image_embeds_save_dir_path=image_embeds_save_dir_path,
                    text_embeds_save_dir_path=text_embeds_save_dir_path
                )

            # Train the diffusion prior
            self._train_diffusion_prior_(opt=opt)

            # Save the diffusion prior model
            self.diffusion_prior_trainer.save(opt.dalle2['diffusion_prior_trainer']['model_save_path'])


    def _train_diffusion_prior_(self, opt: Opt):
        # Get train and validation datasets
        train_dataset, valid_dataset = get_train_valid_ds(opt=opt, return_text=False, return_embeds=True)
        train_dl, valid_dl = get_train_valid_dl(opt=opt, train_dataset=train_dataset, valid_dataset=valid_dataset)

        # Start Neptune run
        neptune_ai = Neptune_AI(opt=opt)
        neptune_ai.start_neptune_run(opt)

        # Upload clip configs to Neptune
        neptune_ai.add_param_neptune_run(opt=opt, 
                                         data_item=opt.dalle2['diffusion_prior_network'], 
                                         neptune_run_save_path='diffusion_prior_network_configs')
        neptune_ai.add_param_neptune_run(opt=opt, 
                                         data_item=opt.dalle2['diffusion_prior'], 
                                         neptune_run_save_path='diffusion_prior_configs')
        neptune_ai.add_param_neptune_run(opt=opt, 
                                         data_item=opt.dalle2['diffusion_prior_trainer'], 
                                         neptune_run_save_path='diffusion_prior_trainer_configs')

        # Training loop
        for epoch_n in tqdm(range(1, opt.dalle2['diffusion_prior_trainer']['epochs'] + 1)):
            # Iterate over the training dataloader
            for i, (image_batch, text_encodings, clip_image_embeds, clip_text_embeds) in enumerate(tqdm(train_dl, disable=False)):
                # Check if training should be done with clip embeddings or text and image inputs
                if opt.dalle2['diffusion_prior_trainer']['train_with_embeds']:
                    # Train the diffusion prior using clip embeddings
                    loss = self.diffusion_prior_trainer(
                        text_embed=clip_text_embeds.cuda(),
                        image_embed=clip_image_embeds.cuda()
                    )
                else:
                    # Train the diffusion prior using text and image inputs
                    loss = self.diffusion_prior_trainer(
                        text=text_encodings.cuda(),
                        image=image_batch.cuda()
                    )

                # Update the diffusion prior trainer
                self.diffusion_prior_trainer.update()
                opt.logger.debug(f'Diffusion Prior loss in epoch {epoch_n} | {i}: {loss}')
                    
                # Upload epoch training loss to neptune_ai
                neptune_ai.log_neptune_run(
                    opt=opt,
                    data_item=loss,
                    neptune_run_save_path=f"training/loss_diffusion_prior"
                )
                
                # Check if early stopping should be applied
                if opt.conductor['trainer']['early_stopping']['usage']:
                    self.loss_queue.push(loss)

                    if self.loss_queue.stop:
                        opt.logger.info('Stop training early')
                        break
                    
        # Stop the Neptune run
        neptune_ai.stop_neptune_run(opt=opt)
        

class Dalle2_Model(DALLE2):
    
    def __init__(self, opt: Opt, testing=False):
        # Create an instance of the Prior_Model class
        prior = Prior_Model(opt=opt, testing=testing)
        
        # Set the testing attribute
        self.testing = testing
        
        # Create the decoder and decoder trainer
        decoder_trainer = self._create_decoder_(opt=opt, clip=prior.clip)

        # Call the superclass constructor
        super().__init__(prior=prior.diffusion_prior_trainer.diffusion_prior, decoder=decoder_trainer.decoder)
     
        
    def _create_decoder_(self, opt: Opt, clip: CLIP):
        # Create two instances of the Unet class
        unet1 = Unet(**opt.dalle2['unet1']).cuda()
        unet2 = Unet(**opt.dalle2['unet2']).cuda()
        
        # Create the decoder using the clip and unet models
        decoder = Decoder(
            clip=clip,
            unet=[unet1, unet2],  # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
            **opt.dalle2['decoder']['params']
        ).cuda()

        # Create the decoder trainer
        decoder_trainer = DecoderTrainer(
            decoder=decoder,
            **opt.dalle2['decoder_trainer']['params']
        )

        # Check if a saved model exists and whether to use it
        if file_exists(opt.dalle2['decoder_trainer']['model_save_path']) and opt.dalle2['decoder']['use_existing_model']:
            decoder_trainer.load(opt.dalle2['decoder_trainer']['model_save_path'])
            
        # Train the decoder if not in testing mode
        if not self.testing:
            decoder_trainer = _train_decoder_(opt=opt, decoder_trainer=decoder_trainer)
            decoder_trainer.save(opt.dalle2['decoder_trainer']['model_save_path'])
            
        return decoder_trainer
    

def main():
    # Load the configuration from the YAML file
    with open('configs/config_datasets.yaml') as f:
        config = yaml.safe_load(f)

    # Parse command-line arguments
    args = parser.parse_args()

    # Update the configuration with the provided data directory
    config['PATH_DATA_DIR'] = args.path_data_dir

    # Save the updated configuration back to the YAML file
    with open('configs/config_datasets.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create an instance of the Opt class
    opt = Opt()
    
    # Create an instance of the Dalle2_Model class
    dalle2 = Dalle2_Model(opt=opt, testing=False)
    
    # Create dataloaders for training and validation datasets
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt, return_text=True, return_embeds=False)
    train_dl, valid_dl = get_train_valid_dl(opt=opt, train_dataset=train_dataset, valid_dataset=valid_dataset)
    
    # Get a batch of images and text encodings from the dataloader
    image_batch, text_encodings, text_batch = next(iter(train_dl))
    
    # Generate images using the Dalle2_Model
    images = dalle2(text_batch,
                    cond_scale=opt.conductor['validation']['cond_scale'],  # classifier free guidance strength (> 1 would strengthen the condition)
                    return_pil_images=True)
    
    # Save the generated images
    for i, image in enumerate(images):
        image.save(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/image_{i}_dalle2.png')

if __name__ == '__main__':
    main()

