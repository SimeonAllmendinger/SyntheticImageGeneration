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
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Decoder, CLIP, Unet, OpenClipAdapter, DecoderTrainer, DiffusionPriorTrainer

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds
from src.components.dalle2.training.train_dalle2 import _train_decoder_
from src.components.utils.training.early_stopping import EarlyStopping

parser = argparse.ArgumentParser(
                prog='SyntheticImageGeneration',
                description='Magic with Text2Image',
                epilog='For help refer to uerib@student.kit.edu')

parser.add_argument('--path_data_dir',
                    default='/home/kit/stud/uerib/SyntheticImageGeneration/',
                    help='PATH to data directory')


class OpenClipAdapterWithContextLength(OpenClipAdapter):
    def __init__(self, opt: Opt):
        self.context_length = opt.dalle2['clip']['context_length']
        super().__init__(name=opt.dalle2['clip']['model_name'], 
                         pretrained=opt.dalle2['clip']['pretrained'])
        
    @property
    def max_text_len(self):
        return self.context_length
        

class Prior_Model():
    
    def __init__(self, opt: Opt, testing=False):
        
        self.testing = testing
        
        if opt.conductor['trainer']['early_stopping']['usage']:
            self.loss_queue = EarlyStopping(opt_early_stopping=opt.conductor['trainer']['early_stopping'])
        #
        self._init_clip_(opt=opt)

        #
        self._create_diffusion_prior_(opt=opt)
        
        
    def _init_clip_(self, opt: Opt):
                
        self.clip = OpenClipAdapterWithContextLength(opt=opt).cuda()
    
    
    def _create_clip_embeds(self, opt: Opt,
                            image_embeds_save_dir_path: str, 
                            text_embeds_save_dir_path: str):
        
        opt.conductor['trainer']['shuffle'] = False
        dataset = get_train_valid_ds(opt=opt, return_text=False, testing=True, return_embeds=False)
        dataloader = get_train_valid_dl(opt=opt, train_dataset=dataset)
        
        opt.logger.info('Create text embeddings...')
        
        for i, (image_batch, text_enc_batch, *_) in enumerate(tqdm(dataloader, disable=False)):

            #
            clip_image_embeds = self.diffusion_prior.clip.embed_image(image_batch.cuda()).image_embed
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_embeds_{i:05d}.pt')

            #
            clip_text_embeds = self.diffusion_prior.clip.embed_text(text_enc_batch.cuda()).text_embed
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_embeds_{i:05d}.pt')
            
        opt.logger.debug('Training embeds created')
        
     
    def _create_diffusion_prior_(self, opt: Opt):
        
        self.diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

        self.diffusion_prior = DiffusionPrior(net=self.diffusion_prior_network,
                                              clip=self.clip,
                                              **opt.dalle2['diffusion_prior']['params']
                                              ).cuda()

        self.diffusion_prior_trainer = DiffusionPriorTrainer(diffusion_prior=self.diffusion_prior,
                                                             **opt.dalle2['diffusion_prior_trainer']['params']
                                                             )
            
        if file_exists(opt.dalle2['diffusion_prior_trainer']['model_save_path']) and opt.dalle2['diffusion_prior']['use_existing_model']:
            self.diffusion_prior_trainer.load(opt.dalle2['diffusion_prior_trainer']['model_save_path'])
        
        if not self.testing:
            
            image_embeds_save_dir_path = os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
            text_embeds_save_dir_path = os.path.join(opt.datasets['PATH_DATA_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR'])
        
            if not opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['use_existing_embeds']:
            
                self._create_clip_embeds(opt=opt,
                                         image_embeds_save_dir_path=image_embeds_save_dir_path,
                                         text_embeds_save_dir_path=text_embeds_save_dir_path)

            self._train_diffusion_prior_(opt=opt)
            
            self.diffusion_prior_trainer.save(opt.dalle2['diffusion_prior_trainer']['model_save_path'])
            
                    
    def _train_diffusion_prior_(self, opt: Opt):
        
        # Dataloader
        train_dataset, valid_dataset = get_train_valid_ds(opt=opt, return_text=False, return_embeds=True)
        train_dl, valid_dl = get_train_valid_dl(opt=opt, train_dataset=train_dataset, valid_dataset=valid_dataset)
    
        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt)
        neptune_ai.start_neptune_run(opt)
        
        # Upload clip configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt,
                                         data_item=opt.dalle2['diffusion_prior_network'],
                                         neptune_run_save_path='diffusion_prior_network_configs')
        neptune_ai.add_param_neptune_run(opt=opt,
                                         data_item=opt.dalle2['diffusion_prior'],
                                         neptune_run_save_path='diffusion_prior_configs')
        neptune_ai.add_param_neptune_run(opt=opt,
                                         data_item=opt.dalle2['diffusion_prior_trainer'],
                                         neptune_run_save_path='diffusion_prior_trainer_configs')

        #
        for epoch_n in tqdm(range(1, opt.dalle2['diffusion_prior_trainer']['epochs'] + 1)):

            for i, (image_batch, text_encodings, clip_image_embeds, clip_text_embeds) in enumerate(tqdm(train_dl, disable=False)):
                
                if opt.dalle2['diffusion_prior_trainer']['train_with_embeds']:
                    loss = self.diffusion_prior_trainer(text_embed=clip_text_embeds.cuda(),
                                                        image_embed=clip_image_embeds.cuda())
                else:
                    loss = self.diffusion_prior_trainer(text=text_encodings.cuda(),
                                                        image=image_batch.cuda())
                self.diffusion_prior_trainer.update()

                opt.logger.debug(
                    f'Diffusion Prior loss in epoch {epoch_n} | {i}: {loss}')
            
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                            data_item=loss,
                                            neptune_run_save_path=f"training/loss_diffusion_prior")
                
                #
                if opt.conductor['trainer']['early_stopping']['usage']:
                    self.loss_queue.push(loss)

                    if self.loss_queue.stop:
                        opt.logger.info('Stop training early')
                        break
        
        #
        neptune_ai.stop_neptune_run(opt=opt)
        

class Dalle2_Model(DALLE2):
    
    def __init__(self, opt: Opt, testing=False):

        prior = Prior_Model(opt=opt)
        self.testing=testing
        #
        decoder_trainer = self._create_decoder_(opt=opt, 
                                                clip=prior.clip)

        #
        super().__init__(prior=prior.diffusion_prior_trainer.diffusion_prior,
                         decoder=decoder_trainer.decoder)
        
    
    def _create_decoder_(self, opt: Opt, clip: CLIP):
        
        unet1 = Unet(**opt.dalle2['unet1']).cuda()
        unet2 = Unet(**opt.dalle2['unet2']).cuda()
        
        decoder = Decoder(
                clip = clip,
                unet = [unet1, unet2],          # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
                **opt.dalle2['decoder']['params']
            ).cuda()

        decoder_trainer = DecoderTrainer(
            decoder=decoder,
            **opt.dalle2['decoder_trainer']['params']
        )

        if file_exists(opt.dalle2['decoder_trainer']['model_save_path']) and opt.dalle2['decoder']['use_existing_model']:
            decoder_trainer.load(opt.dalle2['decoder_trainer']['model_save_path'])
            
        if not self.testing:
            
            decoder_trainer = _train_decoder_(opt=opt, decoder_trainer=decoder_trainer)
            
            decoder_trainer.save(opt.dalle2['decoder_trainer']['model_save_path'])
            
        return decoder_trainer
    

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
    
    dalle2=Dalle2_Model(opt=opt, testing=False)
    
    # Dataloader
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt, return_text=True, return_embeds=False)
    train_dl, valid_dl = get_train_valid_dl(opt=opt, train_dataset=train_dataset, valid_dataset=valid_dataset)
    
    image_batch, text_encodings, text_batch = next(iter(train_dl))
    
    images = dalle2(text_batch,
                # classifier free guidance strength (> 1 would strengthen the condition)
                cond_scale=opt.conductor['validation']['cond_scale'],
                return_pil_images = True
                )
    
    for i, image in enumerate(images):
        image.save(f'/home/kit/stud/uerib/SyntheticImageGeneration/results/image_{i}_dalle2.png')

if __name__ == '__main__':
    main()
    

