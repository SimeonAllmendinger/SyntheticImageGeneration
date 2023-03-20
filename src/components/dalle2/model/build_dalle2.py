import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import exists as file_exists
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Decoder, CLIP, OpenAIClipAdapter, DecoderTrainer, DiffusionPriorTrainer

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds


class Dalle2_Model(DALLE2):
    
    def __init__(self, opt: Opt, train_dataloader, valid_dataloader):

        #
        clip = self._init_clip_(opt=opt,
                                  train_dataloader=train_dataloader,
                                  valid_dataloader=valid_dataloader)

        #
        diffusion_prior_trainer = self._create_diffusion_prior_(opt=opt,
                                                        clip=clip,
                                                        train_dataloader=train_dataloader,
                                                        valid_dataloader=valid_dataloader)

        #
        decoder_trainer =self._create_decoder_(opt=opt)
        
        #
        super().__init__(prior=diffusion_prior_trainer, 
                         decoder=decoder_trainer)
        
    
    def _init_clip_(self, opt: Opt, train_dataloader, valid_dataloader):
        
        if file_exists(opt.dalle2['clip']['model_save_path']) and opt.dalle2['clip']['use_existing_model']:
            clip = torch.load(opt.dalle2['clip']['epochs']['model_save_path'])

        else:

            if opt.dalle2['clip']['pretrained']:
                
                clip = OpenAIClipAdapter(name=opt.dalle2['clip']['model_name']).cuda()
            
            else:
                
                clip = CLIP(**opt.dalle2['clip']['params']).cuda()
            
        return clip
        
    
    def _create_clip_embeds(self, opt: Opt, diffusion_prior: DiffusionPrior, train_dataloader, valid_dataloader, 
                            image_embeds_save_dir_path: str, text_embeds_save_dir_path: str):
        
        for i, (image_batch, text_batch) in enumerate(tqdm(train_dataloader, disable=False)):
            
            #
            clip_image_embeds = diffusion_prior.clip.embed_image(image_batch).image_embed
            clip_text_embeds = diffusion_prior.clip.embed_text(text_batch).text_embed
            
            #
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_batch_{i+1:05d}.pt')
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_batch_{i+1:05d}.pt')
            
        opt.logger.debug('Training embeds created')
        
        
        for i, (image_batch, text_batch) in enumerate(tqdm(valid_dataloader, disable=False)):
            
            #
            clip_image_embeds = diffusion_prior.clip.embed_image(image_batch).image_embed
            clip_text_embeds = diffusion_prior.clip.embed_text(text_batch).text_embed
            
            #
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'valid_image_batch_{i+1:05d}.pt')
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'valid_text_batch_{i+1:05d}.pt')
            
        opt.logger.debug('Valid embeds created')
        
     
    def _create_diffusion_prior_(self, opt: Opt, clip: CLIP, train_dataloader, valid_dataloader):
        
        diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

        diffusion_prior = DiffusionPrior(net=diffusion_prior_network,
                                            clip=clip,
                                            **opt.dalle2['diffusion_prior']['params']
                                            ).cuda()

        diffusion_prior_trainer = DiffusionPriorTrainer(
            diffusion_prior=diffusion_prior,
            **opt.dalle2['diffusion_prior_trainer']['params']
        )
            
        if file_exists(opt.dalle2['diffusion_prior_trainer']['model_save_path']) and opt.dalle2['diffusion_prior_trainer']['use_existing_model']:
            diffusion_prior_trainer.load(opt.dalle2['diffusion_prior']['model_save_path'])
        
        else:
            
            image_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
            text_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR'])
        
            if not opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['use_existing_embeds']:

                self._create_clip_embeds(opt=opt,
                                         diffusion_prior=diffusion_prior,
                                         train_dataloader=train_dataloader,
                                         valid_dataloader=valid_dataloader,
                                         image_embeds_save_dir_path=image_embeds_save_dir_path,
                                         text_embeds_save_dir_path=text_embeds_save_dir_path)

            diffusion_prior_trainer = self._train_diffusion_prior_(opt=opt,
                                                                   diffusion_prior_trainer=diffusion_prior_trainer,
                                                                   train_dataloader=train_dataloader,
                                                                   valid_dataloader=valid_dataloader)
            
            diffusion_prior_trainer.save(opt.dalle2['diffusion_prior']['model_save_path'])
            
            return diffusion_prior_trainer
            
                    
    def _train_diffusion_prior_(self, opt: Opt, diffusion_prior_trainer: DiffusionPriorTrainer, 
                                train_dataloader, valid_dataloader):
        
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
        
        
        for epoch_n in tqdm(range(1, opt.dalle2['diffusion_prior_trainer']['epochs'] + 1)):
            
            for i, (clip_image_embeds, clip_text_embeds) in enumerate(tqdm(train_dataloader, disable=False)):
                
                loss = diffusion_prior_trainer(text_embed=clip_text_embeds,
                                            image_embdes=clip_image_embeds)
                diffusion_prior_trainer.update()
            
            opt.logger.debug(f'Diffusion Prior loss in epoch {epoch_n} | {i}: {loss}')
            
            # Upload epoch valid loss to neptune_ai
            neptune_ai.log_neptune_run(opt=opt,
                                        data_item=valid_loss,
                                        neptune_run_save_path=f"training/loss_diffusion_prior")
        
            #
            with torch.no_grad():
                k = torch.randint(0, len(valid_dataloader) *
                                  opt.conductor['trainer']['batch_size'], (1))
                clip_image_embeds, clip_text_embeds = valid_dataloader[k]

                valid_loss = diffusion_prior_trainer(text_embed=clip_text_embeds,
                                                     image_embdes=clip_image_embeds)

                opt.logger.debug(f'Diffusion Prior valid loss: {valid_loss}')

                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                           data_item=valid_loss,
                                            neptune_run_save_path=f"training/valid_loss_diffusion_prior")
        #
        neptune_ai.stop_neptune_run(opt=opt)
        
        return diffusion_prior_trainer
    
        
    def _create_decoder_(self, opt: Opt, clip: CLIP, train_dataloader, valid_dataloader):
        
        decoder = Decoder(
                clip = clip,
                unet = [dict(**opt.dalle2['unet1']),
                        dict(**opt.dalle2['unet2'])],          # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
                **opt.dalle2['decoder']['params']
            ).cuda()

        decoder_trainer = DecoderTrainer(
            decoder=decoder,
            **opt.dalle2['decoder_trainer']['params']
        )

        if file_exists(opt.dalle2['decoder_trainer']['model_save_path']) and opt.dalle2['decoder_trainer']['use_existing_model']:
            decoder_trainer.load(opt.dalle2['decoder']['model_save_path'])
        
        else:
            
            decoder_trainer = self._train_decoder_(opt=opt,
                                            decoder_trainer=decoder_trainer,
                                            train_dataloader=train_dataloader,
                                            valid_dataloader=valid_dataloader)
            
        return decoder_trainer
    

    def _train_decoder_(self, opt: Opt, decoder_trainer, train_dataloader, valid_dataloader):

        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt)
        neptune_ai.start_neptune_run(opt)
        
        # Upload clip configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt, 
                                        data_item=opt.dalle2['decoder'],
                                        neptune_run_save_path='decoder_configs')
        
        for epoch_n in tqdm(range(1, opt.dalle2['decoder_trainer']['epochs'] + 1)):
            for i, (clip_image_embeds, clip_text_embeds) in enumerate(tqdm(train_dataloader, disable=False)):
                loss = decoder_trainer(
                    text_embed = clip_text_embeds,
                    image_embed = clip_image_embeds,
                    unet_number = opt.conductor['training']['unet_number'], # which unet to train on
                    max_batch_size = opt.dalle2['decoder_trainer']['max_batch_size']
                )
                
                # update the specific unet as well as its exponential moving average
                self.decoder_trainer.update(unet_number=opt.conductor['training']['unet_number']) 
                
                opt.logger.debug(f'Decoder loss in epoch {epoch_n} | {i}: {loss}')
            
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                            data_item=valid_loss,
                                            neptune_run_save_path=f"training/loss_decoder_prior")
        
            #
            with torch.no_grad():
                k = torch.randint(0, len(valid_dataloader) * opt.conductor['trainer']['batch_size'], (1))
                clip_image_embeds, clip_text_embeds = valid_dataloader[k]

                valid_loss = decoder_trainer(text_embed=clip_text_embeds,
                                                     image_embdes=clip_image_embeds)

                opt.logger.debug(f'Decoder valid loss: {valid_loss}')

                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                           data_item=valid_loss,
                                            neptune_run_save_path=f"training/valid_loss_decoder")
        #
        neptune_ai.stop_neptune_run(opt=opt)
        
        return decoder_trainer


def main():
    opt=Opt()
    
    # train
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt, testing=False)
    train_dataloader, valid_dataloader = get_train_valid_dl(opt=opt, 
                                                            train_dataset=train_dataset, 
                                                            valid_dataset=valid_dataset)
        
    dalle2=Dalle2_Model(opt=opt, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)
    image = dalle2(['grasper grasp gallbladder'],
                    # classifier free guidance strength (> 1 would strengthen the condition)
                    cond_scale=2.
                    )
    
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    fig.save('./results/image_test_dalle2.png')
    
if __name__ == '__main__':
    main()
    

