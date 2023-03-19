import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import numpy as np
import open_clip

from transformers import CLIPProcessor
from tqdm import tqdm
from os.path import exists as file_exists
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Decoder, CLIP, OpenAIClipAdapter, DecoderTrainer, DiffusionPriorTrainer

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dalle2_dataset import get_text_tensor
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds


class Dalle2_Model(DALLE2):
    
    def __init__(self, opt: Opt, train_dataloader, valid_dataloader):

        #
        clip = self._create_clip_(opt=opt,
                                  train_dataloader=train_dataloader,
                                  valid_dataloader=valid_dataloader)

        #
        diffusion_prior = self._create_diffusion_prior_(opt=opt,
                                                        clip=clip,
                                                        train_dataloader=train_dataloader,
                                                        valid_dataloader=valid_dataloader)

        #
        self._create_decoder_(opt=opt)
        
        
        super().__init__(prior=diffusion_prior, 
                         decoder=self.decoder)
        
    
    def _create_clip_(self, opt: Opt, train_dataloader, valid_dataloader):
        
        if file_exists(opt.dalle2['clip']['model_save_path']) and opt.dalle2['clip']['use_existing_model']:
            clip = torch.load(opt.dalle2['clip']['epochs']['model_save_path'])

        else:

            if opt.dalle2['clip']['pretrained']:
                
                clip = OpenAIClipAdapter(name=opt.dalle2['clip']['model_name']).cuda()
            
            else:
                
                clip = CLIP(**opt.dalle2['clip']['params']).cuda()

                #
                clip = self._train_clip_(opt=opt,
                                            clip=clip,
                                            train_dataloader=train_dataloader,
                                            valid_dataloader=valid_dataloader)
            
        return clip


    def _train_clip_(self, opt: Opt, clip: CLIP, train_dataloader, valid_dataloader):
        
        # Start run with neptune docs
        neptune_ai = Neptune_AI(opt=opt)
        neptune_ai.start_neptune_run(opt)
        
        # Upload clip configs to neptune_ai
        neptune_ai.add_param_neptune_run(opt=opt, 
                                        data_item=opt.dalle2['clip'],
                                        neptune_run_save_path='clip_configs')
    
        #
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = processor.tokenizer
            
        #
        for epoch_n in tqdm(range(1, opt.dalle2['clip']['epochs'] + 1)):
            
            #
            for i, (image_batch, text_batch) in enumerate(tqdm(train_dataloader)):
                
                text_tensor = get_text_tensor(opt=opt, text_batch=list(text_batch),
                                              tokenizer=tokenizer)
                
                loss = clip(
                    text=text_tensor,
                    image=image_batch,
                    return_loss=True,  # needs to be set to True to return contrastive loss
                )
                loss.backward()
                
                #
                opt.logger.debug(f'Clip loss in epoch {epoch_n} | {i}: {loss}')
                
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                            data_item=loss,
                                            neptune_run_save_path=f"training/loss_clip")
            
            #
            valid_loss_list = list()
            for image_batch, _, text_batch in valid_dataloader:
                valid_loss = clip(
                    text=text_batch,
                    images=image_batch,
                    return_loss=True,  # needs to be set to True to return contrastive loss
                )
                valid_loss_list.append(valid_loss)
            
            #
            valid_loss_epoch_mean = np.mean(valid_loss_list)
            
            #
            opt.logger.debug(f'Clip valid_loss in epoch {epoch_n}: {valid_loss_epoch_mean}')
            
            # Upload epoch valid loss to neptune_ai
            neptune_ai.log_neptune_run(opt=opt,
                                        data_item=valid_loss_epoch_mean,
                                        neptune_run_save_path=f"training/valid_loss_clip")
            
        # Save final model
        torch.save(self.clip, opt.dalle2['clip']['model_save_path'])

        #
        neptune_ai.upload_neptune_run(opt=opt,
                                      data_item=opt.dalle2['clip']['model_save_path'],
                                      neptune_run_save_path='model')

        #
        neptune_ai.stop_neptune_run(opt=opt)
        
        return clip
        
    
    def _create_clip_embeds(self, opt: Opt, diffusion_prior: DiffusionPrior, train_dataloader, valid_dataloader):
        
        #
        image_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
        text_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
        
        for i, (image_batch, text_batch) in enumerate(tqdm(train_dataloader)):
            
            #
            clip_image_embeds = diffusion_prior.clip.embed_image(image_batch).image_embed
            clip_text_embeds = diffusion_prior.clip.embed_text(text_batch).text_embed
            
            #
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_batch_{i+1:05d}.pt')
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_batch_{i+1:05d}.pt')
        
     
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
        
        if file_exists(opt.dalle2['diffusion_prior']['model_save_path']) and opt.dalle2['diffusion_prior']['use_existing_model']:
            diffusion_prior_trainer.load(opt.dalle2['diffusion_prior']['model_save_path'])
        
        else:
            self._create_clip_embeds(opt=opt, 
                                     diffusion_prior=diffusion_prior,
                           train_dataloader=train_dataloader,
                           valid_dataloader=valid_dataloader)
            self._train_diffusion_prior_(opt=opt,
                                      diffusion_prior_trainer=diffusion_prior_trainer,
                           train_dataloader=train_dataloader,
                           valid_dataloader=valid_dataloader)
        
        
    def _train_diffusion_prior_(self, opt: Opt, diffusion_prior_trainer: DiffusionPriorTrainer, clip_text_embeds, clip_image_embeds):
        
        for epoch_n in tqdm(range(1, opt.dalle2['diffusion_prior_trainer']['epochs'] + 1)):
            loss = diffusion_prior_trainer(text_embed=clip_text_embeds, 
                                                image_embdes=clip_image_embeds)
            diffusion_prior_trainer.update()
        
        
    def _create_decoder_(self, opt: Opt, images, text):
        
        self.decoder = Decoder(
            clip = self.clip,
            unet = [dict(**opt.dalle2['unet1']),
                     dict(**opt.dalle2['unet2'])],          # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
            **opt.dalle2['decoder']['params']
        ).cuda()
        
        self.decoder_trainer = DecoderTrainer(
            decoder=self.decoder,
            **opt.dalle2['decoder_trainer']['params']
        )
        
    
    def _train_decoder_(self, opt: Opt, clip_text_embeds, clip_image_embeds):
        
        for i in tqdm(range(10000)):

            for unet_number in (1, 2):
                loss = self.decoder_trainer(
                    text_embed = clip_text_embeds,
                    image_embed = clip_image_embeds,
                    unet_number = unet_number, # which unet to train on
                    max_batch_size = opt.dalle2['decoder_trainer']['max_batch_size']
                )
                
                # update the specific unet as well as its exponential moving average
                self.decoder_trainer.update(unet_number) 

    
    def _set_dalle2_(self, opt: Opt):
    
        path_model_load = os.path.join(opt.base['PATH_BASE_DIR'],opt.conductor['trainer']['PATH_MODEL_LOAD'])
            


def main():
    opt=Opt()
    
    # train
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt, testing=False)
    train_dataloader, valid_dataloader = get_train_valid_dl(opt=opt, 
                                                            train_dataset=train_dataset, 
                                                            valid_dataset=valid_dataset)
        
    dalle2=Dalle2_Model(opt=opt, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)
    
    
if __name__ == '__main__':
    main()
    

