import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import exists as file_exists
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Decoder, CLIP, Unet, OpenAIClipAdapter, DecoderTrainer, DiffusionPriorTrainer

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds
from src.components.dalle2.training.train_dalle2 import _train_decoder_
from src.components.utils.training.early_stopping import EarlyStopping


class Prior_Model():
    
    def __init__(self, opt: Opt):
        
        if opt.conductor['trainer']['early_stopping']['usage']:
            self.loss_queue = EarlyStopping(opt_early_stopping=opt.conductor['trainer']['early_stopping'])
        #
        self._init_clip_(opt=opt)

        #
        self._create_diffusion_prior_(opt=opt)
        
        
    def _init_clip_(self, opt: Opt):
                
        self.clip = OpenAIClipAdapter(name=opt.dalle2['clip']['model_name']).cuda()
            
    
    def _create_clip_embeds(self, opt: Opt,
                            image_embeds_save_dir_path: str, 
                            text_embeds_save_dir_path: str):
        
        opt.conductor['trainer']['shuffle'] = False
        dataset = get_train_valid_ds(opt=opt, testing=True)
        dataloader = get_train_valid_dl(opt=opt, train_dataset=dataset)
        
        for i, (image_batch, text_batch) in enumerate(tqdm(dataloader, disable=False)):
  
            #
            clip_image_embeds = self.diffusion_prior.clip.embed_image(image_batch).image_embed
            torch.save(clip_image_embeds, image_embeds_save_dir_path + f'image_embeds_{i:05d}.pt')

            #
            clip_text_embeds = self.diffusion_prior.clip.embed_text(text_batch).text_embed
            torch.save(clip_text_embeds, text_embeds_save_dir_path + f'text_embeds_{i:05d}.pt')
            
        opt.logger.debug('Training embeds created')
        
     
    def _create_diffusion_prior_(self, opt: Opt):
        
        self.diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

        self.diffusion_prior = DiffusionPrior(net=self.diffusion_prior_network,
                                            clip=self.clip,
                                            **opt.dalle2['diffusion_prior']['params']
                                            ).cuda()

        self.diffusion_prior_trainer = DiffusionPriorTrainer(
            diffusion_prior=self.diffusion_prior,
            **opt.dalle2['diffusion_prior_trainer']['params']
        )
            
        if file_exists(opt.dalle2['diffusion_prior_trainer']['model_save_path']) and opt.dalle2['diffusion_prior']['use_existing_model']:
            self.diffusion_prior_trainer.load(opt.dalle2['diffusion_prior_trainer']['model_save_path'])
        
        else:
            
            image_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_IMAGE_EMBEDDING_DIR'])
            text_embeds_save_dir_path = os.path.join(opt.base['PATH_BASE_DIR'], opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['PATH_CLIP_TEXT_EMBEDDING_DIR'])
        
            if not opt.datasets['data'][opt.datasets['data']['dataset']]['clip']['use_existing_embeds']:

                self._create_clip_embeds(opt=opt,
                                         image_embeds_save_dir_path=image_embeds_save_dir_path,
                                         text_embeds_save_dir_path=text_embeds_save_dir_path)

            self._train_diffusion_prior_(opt=opt)
            
            self.diffusion_prior_trainer.save(opt.dalle2['diffusion_prior_trainer']['model_save_path'])
            
                    
    def _train_diffusion_prior_(self, opt: Opt):
        
        # Dataloader
        dataset = get_train_valid_ds(opt=opt, testing=True)
        dataloader = get_train_valid_dl(opt=opt, train_dataset=dataset)
    
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

            for i, (*_, clip_image_embeds, clip_text_embeds) in enumerate(tqdm(dataloader, disable=False)):
                loss = self.diffusion_prior_trainer(text_embed=clip_text_embeds,
                                                    image_embed=clip_image_embeds)
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
    
    def __init__(self, opt: Opt):

        prior = Prior_Model(opt=opt)
        
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
            
        else:
            decoder_trainer = _train_decoder_(opt=opt, decoder_trainer=decoder_trainer)
            
            decoder_trainer.save(opt.dalle2['decoder_trainer']['model_save_path'])
            
        return decoder_trainer
    

def main():
    opt=Opt()
    
    dalle2=Dalle2_Model(opt=opt)
    
    image = dalle2(['grasper grasp gallbladder'],
                    # classifier free guidance strength (> 1 would strengthen the condition)
                    cond_scale=2.0,
                    return_pil_images = True
                    )
    
    image[0].save('./results/image_test_dalle2.png')
    
if __name__ == '__main__':
    main()
    

