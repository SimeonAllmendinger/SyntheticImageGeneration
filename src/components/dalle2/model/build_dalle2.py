import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP, OpenAIClipAdapter, DecoderTrainer, DiffusionPriorTrainer
from tqdm import tqdm

from src.components.utils.opt.build_opt import Opt


class Dalle2_Model(DALLE2):
    
    def __init__(self, opt: Opt):
        
        #
        self._create_clip_(opt=opt)
        self._train_clip_(opt=opt)
        
        #
        self._create_diffusion_prior_(opt=opt)
        self._train_diffusion_prior_(opt=opt)
        
        #
        self._create_decoder_(opt=opt)
        
        
        super().__init__(prior=self.diffusion_prior, 
                         decoder=self.decoder)
        
    
    def _create_clip_(self, opt: Opt):
        
        #
        self.clip = CLIP(**opt.dalle2['clip']['params']).cuda()
        
        if opt.dalle2['clip']['pretrained']:
            self.clip = OpenAIClipAdapter()


    def _train_clip_(self, opt: Opt):
        
        # mock data

        text = torch.randint(0, 49408, (4, 256)).cuda()
        images = torch.randn(4, 3, 256, 256).cuda()

        # train

        for epoch_n in tqdm(1, range(opt.dalle2['clip']['epochs']) + 1):
            # TODO Dataloader
            loss = self.clip(
                text=text,
                images=images,
                return_loss=True,  # needs to be set to True to return contrastive loss
            )
            loss.backward()
            opt.logger.debug(f'Clip loss in epoch {epoch_n}: {loss}')
        
    
    def _create_diffusion_prior_(self, opt: Opt):
        
        self.diffusion_prior_network = DiffusionPriorNetwork(**opt.dalle2['diffusion_prior_network']['params']).cuda()

        self.diffusion_prior = DiffusionPrior(net=self.diffusion_prior_network,
                                              clip=self.clip,
                                              **opt.dalle2['diffusion_prior_network']['params']
                                              ).cuda()

        self.diffusion_prior_trainer = DiffusionPriorTrainer(
            diffusion_prior=self.diffusion_prior,
            **opt.dalle2['diffusion_prior_trainer']['params']
        )
        
    def _train_diffusion_prior_(self, opt: Opt):
        
        for i in tqdm(range(10000)):
            loss = self.diffusion_prior(text, images)
            loss.backward()
    
        
    def _set_decoder_(self, opt: Opt, images, text):
        
        self.decoder = Decoder(
            clip = self.clip,
            unet = [dict(**opt.dalle2['unet1']),
                     dict(**opt.dalle2['unet2'])],          # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
            image_sizes = (128, 256),                       # resolutions, 256 for first unet, 512 for second. these must be unique and in ascending order (matches with the unets passed in)
            timesteps = 1000,
            image_cond_drop_prob = 0.1,
            text_cond_drop_prob = 0.5
        ).cuda()
        
        self.decoder_trainer = DecoderTrainer(
            decoder=self.decoder,
            lr = 3e-4,
            wd = 1e-2,
            ema_beta = 0.99,
            ema_update_after_step = 1000,
            ema_update_every = 10,
)
        
        for i in tqdm(range(10000)):

            for unet_number in (1, 2):
                loss = self.decoder_trainer(
                    images,
                    text = text,
                    unet_number = unet_number, # which unet to train on
                    max_batch_size = 4         # gradient accumulation - this sets the maximum batch size in which to do forward and backwards pass - for this example 32 / 4 == 8 times
                )

                self.decoder_trainer.update(unet_number) # update the specific unet as well as its exponential moving average

    
    
    def _set_dalle2_(self, opt: Opt):
    
        path_model_load = os.path.join(opt.base['PATH_BASE_DIR'],opt.conductor['trainer']['PATH_MODEL_LOAD'])
            
        if self.testing:
            
            #
            test_model_save = os.path.join(opt.base['PATH_BASE_DIR'],opt.conductor['testing']['PATH_MODEL_TESTING'])
            self.imagen_model = load_imagen_from_checkpoint(test_model_save)
            
        elif (opt.conductor['trainer']['use_existing_model'] and glob.glob(path_model_load)):
            
            #
            self.imagen_model = load_imagen_from_checkpoint(path_model_load)

        else:
            
            if self.is_elucidated:
                # elucidated imagen-config, which contains the unets above (base unet and super resoluting ones)
                # the config class can be safed and loaded afterwards
                self.imagen_model = ElucidatedImagenConfig(unets=[dict(**opt.elucidated_imagen['unet1']),
                                                            dict(**opt.elucidated_imagen['unet2'])],
                                                     **opt.elucidated_imagen['elucidated_imagen']
                                                     ).create()
            else:
                # imagen-config, which contains the unets above (base unet and super resoluting ones)
                # the config class can be safed and loaded afterwards
                self.imagen_model = ImagenConfig(unets=[dict(**opt.imagen['unet1']),
                                                  dict(**opt.imagen['unet2'])],
                                           **opt.imagen['imagen']
                                           ).create()


def main():
    opt=Opt()
    dalle2=Dalle2_Model(opt=opt)
    
if __name__ == '__main__':
    main()
    

