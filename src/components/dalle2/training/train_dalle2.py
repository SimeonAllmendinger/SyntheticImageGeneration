import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch

from tqdm import tqdm
from dalle2_pytorch import DecoderTrainer

from src.components.utils.opt.build_opt import Opt
from src.components.utils.neptune.neptune_ai import Neptune_AI
from src.components.data_manager.dataset_handler import get_train_valid_dl, get_train_valid_ds
from src.components.utils.training.early_stopping import EarlyStopping


def _train_decoder_(opt: Opt, decoder_trainer: DecoderTrainer):

    # train
    train_dataset, valid_dataset = get_train_valid_ds(opt=opt, return_embeds=True, return_text=False)
    train_dataloader, valid_dataloader = get_train_valid_dl(opt=opt, 
                                                            train_dataset=train_dataset,
                                                            valid_dataset=valid_dataset)
    
    loss_queue = EarlyStopping(opt_early_stopping=opt.conductor['trainer']['early_stopping'])
    
    # Start run with neptune docs
    neptune_ai = Neptune_AI(opt=opt)
    neptune_ai.start_neptune_run(opt)
    
    # Upload deecoder configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                    data_item=opt.dalle2['decoder'],
                                    neptune_run_save_path='decoder_configs')
    
    # Upload decoder configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                    data_item=opt.dalle2['decoder_trainer'],
                                    neptune_run_save_path='decoder_trainer_configs')
    
    # Upload decoder configs to neptune_ai
    neptune_ai.add_param_neptune_run(opt=opt, 
                                    data_item=opt.conductor['trainer'],
                                    neptune_run_save_path='primary_configs')

    for unet_k in range(1, opt.conductor['trainer']['unet_number']+1):
        for epoch_n in tqdm(range(1, opt.conductor['trainer']['max_epochs'] + 1)):
            for i, (image_batch, text_encodings, image_embed_batch, text_embed_batch) in enumerate(tqdm(train_dataloader, disable=False)):
                
                if opt.dalle2['decoder_trainer']['train_with_embeds'] and unet_k>1:
                    loss = decoder_trainer(
                        text_encodings=text_embed_batch.cuda(),
                        image=image_batch.cuda(),
                        # which unet to train on
                        unet_number=unet_k,
                        max_batch_size=opt.conductor['trainer']['batch_size']
                    )
                    
                else:
                    loss = decoder_trainer(
                        text=text_encodings.cuda(),
                        image=image_batch.cuda(),
                        # which unet to train on
                        unet_number=unet_k,
                        max_batch_size=opt.conductor['trainer']['batch_size']
                    )

                # update the specific unet as well as its exponential moving average
                decoder_trainer.update(unet_number=unet_k) 
                
                #
                opt.logger.debug(f'Decoder loss in epoch {epoch_n} of unet {unet_k} | {i}: {loss}')
            
                # Upload epoch valid loss to neptune_ai
                neptune_ai.log_neptune_run(opt=opt,
                                            data_item=loss,
                                            neptune_run_save_path=f"training/loss_decoder_prior")
                
                if opt.conductor['trainer']['early_stopping']['usage']:
                    loss_queue.push(loss)

                    if loss_queue.stop:
                        opt.logger.info('Stop training early')
                        break

            #
            '''image_batch, text_encodings, *_ = next(iter(valid_dataloader))

            valid_loss = decoder_trainer(text=text_encodings,
                                        image=image_batch,
                                        # which unet to train on
                                        unet_number=unet_k,
                                        max_batch_size=opt.conductor['trainer']['batch_size']
                                        )

            opt.logger.debug(f'Decoder valid loss: {valid_loss}')

            # Upload epoch valid loss to neptune_ai
            decoder_trainer.save(opt.dalle2['decoder_trainer']['model_save_path'])
            
            neptune_ai.log_neptune_run(opt=opt,
                                        data_item=valid_loss,
                                        neptune_run_save_path=f"training/valid_loss_decoder")'''
            
        neptune_ai.upload_neptune_run(opt=opt,
                            data_item=opt.dalle2['decoder_trainer']['model_save_path'],
                            neptune_run_save_path=f"model")
    #
    neptune_ai.stop_neptune_run(opt=opt)
    
    return decoder_trainer


if __name__ == '__main__':
    opt=Opt()
    
    decoder_trainer=DecoderTrainer()
    _train_decoder_(opt=opt, decoder_trainer=decoder_trainer)
