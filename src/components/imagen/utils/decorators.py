import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob

from src.components.utils.opt.build_opt import Opt

def check_text_encoder(func):
    opt=Opt()
    def wrapper(*args, **kwargs):
        
        if opt.imagen['imagen']['text_encoder_name'] == 'google/t5-v1_1-base':      
            assert opt.imagen['imagen']['text_embed_dim'] == 768, "Text embed dim must be changed to 768"
    
        elif opt.imagen['imagen']['text_encoder_name'] == 'ohe_encoder':
            assert opt.imagen['imagen']['text_embed_dim'] == 29, "Text embed dim must be changed to 29"
        
        else:
            assert True, "Choose an existing encoder: 'ohe_encoder' or 'google/t5-v1_1-base'"
                
        return func(*args, **kwargs)
        
    return wrapper


def check_dataset_name(func):
    
    opt=Opt()
    
    def wrapper(*args, **kwargs):
        
        assert opt.imagen['data']['dataset'] == 'CholecT45' or opt.imagen['data']['dataset'] == 'CholecSeg8k', 'Choose an existing dataset: CholecT45 or CholecSeg8k'
        
        return func(*args, **kwargs)
        
    return wrapper


def model_starter(func):
    
    opt=Opt()
    
    def wrapper(*args, **kwargs):
        
        #TODO: opt.logger.info('Load Imagen model from Checkpoint for Validation')
        
        if opt.imagen['trainer']['use_existing_model'] or glob.glob(opt.imagen['trainer']['PATH_MODEL_CHECKPOINT']):
                
            opt.logger.info('Load Imagen model from Checkpoint for further Training')
        
        else:

            opt.logger.info('Create new Imagen model')
        
        return func(*args, **kwargs)
    
    opt.logger.debug('Imagen Model built')
    
    return wrapper