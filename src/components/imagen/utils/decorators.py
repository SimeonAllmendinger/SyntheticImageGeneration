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
                if opt.datasets['data']['Cholec80']['use_phase_labels']:
                    assert opt.imagen['imagen']['text_embed_dim'] == 36, "Text embed dim must be changed to 36"
        
                else:
                    assert opt.imagen['imagen']['text_embed_dim'] == 29, "Text embed dim must be changed to 29"
        
        else:
            assert True, "Choose an existing encoder: 'ohe_encoder' or 'google/t5-v1_1-base'"
                
        return func(*args, **kwargs)
        
    return wrapper


def check_dataset_name(func):
    
    opt = Opt()

    def wrapper(*args, **kwargs):

        assert opt.datasets['data']['dataset'] == 'CholecT45' or opt.datasets['data'][
            'dataset'] == 'CholecSeg8k' or opt.datasets['data']['dataset'] == 'Both', 'Choose an existing dataset: CholecT45, CholecSeg8k or Both'

        return func(*args, **kwargs)

    return wrapper


def model_starter(func):
    
    opt=Opt()
    
    def wrapper(*args, **kwargs):
        
        #TODO: opt.logger.info('Load Imagen model from Checkpoint for Validation')
        
        if opt.conductor['trainer']['use_existing_model'] or glob.glob(opt.conductor['trainer']['PATH_MODEL_CHECKPOINT']):
                
            opt.logger.info('Load Imagen model from Checkpoint for further Training')
        
        else:

            opt.logger.info('Create new Imagen model')
        
        return func(*args, **kwargs)
    
    opt.logger.debug('Imagen Model built')
    
    return wrapper