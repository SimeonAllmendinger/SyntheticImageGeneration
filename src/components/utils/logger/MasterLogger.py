import os
import logging
import logging.config

def _get_logger_(path_base_dir: str, verbose=False):
    
    logging.config.fileConfig(os.path.join(path_base_dir,'configs/utils/config_logger.conf'))
    master_logger = logging.getLogger('MasterLogger')
    
    if verbose:
        # 'application' code
        master_logger.debug('debug message')
        master_logger.info('info message')
        master_logger.warning('warn message')
        master_logger.error('error message')
        master_logger.critical('critical message')
        
    return master_logger

if __name__ == '__main__':
    _get_logger_(verbose=True)