import logging
import logging.config

def get_logger(verbose=False):
    logging.config.fileConfig('configs/config_logger.conf')
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
    get_logger(verbose=True)