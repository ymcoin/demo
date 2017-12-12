import logging
import time
def get_logger(name='1',file_name='temp',verbose=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    # file_name = time.strftime("%d_%m_%Y_%H_%M_%S") + '.txt'
    file = file_name
    fh = logging.FileHandler(file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if (verbose == True):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger