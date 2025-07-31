import logging
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    设置日志记录器。
    
    :param log_file: 日志文件路径，如果为None，则仅在控制台输出日志。
    :param log_level: 日志级别，默认为INFO。
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
