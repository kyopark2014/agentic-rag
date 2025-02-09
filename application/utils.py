import logging
import sys

#logging
def CreateLogger(logger_name):
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s')
    #formatter = logging.Formatter('%(asctime)s | %(filename)s:%(lineno)d | %(message)s')
    formatter = logging.Formatter('%(filename)s:%(lineno)d | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler) 

    try:
        with open("/home/config.json", "r", encoding="utf-8") as f:
            file_handler = logging.FileHandler('/var/log/application/logs.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    except Exception:
        logger.info(f"Not available log saving")

    return logger