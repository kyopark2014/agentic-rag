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

def get_contents_type(file_name):
    if file_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif file_name.lower().endswith((".pdf")):
        content_type = "application/pdf"
    elif file_name.lower().endswith((".txt")):
        content_type = "text/plain"
    elif file_name.lower().endswith((".csv")):
        content_type = "text/csv"
    elif file_name.lower().endswith((".ppt", ".pptx")):
        content_type = "application/vnd.ms-powerpoint"
    elif file_name.lower().endswith((".doc", ".docx")):
        content_type = "application/msword"
    elif file_name.lower().endswith((".xls")):
        content_type = "application/vnd.ms-excel"
    elif file_name.lower().endswith((".py")):
        content_type = "text/x-python"
    elif file_name.lower().endswith((".js")):
        content_type = "application/javascript"
    elif file_name.lower().endswith((".md")):
        content_type = "text/markdown"
    elif file_name.lower().endswith((".png")):
        content_type = "image/png"
    else:
        content_type = "no info"    
    return content_type