import logging


def get_logger(filename):
    # setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter("[%(levelname)s][%(asctime)s;%(filename)s:%(lineno)d:] %(message)s"))
    handler = logging.FileHandler(filename=filename, encoding="utf8")
    handler.setLevel("INFO")
    handler.setFormatter(
        logging.Formatter("[%(levelname)s][%(asctime)s;%(filename)s:%(lineno)d:] %(message)s")
    )
    logger.root.addHandler(handler)
    return logger