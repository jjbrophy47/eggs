"""
Informational print statements for logging purposes.
"""
import sys
import logging

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def performance(y_test, proba, logger=None,
                name='', do_print=False):
    """
    Returns AUROC and accuracy scores.
    """
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    score_str = '[{}] auc: {:.3f}, ap: {:.3f}'

    if logger:
        logger.info(score_str.format(name, auc, ap))
    elif do_print:
        print(score_str.format(name, auc))

    return auc, ap


def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []
