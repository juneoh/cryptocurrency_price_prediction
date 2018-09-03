#!/usr/bin/env python3
"""Provide formatted logging to STDOUT and file.
"""
import logging
import logging.handlers
import os


def get_logger(log_file):
    """Prepare formatted logger to stream and file.

    Args:
        log_file (str): The file to save the logs into.

    Returns:
        logging.Logger: The logger object.
    """
    # Create logger and formatter.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Create and attach stream handler.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create and attach file handler.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='d', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
