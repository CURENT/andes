"""
Development related functions
"""
import logging
logger = logging.getLogger(__name__)


def warn_experimental(message: str):
    return logger.warning(f"Experimental feature: {message}")
