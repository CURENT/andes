"""
Development related functions
"""
import logging
logger = logging.getLogger(__name__)


def warn_experimental(feature: str):
    return logger.warning(f"{feature} is experimental.")
