"""
Stability criteria module.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def deltadelta(delta, diff_limit):
    """
    Test if a system is stable by comparing  the maximum rotor angle difference
    with a threshold.

    Returns
    -------
    bool
        True if the system is stable, False otherwise.
    """

    if len(delta) < 2:
        return True

    diff_max = np.max(delta - np.min(delta))

    return (diff_max < np.deg2rad(diff_limit)).tolist()
