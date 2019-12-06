import functools
import operator
import logging
import numpy as np

logger = logging.getLogger(__name__)


def list_flatten(idx):
    if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
        return functools.reduce(operator.iconcat, idx, [])
    else:
        return idx
