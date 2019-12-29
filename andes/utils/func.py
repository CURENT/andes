import functools
import operator

from andes.shared import np


def list_flatten(input_list):
    """
    Flatten a multi-dimensional list into a flat 1-D list.
    """
    if len(input_list) > 0 and isinstance(input_list[0], (list, np.ndarray)):
        return functools.reduce(operator.iconcat, input_list, [])
    else:
        return input_list
