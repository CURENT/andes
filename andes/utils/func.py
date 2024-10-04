import functools
import operator
from typing import Iterable, Sized

from andes.shared import np


def list_flatten(input_list):
    """
    Flatten a multi-dimensional list into a flat 1-D list.
    """

    if len(input_list) > 0 and isinstance(input_list[0], (list, np.ndarray)):
        return functools.reduce(operator.iconcat, input_list, [])

    return input_list


def interp_n2(t, x, y):
    """
    Interpolation function for N * 2 value arrays.

    Parameters
    ----------
    t : float
        Point for which the interpolation is calculated
    x : 1-d array with two values
        x-axis values
    y : 2-d array with size N-by-2
        Values corresponding to x

    Returns
    -------
    N-by-1 array
        interpolated values at `t`

    """

    return y[:, 0] + (t - x[0]) * (y[:, 1] - y[:, 0]) / (x[1] - x[0])


def validate_keys_values(keys, values):
    """
    Validate the inputs for the func `find_idx`.

    Parameters
    ----------
    keys : str, array-like, Sized
        A string or an array-like of strings containing the names of parameters for the search criteria.
    values : array, array of arrays, Sized
        Values for the corresponding key to search for. If keys is a str, values should be an array of
        elements. If keys is a list, values should be an array of arrays, each corresponds to the key.

    Returns
    -------
    tuple
        Sanitized keys and values

    Raises
    ------
    ValueError
        If the inputs are not valid.
    """
    if isinstance(keys, str):
        keys = (keys,)
        if not isinstance(values, (int, float, str, np.floating)) and not isinstance(values, Iterable):
            raise ValueError(f"value must be a string, scalar or an iterable, got {values}")

        if len(values) > 0 and not isinstance(values[0], (list, tuple, np.ndarray)):
            values = (values,)

    elif isinstance(keys, Sized):
        if not isinstance(values, Iterable):
            raise ValueError(f"value must be an iterable, got {values}")

        if len(values) > 0 and not isinstance(values[0], Iterable):
            raise ValueError(f"if keys is an iterable, values must be an iterable of iterables. got {values}")

        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")

        if isinstance(values[0], Iterable):
            if not all([len(val) == len(values[0]) for val in values]):
                raise ValueError("All items in values must have the same length")

    return keys, values
