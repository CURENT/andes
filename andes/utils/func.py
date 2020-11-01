import functools
import operator

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
