import numpy as np


def safe_div(a, b, out=None):
    """
    Safe division for numpy. Division by zero yields zero.

    `safe_div` cannot be used in `e_str` due to unsupported Derivative.
    Parameters
    ----------
    out : None or array-like
        If not None, out contains the default values for where b==0.
    """
    if out is None:
        out = np.zeros_like(a)

    return np.divide(a, b, out=out, where=(b != 0))
