"""
Utility functions for compiling functions with numba.
"""

from typing import Union, Callable

from andes.shared import numba


def to_jit(func: Union[Callable, None],
           parallel: bool = False,
           cache: bool = False,
           nopython: bool = False,
           ):
    """
    Helper function for converting a function to a numba jit-compiled function.

    Note that this function will be compiled just-in-time when first called,
    based on the argument types.
    """

    if func is not None:
        return numba.jit(func,
                         parallel=parallel,
                         cache=cache,
                         nopython=nopython,
                         )

    return func
