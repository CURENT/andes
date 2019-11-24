import functools
import operator


def list_flatten(idx):
    return functools.reduce(operator.iconcat, idx, [])
