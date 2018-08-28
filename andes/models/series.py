from cvxopt import matrix, spdiag, mul, div, spmatrix, sparse  # NOQA

from .base import ModelBase

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class SeriesBase(ModelBase):
    """Base class for AC series devices"""
    def __init__(self, system, name):
        super(SeriesBase, self).__init__(system, name)
