from cvxopt import matrix, spdiag, mul, div, spmatrix, sparse

from .base import ModelBase
from ..consts import *
from ..utils.math import *


class SeriesBase(ModelBase):
    """Base class for AC series devices"""
    def __init__(self, system, name):
        super(SeriesBase, self).__init__(system, name)

