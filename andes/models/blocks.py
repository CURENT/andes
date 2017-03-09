"""Control blocks"""
from cvxopt import matrix, sparse, spmatrix
from cvxopt import mul, div, log, sin, cos
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class PI(object):
    """PI controller class as addon base class"""
    def __init__(self, system, name):
        pass


