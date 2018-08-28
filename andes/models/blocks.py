"""Control blocks"""
from cvxopt import matrix, sparse, spmatrix  ## NOQA
from cvxopt import mul, div, log, sin, cos  ## NOQA
from .base import ModelBase  ## NOQA
# from ..consts import *
# from ..utils.math import *


class PI1(object):
    """PI controller class as addon base class"""

    def __init__(self, params=None, inputs=None, outputs=None):

        assert type(outputs) == list

        if hasattr(self, 'nPI1'):
            self.nPI1 += 1
        else:
            self.nPI1 = 1

        if hasattr(self, 'PI1_label'):
            self.PI1_label.append('PI1_' + str(self.nPI1))
        else:
            self.PI1_label = ['PI1' + str(self.nPI1)]

        if params:
            assert type(params) == dict
        else:
            params = {
                'Kp' + str(self.nPI1): 0.1,
                'Ki' + str(self.nPI1): 0,
            }

        self._algebs = ['']
        if inputs:
            assert type(inputs) == list
        else:
            inputs = ['PI_IN_']
