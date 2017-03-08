from cvxopt import matrix, sparse, spmatrix
from cvxopt import mul, div, log, sin, cos
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class GovernorBase(ModelBase):
    """Turbine governor base class"""
    def __init__(self, system, name):
        super(GovernorBase, self).__init__(system, name)
        self._group = 'Governor'
        self.remove_param('Vn')
        self.remove_param('Sn')
        self._data.update({'gen': None,
                           'pmax': 1.0,
                           'pmin': 0.0,
                           'R': 1.0,
                           'wref0': 1.0,
                           })
        self._descr.update({'gen': 'generator index',
                            'pmax': 'maximum turbine output',
                            'pmin': 'minimum turbine output',
                            'R': 'speed regulation droop',
                            'wref0': 'initial reference speed',
                            })
        self._params.extend(['pmax', 'pmin', 'R'])
        self._algebs.extend(['wref', 'porder'])
        self._service.extend(['pm0', 'gain'])
        self._ctrl.update({'gen': [{'model': 'Synchronous',
                                    'src': 'Sn'},
                                   {'model': 'Synchronous',
                                    'src': 'pm0'}
                                   ]
                           })

    def init1(self, dae):
        self.gain = div(1.0, self.R)
