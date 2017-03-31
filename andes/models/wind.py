"""Wind power classes"""
from cvxopt import matrix, mul, spmatrix, div, sin, cos
from .base import ModelBase
from ..utils.math import *
from ..consts import *


class WindBase(ModelBase):
    """Base class for wind time series"""
    def __init__(self, system, name):
        super(WindBase, self).__init__(system, name)
        self.remove_param('Sn')
        self.remove_param('Vn')
        self.data.update({'T': 0.01,
                          'Vwn': 15,
                          'dt': 0.1,
                          'rho': 1.225,
                          })
        self._descr.update({'T': 'Filter time constant',
                            'Vwn': 'Wind speed base',
                            'dt': 'Sampling time step',
                            'rho': 'Air density',
                            })
        self._units.update({'T': 's',
                            'Vwn': 'm/s',
                            'rho': 'kg/m^3',
                            'dt': 's',
                            })
        self.params.extend(['T', 'Vwn', 'dt', 'rho'])
