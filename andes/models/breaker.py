from cvxopt import matrix, spmatrix  # NOQA
from cvxopt import mul, div, exp  # NOQA
from ..consts import *  # NOQA
from ..utils.math import neg
from .base import ModelBase
import logging

logger = logging.getLogger(__name__)


class Breaker(ModelBase):
    """Simple line breaker model"""

    def __init__(self, system, name):
        super(Breaker, self).__init__(system, name)
        self._group = 'Relay'
        self._name = 'Breaker'
        self._data.update({
            'bus': None,
            'line': None,
            't1': 0.0,
            't2': 0.0,
            't3': 0.0,
            't4': 0.0,
            'u1': False,
            'u2': False,
            'u3': False,
            'u4': False,
            'fn': 60.0,
        })
        self._params.extend(['t1', 't2', 't3', 't4', 'u1', 'u2', 'u3', 'u4'])
        self._descr.update({
            'bus': 'Bus idx',
            'line': 'Line idx',
            't1': 'Time of the 1st switch',
            't2': 'Time of the 2nd switch',
            't3': 'Time of the 3rd switch',
            't4': 'Time of the 4th switch',
            'u1': 'Apply the 1st switch',
            'u2': 'Apply the 2nd switch',
            'u3': 'Apply the 3rd switch',
            'u4': 'Apply the 4th switch',
            'fn': 'rated frequency',
        })
        self._units.update({
            't1': 's',
            't2': 's',
            't3': 's',
            't4': 's',
            'u1': 'boolean',
            'u2': 'boolean',
            'u3': 'boolean',
            'u4': 'boolean',
        })
        self._mandatory.extend(['bus', 'line'])
        self._service.extend(['times', 'time'])
        self.param_remove('Sn')
        self._init()

    def setup(self):
        super(Breaker, self).setup()
        # check if `self.bus` is connected by `self.line`
        self.copy_data_ext('Line', 'bus1', idx=self.line)
        self.copy_data_ext('Line', 'bus2', idx=self.line)
        for i in range(self.n):
            if self.bus[i] != self.bus1[i] and self.bus[i] != self.bus2[i]:
                logger.warning(
                    '<Breaker> {} on line {} and bus {} is incorrect '
                    'and is thus disabled.'.
                    format(self.idx[i], self.line[i], self.bus[i]))
                self.u[i] = 0

    def get_times(self):
        """Return all the action times and times-1e-6 in a list"""
        if not self.n:
            return []
        self.times = list(mul(self.u1, self.t1)) + \
            list(mul(self.u2, self.t2)) + \
            list(mul(self.u3, self.t3)) + \
            list(mul(self.u4, self.t4))

        self.times = matrix(list(set(self.times)))
        self.times = list(self.times) + list(self.times - 1e-6)
        return self.times

    def is_time(self, t):
        if not self.n:
            return
        return t in self.times

    def apply(self, actual_time):
        if self.time != actual_time:
            self.time = actual_time
        else:
            return

        for i in range(self.n):
            tn = matrix([self.t1[i], self.t2[i], self.t3[i], self.t4[i]])
            tn = mul(self.u[i], tn)
            if actual_time in tn:

                line_int = self.system.Line.uid[self.line[i]]
                u0 = self.system.Line.u[line_int]
                self.system.Line.switch(self.line[i], neg(u0))

                if u0 == 1:
                    inf = ' Breaker <{}>: Line <{}> disconnected ' \
                          'at t = {}.'.format(
                              self.idx[i], self.line[i], actual_time)
                elif u0 == 0:
                    inf = ' Breaker <{}>: Line <{}> reconnected ' \
                          'at t = {}.'.format(
                              self.idx[i], self.line[i], actual_time)
                logger.info(inf)
        self.system.check_islands()

    def insert(self, idx=None, name=None, **kwargs):
        if self.n:
            self._param_to_list()

        self.elem_add(idx, name, **kwargs)
        self._param_to_matrix()
