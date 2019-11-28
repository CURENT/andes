"""
Timed event base class
"""
import logging

from cvxopt import matrix, mul, spmatrix
from andes.utils.math import zeros

from .base import ModelBase
import numpy.random

logger = logging.getLogger(__name__)


class EventBase(ModelBase):
    """
    Base class for timed events
    """

    def define(self):
        self.param_remove('Sn')
        self.param_remove('Vn')
        self._group = 'Event'
        self._last_time = -1

    def get_times(self):
        """
        Return a list of occurrance switch_times of the events

        :return: list of switch_times
        """
        if not self.n:
            return list()

        ret = list()

        for item in self._event_times:
            ret += list(self.__dict__[item])

        return ret + list(matrix(ret) - 1e-6)

    def is_time(self, t):
        """
        Return the truth value of whether ``t`` is a time in
        ``self.get_times()``

        :param float t: time to check for
        :return bool: truth value
        """
        if not self.n:
            return False
        else:
            return t in self.get_times()

    def apply(self, sim_time):
        """
        Apply the event and
        :param sim_time:
        :return:
        """
        if not self.n:
            return

        # skip if already applied
        if self._last_time != sim_time:
            self._last_time = sim_time
        else:
            return


class GenTrip(EventBase):
    """
    Timed generator trip and reconnect
    """

    def define(self):
        super(GenTrip, self).define()
        self.param_define(
            'gen',
            default=None,
            mandatory=True,
            descr='generator idx',
            tomatrix=False)
        self.param_define(
            't1',
            default=-1,
            mandatory=False,
            descr='generator trip time',
            event_time=True)
        self.param_define(
            't2',
            default=-1,
            mandatory=False,
            descr='generator reconnect time',
            event_time=True)

        self._init()

    def apply(self, sim_time):
        super(GenTrip, self).apply(sim_time)
        for i in range(self.n):
            gen_idx = self.gen[i]

            if self.t1[i] == sim_time:
                logger.info('Applying generator trip on <{}> at t={}'.format(
                    gen_idx, sim_time))
                self.system.Synchronous.set_field('u', gen_idx, 0)
            if self.t2[i] == sim_time:
                logger.info('Applying generator reconnect on <{}> at t={}'.format(
                    gen_idx, sim_time))
                self.system.Synchronous.set_field('u', gen_idx, 1)


class LoadShed(EventBase):
    """
    Timed load shedding and reconnect
    """

    def define(self):
        super(LoadShed, self).define()
        self.param_define(
            'load',
            default=None,
            mandatory=True,
            descr='load idx',
            tomatrix=False)
        self.param_define(
            'group',
            default='StaticLoad',
            mandatory=True,
            descr='load group, StaticLoad or DynLoad',
            tomatrix=False)
        self.param_define(
            't1',
            default=-1,
            mandatory=False,
            descr='load shedding time',
            event_time=True)
        self.param_define(
            't2',
            default=-1,
            mandatory=False,
            descr='load reconnect time',
            event_time=True)

        self._init()

    def apply(self, sim_time):
        super(LoadShed, self).apply(sim_time)

        for i in range(self.n):
            load_idx = self.load[i]
            group = self.group[i]

            if self.t1[i] == sim_time:
                logger.info('Applying load shedding on <{}> at t={}'.format(
                    load_idx, sim_time))
                self.system.__dict__[group].set_field('u', load_idx, 0)
            if self.t2[i] == sim_time:
                logger.info('Applying Gen reconnect on <{}> at t={}'.format(
                    load_idx, sim_time))
                self.system.__dict__[group].set_field('u', load_idx, 1)


class LoadScale(EventBase):
    """
    Timed load scaling or increment
    """

    def define(self):
        super(LoadScale, self).define()
        self.param_define(
            'load',
            default=None,
            mandatory=True,
            descr='load idx',
            tomatrix=False)
        self.param_define(
            'group',
            default='StaticLoad',
            mandatory=True,
            descr='load group, StaticLoad or DynLoad',
            tomatrix=False)
        self.param_define(
            't1',
            default=-1,
            mandatory=False,
            descr='load scaling time',
            event_time=True)
        self.param_define('scale', default=1, mandatory=False, descr='load scaling factor', to_matrix=True)
        self.param_define('inc', default=0, mandatory=False, descr='load increment value', to_matrix=True)
        self.param_define('rand', default=0, mandatory=False,
                          descr='Multiply the load inc by a Gaussian (0, 0.1) rand', to_matrix=True)

        self._rand_coeff = []
        self._init()

    def apply(self, sim_time):
        super(LoadScale, self).apply(sim_time)

        if len(self._rand_coeff) == 0:
            self._rand_coeff = numpy.random.rand(self.n)

        for i in range(self.n):
            load_idx = self.load[i]
            group = self.group[i]

            if self.t1[i] == sim_time:
                logger.info('Applying load scaling on <{}> at t={}'.format(
                    load_idx, sim_time))
                old_load = self.system.__dict__[group].get_field('p', load_idx)

                new_load = mul(old_load, self.scale[i])
                if self.rand[i]:
                    new_load += self.inc[i] * self._rand_coeff[i]
                else:
                    new_load += self.inc[i]

                self.system.__dict__[group].set_field('p', load_idx, new_load)


class LoadRamp(ModelBase):
    """
    Continuous load ramping
    """
    def define(self):
        self.param_remove('Sn')
        self.param_remove('Vn')
        self._group = 'Event'

        self.param_define('load', default=None, mandatory=True, descr="load idx", tomatrix=False)
        self.param_define('group', default='StaticLoad', mandatory=True,
                          descr="load group, StaticLoad or DynLoad", tomatrix=False)
        self.param_define('t1', default=-1, mandatory=True, descr='start time', tomatrix=True)
        self.param_define('t2', default=-1, mandatory=True, descr='end time', tomatrix=True)
        self.param_define('p_rate', default=1, mandatory=False, descr='rate of ramping per hour in percentage',
                          tomatrix=True)
        self.param_define('p_amount', default=0, mandatory=False, descr='the amount of ramping per hour in pu',
                          tomatrix=True)
        self.param_define('q_rate', default=1, mandatory=False, descr='rate of ramping per hour in percentage',
                          tomatrix=True)
        self.param_define('q_amount', default=0, mandatory=False, descr='the amount of ramping per hour in pu',
                          tomatrix=True)

        self.service_define("p0", matrix)
        self.service_define("q0", matrix)

        self.service_define("p_out", matrix)
        self.service_define("q_out", matrix)

        self.calls.update({'gcall': True,
                           'init1': True,
                           })

        self._init()

    def init1(self, dae):
        # check the exclusivity of rate and amount

        # obtain the p0 and q0 at the time of the start
        self.copy_data_ext("StaticLoad", field="p0", dest="p0", idx=self.load)
        self.copy_data_ext("StaticLoad", field="q0", dest="q0", idx=self.load)
        self.copy_data_ext("StaticLoad", field="a", dest="a", idx=self.load)
        self.copy_data_ext("StaticLoad", field="v", dest="v", idx=self.load)

        self.p_out = zeros(self.n, 1)
        self.q_out = zeros(self.n, 1)

    def gcall(self, dae):
        # call the function to calculate the load (p and q) at the present time
        if dae.t < 0:
            return
        # calculate the load increase
        self.calc_p(dae.t)
        self.calc_q(dae.t)

        # apply the load change to the bus equations
        dae.g += spmatrix(self.p_out, self.a, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(self.q_out, self.v, [0] * self.n, (dae.m, 1), 'd')

    def calc_p(self, t):
        for i in range(self.n):
            if t < self.t1[i]:
                self.p_out[i] = 0
            elif t > self.t2[i]:
                continue
            else:
                self.p_out[i] = (self.p_rate[i] * self.p0[i] / 60 / 60) * (t - self.t1[i]) +\
                                (self.p_amount[i] / 60 / 60) * (t - self.t1[i])

    def calc_q(self, t):
        for i in range(self.n):
            if t < self.t1[i]:
                self.q_out[i] = 0
            elif t > self.t2[i]:
                continue
            else:
                self.q_out[i] = (self.q_rate[i] * self.q0[i] / 60 / 60) * (t - self.t1[i]) +\
                                (self.q_amount[i] / 60 / 60) * (t - self.t1[i])

    def get_times(self):
        return []

    def is_time(self, t):
        return False
