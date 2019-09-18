"""
Timed event base class
"""

from cvxopt import matrix, mul

from .base import ModelBase


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
        Return a list of occurrance times of the events

        :return: list of times
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
                self.log('Applying generator trip on <{}> at t={}'.format(
                    gen_idx, sim_time))
                self.system.Synchronous.set_field('u', gen_idx, 0)
            if self.t2[i] == sim_time:
                self.log('Applying generator reconnect on <{}> at t={}'.format(
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
                self.log('Applying load shedding on <{}> at t={}'.format(
                    load_idx, sim_time))
                self.system.__dict__[group].set_field('u', load_idx, 0)
            if self.t2[i] == sim_time:
                self.log('Applying Gen reconnect on <{}> at t={}'.format(
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

        self._init()

    def apply(self, sim_time):
        super(LoadScale, self).apply(sim_time)

        for i in range(self.n):
            load_idx = self.load[i]
            group = self.group[i]

            if self.t1[i] == sim_time:
                self.log('Applying load scaling on <{}> at t={}'.format(
                    load_idx, sim_time))
                old_load = self.system.__dict__[group].get_field('p', load_idx)

                self.system.__dict__[group].set_field('p', load_idx, mul(old_load, self.scale) + self.inc)


class LoadRamp(ModelBase):
    """
    Continuous load ramping
    """
    def define(self):
        self.param_remove('Sn')
        self.param_remove('Vn')
        self._group = 'Event'

        self.param_define('load', default=None, mandatory=True, descr="load idx", tomatrix=True)
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

    def init1(self):
        # check the exclusivity of rate and amount

        # self.
        pass

    def gcall(self):
        # obtain the p0 and q0 at the time of the start

        # call the function to calculate the load (p and q) at the present time

        # apply the load

        pass

    def calc_p(self):
        pass

    def calc_q(self):
        pass
