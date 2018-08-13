"""
Timed event base class
"""

from .base import ModelBase
from cvxopt import matrix


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
        Return the truth value of whether ``t`` is a time in ``self.get_times()``

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
        self.param_add('gen', default=None, mandatory=True, descr='generator idx', tomatrix=False)
        self.param_add('t1', default=-1, mandatory=False, descr='generator trip time', event_time=True)
        self.param_add('t2', default=-1, mandatory=False, descr='generator reconnect time', event_time=True)

        self._meta_to_attr()

    def apply(self, sim_time):
        super(GenTrip, self).apply(sim_time)
        for i in range(self.n):
            gen_idx = self.gen[i]

            if self.t1[i] == sim_time:
                self.message('Applying Gen trip on <{}> at t={}'.format(gen_idx, sim_time))
                self.system.Synchronous.set_field('u', gen_idx, 0)
            if self.t2[i] == sim_time:
                self.message('Applying Gen reconnect on <{}> at t={}'.format(gen_idx, sim_time))
                self.system.Synchronous.set_field('u', gen_idx, 1)
