import unittest

from andes.core.common import DummyValue
from andes.core.discrete import (Average, Delay, Derivative, Limiter,
                                 SortedLimiter, Switcher,)
from andes.core.param import NumParam
from andes.core.var import Algeb
from andes.shared import np


class TestDiscrete(unittest.TestCase):
    def setUp(self):
        self.lower = NumParam()
        self.upper = NumParam()
        self.u = Algeb()

        self.upper.v = np.array([2,    2,   2, 2,   2,   2, 2.8, 3.9])
        self.u.v = np.array([-3, -1.1,  -5, 0,   1,   2,   3,  10])
        self.lower.v = np.array([-2,   -1, 0.5, 0, 0.5, 1.5,   2,   3])

    def test_limiter(self):
        """
        Tests for `Limiter` class
        Returns
        -------

        """
        cmp = Limiter(self.u, self.lower, self.upper)
        cmp.list2array(len(self.u.v))

        cmp.check_var()

        self.assertSequenceEqual(cmp.zl.tolist(),
                                 [1., 1., 1., 1., 0., 0., 0., 0.])
        self.assertSequenceEqual(cmp.zi.tolist(),
                                 [0., 0., 0., 0., 1., 0., 0., 0.])
        self.assertSequenceEqual(cmp.zu.tolist(),
                                 [0., 0., 0., 0., 0., 1., 1., 1.])

    def test_sorted_limiter(self):
        """
        Tests for `SortedLimiter` class

        Returns
        -------

        """
        cmp = Limiter(self.u, self.lower, self.upper)
        cmp.list2array(len(self.u.v))
        cmp.check_var()

        rcmp = SortedLimiter(self.u, self.lower, self.upper, n_select=1)
        rcmp.list2array(len(self.u.v))
        rcmp.check_var()

        self.assertSequenceEqual(rcmp.zl.tolist(),
                                 [0., 0., 1., 0., 0., 0., 0., 0.])
        self.assertSequenceEqual(rcmp.zi.tolist(),
                                 [1., 1., 0., 1., 1., 1., 1., 0.])
        self.assertSequenceEqual(rcmp.zu.tolist(),
                                 [0., 0., 0., 0., 0., 0., 0., 1.])

        # test when no `n_select` is specified
        rcmp_noselect = SortedLimiter(self.u, self.lower, self.upper)
        rcmp_noselect.list2array(len(self.u.v))
        rcmp_noselect.check_var()

        self.assertSequenceEqual(rcmp_noselect.zl.tolist(),
                                 cmp.zl.tolist())
        self.assertSequenceEqual(rcmp_noselect.zi.tolist(),
                                 cmp.zi.tolist())
        self.assertSequenceEqual(rcmp_noselect.zu.tolist(),
                                 cmp.zu.tolist())

        # test when no `n_select` is over range
        rcmp_noselect = SortedLimiter(self.u, self.lower, self.upper,
                                           n_select=999)
        rcmp_noselect.list2array(len(self.u.v))
        rcmp_noselect.check_var()

        self.assertSequenceEqual(rcmp_noselect.zl.tolist(),
                                 cmp.zl.tolist())
        self.assertSequenceEqual(rcmp_noselect.zi.tolist(),
                                 cmp.zi.tolist())
        self.assertSequenceEqual(rcmp_noselect.zu.tolist(),
                                 cmp.zu.tolist())

    def test_switcher(self):
        p = NumParam()
        p.v = np.array([0, 1, 2, 2, 1, 3, 1])
        switcher = Switcher(u=p, options=[0, 1, 2, 3, 4])
        switcher.list2array(len(p.v))

        switcher.check_var()

        self.assertSequenceEqual(switcher.s0.tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(switcher.s1.tolist(), [0, 1, 0, 0, 1, 0, 1])
        self.assertSequenceEqual(switcher.s2.tolist(), [0, 0, 1, 1, 0, 0, 0])
        self.assertSequenceEqual(switcher.s3.tolist(), [0, 0, 0, 0, 0, 1, 0])
        self.assertSequenceEqual(switcher.s4.tolist(), [0, 0, 0, 0, 0, 0, 0])


class TestDelay(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 5   # number of input values to delay
        self.step = 2  # steps of delay
        self.time = 1.0  # delay period in second

        self.data = DummyValue(0)
        self.data.v = np.zeros(self.n)

        self.dstep = Delay(u=self.data, mode='step', delay=self.step)
        self.dstep.list2array(self.n)

        self.dtime = Delay(u=self.data, mode='time', delay=self.time)
        self.dtime.list2array(self.n)

        self.avg = Average(u=self.data, mode='step', delay=1)
        self.avg.list2array(self.n)

        self.v = self.dstep.v
        self.vt = self.dtime.v
        self.va = self.avg.v

        self.n_forward = 5
        self.tstep = 0.2
        self.dae_t = 0
        self.k = 0.2

    def test_delay_step(self):
        for i in range(self.n_forward):
            self.dstep.check_var(i)
            self.data.v += 1

        self.assertSequenceEqual(self.v.tolist(), [self.n_forward - self.step - 1] * self.n)

    def test_delay_time(self):
        self.n_forward = 10
        for i in range(self.n_forward):
            self.data.v[:] = self.dae_t
            self.dtime.check_var(self.dae_t)
            self.dae_t += self.tstep

        np.testing.assert_almost_equal(self.vt, [(self.n_forward - 1) * self.tstep - self.time] * self.n)

    def test_average(self):
        for i in range(self.n_forward):
            self.data.v[:] = self.dae_t * self.k
            self.avg.check_var(self.dae_t)

            if self.dae_t == 0:
                np.testing.assert_almost_equal(self.va, 0)
            else:
                np.testing.assert_almost_equal(self.va, 0.5 * (2 * self.dae_t - self.tstep) * self.k)

            self.dae_t += self.tstep


class TestDerivative(unittest.TestCase):
    """
    Test `andes.core.discrete.Derivative`
    """

    def setUp(self) -> None:
        self.n = 5
        self.data = DummyValue(0)
        self.data.v = np.zeros(self.n)

        self.der = Derivative(u=self.data)
        self.der.list2array(self.n)
        self.v = self.der.v

        self.n_forward = 10
        self.k = 0.2
        self.t_step = 0.1
        self.dae_t = 0

    def test_derivative(self):
        for i in range(self.n_forward):
            self.data.v[:] = self.dae_t * self.k
            self.der.check_var(self.dae_t)

            if self.dae_t == 0:
                np.testing.assert_almost_equal(self.v, 0)
            else:
                np.testing.assert_almost_equal(self.v, self.k)

            self.dae_t += self.t_step
