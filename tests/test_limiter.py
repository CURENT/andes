import unittest

from andes.core.var import Algeb
from andes.core.param import NumParam
from andes.core.discrete import Comparer, SortedLimiter

import numpy as np


class TestDiscrete(unittest.TestCase):
    def setUp(self):
        self.lower = NumParam()
        self.upper = NumParam()
        self.var = Algeb()

        self.upper.v = np.array([2,    2,   2, 2,   2,   2, 2.8, 3.9])
        self.var.v = np.array([-3, -1.1,  -5, 0,   1,   2,   3,  10])
        self.lower.v = np.array([-2,   -1, 0.5, 0, 0.5, 1.5,   2,   3])

    def test_comparer(self):
        """
        Tests for `Comparer` class
        Returns
        -------

        """
        self.cmp = Comparer(self.var, self.lower, self.upper)
        self.cmp.check_var()

        self.assertSequenceEqual(self.cmp.zl.tolist(),
                                 [1., 1., 1., 1., 0., 0., 0., 0.])
        self.assertSequenceEqual(self.cmp.zi.tolist(),
                                 [0., 0., 0., 0., 1., 0., 0., 0.])
        self.assertSequenceEqual(self.cmp.zu.tolist(),
                                 [0., 0., 0., 0., 0., 1., 1., 1.])

    def test_sorted_limiter(self):
        """
        Tests for `SortedLimiter` class

        Returns
        -------

        """
        self.cmp = Comparer(self.var, self.lower, self.upper)
        self.cmp.check_var()

        self.rcmp = SortedLimiter(self.var, self.lower, self.upper, n_select=1)
        self.rcmp.check_var()

        self.assertSequenceEqual(self.rcmp.zl.tolist(),
                                 [0., 0., 1., 0., 0., 0., 0., 0.])
        self.assertSequenceEqual(self.rcmp.zi.tolist(),
                                 [1., 1., 0., 1., 1., 1., 1., 0.])
        self.assertSequenceEqual(self.rcmp.zu.tolist(),
                                 [0., 0., 0., 0., 0., 0., 0., 1.])

        # test when no `n_select` is specified
        self.rcmp_noselect = SortedLimiter(self.var, self.lower, self.upper)
        self.rcmp_noselect.check_var()

        self.assertSequenceEqual(self.rcmp_noselect.zl.tolist(),
                                 self.cmp.zl.tolist())
        self.assertSequenceEqual(self.rcmp_noselect.zi.tolist(),
                                 self.cmp.zi.tolist())
        self.assertSequenceEqual(self.rcmp_noselect.zu.tolist(),
                                 self.cmp.zu.tolist())

        # test when no `n_select` is over range
        self.rcmp_noselect = SortedLimiter(self.var, self.lower, self.upper,
                                           n_select=999)
        self.rcmp_noselect.check_var()

        self.assertSequenceEqual(self.rcmp_noselect.zl.tolist(),
                                 self.cmp.zl.tolist())
        self.assertSequenceEqual(self.rcmp_noselect.zi.tolist(),
                                 self.cmp.zi.tolist())
        self.assertSequenceEqual(self.rcmp_noselect.zu.tolist(),
                                 self.cmp.zu.tolist())
