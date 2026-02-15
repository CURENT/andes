import unittest

import numpy as np

from andes.core.block import HVGate, LVGate
from andes.core.common import DummyValue
from andes.core.var import Algeb


class TestGates(unittest.TestCase):

    def setUp(self) -> None:
        self.n = 5
        self.u1 = DummyValue(0)
        self.u1.v = np.zeros(self.n)

        self.u2 = Algeb(tex_name='u2')
        self.u2.v = np.array([-2, -1, 0, 1, 2])

    def test_hvgate(self):
        """
        Test `andes.core.discrete.HVGate`
        """
        self.hv = HVGate(self.u1, self.u2)
        self.hv.lt.list2array(self.n)
        self.hv.lt.check_var()
        np.testing.assert_almost_equal(self.hv.lt.z0, np.array([1, 1, 1, 0, 0]))
        np.testing.assert_almost_equal(self.hv.lt.z1, np.array([0, 0, 0, 1, 1]))

    def test_lvgate(self):
        """
        Test `andes.core.discrete.LVGate`
        """

        self.lv = LVGate(self.u1, self.u2)
        self.lv.lt.list2array(self.n)
        self.lv.lt.check_var()
        np.testing.assert_almost_equal(self.lv.lt.z1, np.array([0, 0, 0, 1, 1]))
        np.testing.assert_almost_equal(self.lv.lt.z0, np.array([1, 1, 1, 0, 0]))
