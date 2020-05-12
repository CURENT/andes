import unittest

import andes
import numpy as np
from andes.core.common import DummyValue


class TestFlagNotNone(unittest.TestCase):
    def setUp(self) -> None:
        self.list = DummyValue(0)
        self.list.v = [0, 0, None, 2, 5.]

        self.array = DummyValue(0)
        self.array.v = np.array(self.list.v)

    def test_flag_not_none(self):
        self.fn = andes.core.service.FlagNotNone(self.list)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 0, 1, 1]))

        self.fn = andes.core.service.FlagNotNone(self.array)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 0, 1, 1]))


class TestParamCalc(unittest.TestCase):
    def setUp(self) -> None:
        self.p1 = DummyValue(0)
        self.p1.v = np.array([2, 4.5, 3, 8])

        self.p2 = DummyValue(0)
        self.p2.v = np.array([1, 2., 3, 1.0])

    def test_param_calc(self):
        self.pc = andes.core.service.ParamCalc(self.p1, self.p2, func=np.multiply)
        np.testing.assert_almost_equal(self.pc.v, np.array([2., 9., 9., 8.]))

        self.pc = andes.core.service.ParamCalc(self.p1, self.p2, func=np.add)
        np.testing.assert_almost_equal(self.pc.v, np.array([3, 6.5, 6, 9]))
