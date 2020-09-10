import unittest

import andes
import numpy as np
from andes.core.common import DummyValue


class TestFlagValue(unittest.TestCase):
    def setUp(self) -> None:
        self.list = DummyValue(0)
        self.list.v = [0, 0, None, 2, 5.]

        self.array = DummyValue(0)
        self.array.v = np.array(self.list.v)

    def test_flag_not_none(self):
        self.fn = andes.core.service.FlagValue(self.list, value=None)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 0, 1, 1]))

        self.fn = andes.core.service.FlagValue(self.array, value=None)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 0, 1, 1]))


class TestFlagCondition(unittest.TestCase):
    def setUp(self) -> None:
        self.list = DummyValue(0)
        self.list.v = [0, 0, -1, -2, 5.]

        self.array = DummyValue(0)
        self.array.v = np.array(self.list.v)

    def test_flag_cond(self):
        self.fn = andes.core.service.FlagCondition(self.list, func=lambda x: np.less(x, 0))
        np.testing.assert_almost_equal(self.fn.v, np.array([0, 0, 1, 1, 0]))

    def test_flag_less_than(self):
        self.fn = andes.core.service.FlagLessThan(self.list)
        np.testing.assert_almost_equal(self.fn.v, np.array([0, 0, 1, 1, 0]))

    def test_flag_less_equal(self):
        self.fn = andes.core.service.FlagLessThan(self.list, equal=True)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 1, 1, 0]))

    def test_flag_greater_than(self):
        self.fn = andes.core.service.FlagGreaterThan(self.list)
        np.testing.assert_almost_equal(self.fn.v, np.array([0, 0, 0, 0, 1]))

    def test_flag_greater_equal(self):
        self.fn = andes.core.service.FlagGreaterThan(self.list, equal=True)
        np.testing.assert_almost_equal(self.fn.v, np.array([1, 1, 0, 0, 1]))

    def test_apply_func(self):
        self.fn = andes.core.service.ApplyFunc(self.list, np.abs)
        np.testing.assert_almost_equal(self.fn.v, np.array([0, 0, 1, 2, 5]))


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


class TestBackRef(unittest.TestCase):
    def test_backref_ieee14(self):
        ss = andes.load(andes.get_case("ieee14/ieee14_gentrip.xlsx"))

        self.assertSequenceEqual(ss.Area.Bus.v[0], [1, 2, 3, 4, 5])
        self.assertSequenceEqual(ss.Area.Bus.v[1], [6, 7, 8, 9, 10, 11, 12, 13, 14])

        self.assertSequenceEqual(ss.StaticGen.SynGen.v[0], ['GENROU_2'])
        self.assertSequenceEqual(ss.SynGen.TurbineGov.v[0], ['TGOV1_1'])
