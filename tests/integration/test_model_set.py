import unittest
import numpy as np

import andes


class TestModelMethods(unittest.TestCase):
    """
    Test methods of Model.
    """

    def test_model_set(self):
        """
        Test `Model.set()` method.
        """

        ss = andes.run(
            andes.get_case("ieee14/ieee14.json"),
            default_config=True,
            no_output=True,
        )
        ss.TDS.init()

        omega_addr = ss.GENROU.omega.a.tolist()

        # set a single value
        ss.GENROU.set("M", "GENROU_1", "v", 2.0)
        self.assertEqual(ss.GENROU.M.v[0], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[0], omega_addr[0]], 2.0)

        # set a list of values
        ss.GENROU.set("M", ["GENROU_1", "GENROU_2"], "v", [2.0, 3.5])
        np.testing.assert_equal(ss.GENROU.M.v[[0, 1]], [2.0, 3.5])
        self.assertEqual(ss.TDS.Teye[omega_addr[0], omega_addr[0]], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[1], omega_addr[1]], 3.5)

        # set a list of values
        ss.GENROU.set("M", ["GENROU_3", "GENROU_5"], "v", [2.0, 3.5])
        np.testing.assert_equal(ss.GENROU.M.v[[2, 4]], [2.0, 3.5])
        self.assertEqual(ss.TDS.Teye[omega_addr[2], omega_addr[2]], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[4], omega_addr[4]], 3.5)

        # set a list of idxes with a single element to an array of values
        ss.GENROU.set("M", ["GENROU_4"], "v", np.array([4.0]))
        np.testing.assert_equal(ss.GENROU.M.v[3], 4.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 4.0)

        # set an array of idxes with a single element to an array of values
        ss.GENROU.set("M", np.array(["GENROU_4"]), "v", np.array([5.0]))
        np.testing.assert_equal(ss.GENROU.M.v[3], 5.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 5.0)

        # set an array of idxes with a list of single value
        ss.GENROU.set("M", np.array(["GENROU_4"]), "v", 6.0)
        np.testing.assert_equal(ss.GENROU.M.v[3], 6.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 6.0)

        # set when destination idx is list
        ss.Bus.set(src='name', attr='v', idx=(1, 2, 3), value=['A', 'B', 'C'])
        self.assertEqual(ss.Bus.name.v[:3], ['A', 'B', 'C'])

    def test_find_idx(self):
        ss = andes.load(andes.get_case('ieee14/ieee14_pvd1.xlsx'))
        mdl = ss.PVD1

        # not allow all matches
        self.assertListEqual(mdl.find_idx(keys='gammap', values=[0.1], allow_all=False),
                             [1])

        # allow all matches
        self.assertListEqual(mdl.find_idx(keys='gammap', values=[0.1], allow_all=True),
                             [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # multiple values
        self.assertListEqual(mdl.find_idx(keys='name', values=['PVD1_1', 'PVD1_2'],
                                          allow_none=False, default=False),
                             [1, 2])
        # non-existing value
        self.assertListEqual(mdl.find_idx(keys='name', values=['PVD1_999'],
                                          allow_none=True, default=False),
                             [False])

        # non-existing value is not allowed
        with self.assertRaises(IndexError):
            mdl.find_idx(keys='name', values=['PVD1_999'],
                         allow_none=False, default=False)

        # multiple keys
        self.assertListEqual(mdl.find_idx(keys=['gammap', 'name'],
                                          values=[[0.1, 0.1], ['PVD1_1', 'PVD1_2']]),
                             [1, 2])

        # multiple keys, with non-existing values
        self.assertListEqual(mdl.find_idx(keys=['gammap', 'name'],
                                          values=[[0.1, 0.1], ['PVD1_1', 'PVD1_999']],
                                          allow_none=True, default='CURENT'),
                             [1, 'CURENT'])

        # multiple keys, with non-existing values not allowed
        with self.assertRaises(IndexError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[[0.1, 0.1], ['PVD1_1', 'PVD1_999']],
                         allow_none=False, default=999)

        # multiple keys, values are not iterable
        with self.assertRaises(ValueError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[0.1, 0.1])

        # multiple keys, items length are inconsistent in values
        with self.assertRaises(ValueError):
            mdl.find_idx(keys=['gammap', 'name'],
                         values=[[0.1, 0.1], ['PVD1_1']])

    def test_model_alter(self):
        """
        Test `Model.alter()` method.
        """

        ss = andes.run(
            andes.get_case('5bus/pjm5bus.xlsx'),
            default_config=True,
            no_output=True,
        )
        ss.TDS.init()

        # alter `v`
        ss.GENCLS.alter(src='M', idx=2, value=1, attr='v')
        self.assertEqual(ss.GENCLS.M.v[1], 1 * ss.GENCLS.M.pu_coeff[1])

        # alter `vin`
        ss.GENCLS.alter(src='M', idx=2, value=1, attr='vin')
        self.assertEqual(ss.GENCLS.M.v[1], 1)

        # alter `vin` on instances without `vin` falls back to `v`
        ss.GENCLS.alter(src='p0', idx=2, value=1, attr='vin')
        self.assertEqual(ss.GENCLS.p0.v[1], 1)

        # # alter `v` when destination idx is list
        ss.Bus.alter(src='name', idx=[0, 1], value=['A', 'B'], attr='vin')
        np.testing.assert_equal(ss.Bus.name.v[:2], ['A', 'B'])
