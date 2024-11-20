"""
Tests for group functions.
"""
import unittest

import numpy as np

import andes


class TestGroup(unittest.TestCase):
    """
    Test the group class functions.
    """

    def setUp(self):
        self.ss = andes.run(andes.get_case("ieee14/ieee14_pvd1.xlsx"),
                            default_config=True,
                            no_output=True,
                            )
        self.ss.config.warn_limits = 0
        self.ss.config.warn_abnormal = 0

    def test_group_access(self):
        """
        Test methods such as `idx2model`
        """
        ss = self.ss

        # --- idx2uid ---
        self.assertIsNone(ss.DG.idx2uid(None))
        self.assertListEqual(ss.DG.idx2uid([None]), [None])

        # --- idx2model ---
        # what works
        self.assertIs(ss.DG.idx2model(1), ss.PVD1)
        self.assertListEqual(ss.DG.idx2model([1]), [ss.PVD1])
        self.assertListEqual(ss.DG.idx2model([1, 2]), [ss.PVD1, ss.PVD1])
        self.assertListEqual(ss.DG.idx2model((1, 2)), [ss.PVD1, ss.PVD1])
        self.assertListEqual(ss.DG.idx2model(np.array((1, 2))), [ss.PVD1, ss.PVD1])

        # what does not work
        self.assertRaises(KeyError, ss.DG.idx2model, idx='1')
        self.assertRaises(KeyError, ss.DG.idx2model, idx=88)
        self.assertRaises(KeyError, ss.DG.idx2model, idx=[1, 88])

        # --- get ---
        self.assertRaises(KeyError, ss.DG.get, 'xc', 999)

        np.testing.assert_equal(ss.DG.get('xc', 1,), 1)

        np.testing.assert_equal(ss.DG.get('xc', [1, 2, 3], allow_none=True),
                                [1, 1, 1])

        np.testing.assert_equal(ss.DG.get('xc',
                                          [1, 2, None],
                                          allow_none=True,
                                          default=999),
                                [1, 1, 999])

        # --- set ---
        ss.DG.set('xc', 1, 'v', 2.0)
        np.testing.assert_equal(ss.DG.get('xc', [1, 2]), [2, 1])

        ss.DG.set('xc', (1, 2, 3), 'v', 2.0)
        np.testing.assert_equal(ss.DG.get('xc', [1, 2, 3, 4]),
                                [2, 2, 2, 1])

        ss.DG.set('xc', (1, 2, 3), 'v', [6, 7, 8])
        np.testing.assert_equal(ss.DG.get('xc', [1, 2, 3, 4]),
                                [6, 7, 8, 1])

        # --- find_idx ---
        # same Model
        self.assertListEqual(ss.DG.find_idx('name', ['PVD1_1', 'PVD1_2']),
                             ss.PVD1.find_idx('name', ['PVD1_1', 'PVD1_2']),
                             )

        self.assertListEqual(ss.DG.find_idx(['name', 'Sn'],
                                            [('PVD1_1', 'PVD1_2'),
                                             (1.0, 1.0)]),
                             ss.PVD1.find_idx(['name', 'Sn'],
                                              [('PVD1_1', 'PVD1_2'),
                                               (1.0, 1.0)]))

        # cross Model, given results
        self.assertListEqual(ss.StaticGen.find_idx(keys='bus',
                                                   values=[1, 2, 3, 4]),
                             [1, 2, 3, 6])
        self.assertListEqual(ss.StaticGen.find_idx(keys='bus',
                                                   values=[1, 2, 3, 4],
                                                   allow_all=True),
                             [[1], [2], [3], [6]])

        self.assertListEqual(ss.StaticGen.find_idx(keys='bus',
                                                   values=[1, 2, 3, 4, 2024],
                                                   allow_none=True,
                                                   default=2011,
                                                   allow_all=True),
                             [[1], [2], [3], [6], [2011]])

        # --- get_field ---
        ff = ss.DG.get_field('f', list(ss.DG._idx2model.keys()), 'v_code')
        self.assertTrue(any([item == 'y' for item in ff]))

        # --- get group idx ---
        self.assertSetEqual(set(ss.DG.get_all_idxes()),
                            set(ss.PVD1.idx.v))
        self.assertSetEqual(set(ss.StaticGen.get_all_idxes()),
                            set(ss.PV.idx.v + ss.Slack.idx.v))


class TestGroupAdditional(unittest.TestCase):
    """
    Test additional group functions.
    """

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('5bus/pjm5bus.xlsx'),
            setup=True,
            default_config=True,
            no_output=True,
        )

    def test_group_alter(self):
        """
        Test `Group.alter` method.
        """

        # alter `v`
        self.ss.SynGen.alter(src='M', idx=2, value=1, attr='v')
        self.assertEqual(self.ss.GENCLS.M.v[1],
                         1 * self.ss.GENCLS.M.pu_coeff[1])

        # alter `vin`
        self.ss.SynGen.alter(src='M', idx=2, value=1, attr='vin')
        self.assertEqual(self.ss.GENCLS.M.v[1], 1)

        # alter `vin` on instances without `vin` falls back to `v`
        self.ss.SynGen.alter(src='p0', idx=2, value=1, attr='vin')
        self.assertEqual(self.ss.GENCLS.p0.v[1], 1)

    def test_as_dict(self):
        """
        Test `Group.as_dict()`.
        """
        self.assertIsInstance(self.ss.SynGen.as_dict(), dict)
