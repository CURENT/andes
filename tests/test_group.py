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
        self.assertListEqual(ss.DG.find_idx('name', ['PVD1_1', 'PVD1_2']),
                             ss.PVD1.find_idx('name', ['PVD1_1', 'PVD1_2']),
                             )

        self.assertListEqual(ss.DG.find_idx(['name', 'Sn'],
                                            [('PVD1_1', 'PVD1_2'),
                                             (1.0, 1.0)]),
                             ss.PVD1.find_idx(['name', 'Sn'],
                                              [('PVD1_1', 'PVD1_2'),
                                               (1.0, 1.0)]))

        # --- get_field ---
        ff = ss.DG.get_field('f', list(ss.DG._idx2model.keys()), 'v_code')
        self.assertTrue(any([item == 'y' for item in ff]))
