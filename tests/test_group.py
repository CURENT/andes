"""
Tests for group functions.
"""
import andes
import numpy as np
import unittest


class TestGroup(unittest.TestCase):
    """
    Test the group class functions.
    """
    def setUp(self):
        self.ss = andes.run(andes.get_case("ieee14/ieee14_pvd1.xlsx"),
                            default_config=True)
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
