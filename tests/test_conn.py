import unittest

import numpy as np

import andes


class TestConnMan(unittest.TestCase):
    """
    Test class `ConnMan`.
    """

    def setUp(self) -> None:
        self.ss = andes.load(andes.get_case("ieee14/ieee14_conn.xlsx"),
                             setup=True, default_config=True, no_output=True)

    def test_conn_init(self):
        """
        Test `ConnMan` initialization.
        """
        # normally, flag `is_needed` should be False after successful init
        self.assertFalse(self.ss.conn.is_needed)

        self.assertIsInstance(self.ss.conn.busu0, np.ndarray)
        self.assertEqual(self.ss.conn.busu0.shape, (self.ss.Bus.n,))

        self.assertIsInstance(self.ss.conn.changes, dict)
        self.assertEqual(self.ss.conn.changes['on'].shape, (self.ss.Bus.n,))
        self.assertEqual(self.ss.conn.changes['off'].shape, (self.ss.Bus.n,))

    def test_turn_off(self):
        """
        Test if connected devices are turned off.
        """
        # assert there is an offline bus
        self.assertEqual(self.ss.Bus.get(src='u', attr='v', idx=15), 0)

        # assert connected devices are turned off
        self.assertEqual(self.ss.Line.get(src='u', attr='v', idx='Line_21'), 0)
        self.assertEqual(self.ss.StaticGen.get(src='u', attr='v', idx=6), 0)
        self.assertEqual(self.ss.StaticLoad.get(src='u', attr='v', idx='PQ_12'), 0)
        self.assertEqual(self.ss.StaticShunt.get(src='u', attr='v', idx='Shunt_3'), 0)

    def test_turn_off_after_pflow(self):
        """
        Test if `ConnMan` works after solving PFlow.
        """
        ss = andes.load(andes.get_case('ieee14/ieee14_conn.xlsx'),
                        setup=False, no_output=True, default_config=True)
        ss.Bus.set(src='u', attr='v', idx=15, value=1)
        ss.setup()

        ss.PFlow.run()
        self.assertTrue(ss.PFlow.converged)

        # turn off a bus
        ss.Bus.alter(src='u', idx=15, value=0)
        # flag PFlow.converged should be reset as False by `Bus.set()`
        self.assertFalse(ss.PFlow.converged)
        self.assertTrue(ss.conn.is_needed)

    def test_turn_on_after_setup(self):
        """
        Test if raise NotImplementedError when turning on a bus after system setup.
        """
        with self.assertRaises(NotImplementedError):
            self.ss.Bus.set(src='u', attr='v', idx=15, value=1)
