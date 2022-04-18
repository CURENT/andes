import unittest

import andes
import numpy as np


class TestAddressing(unittest.TestCase):
    """
    Tests for DAE addressing.
    """

    def test_ieee14_address(self):
        """
        Test IEEE14 address.
        """

        ss = andes.system.example()

        # bus variable indices (internal)
        np.testing.assert_array_equal(ss.Bus.a.a,
                                      np.arange(0, ss.Bus.n, 1))
        np.testing.assert_array_equal(ss.Bus.v.a,
                                      np.arange(ss.Bus.n, 2*ss.Bus.n, 1))
        # external variable indices
        np.testing.assert_array_equal(ss.PQ.a.a,
                                      [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
        np.testing.assert_array_equal(ss.PQ.v.a,
                                      ss.Bus.n + np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]))

        np.testing.assert_array_equal(ss.PV.q.a,
                                      np.array([28, 29, 30, 31]))

        np.testing.assert_array_equal(ss.Slack.q.a,
                                      np.array([32]))

        np.testing.assert_array_equal(ss.Slack.p.a,
                                      np.array([33]))
