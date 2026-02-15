"""
Test ANDES snapshot based on dill.
"""

import os
import unittest

import numpy as np

import andes
from andes.utils.snapshot import load_ss, save_ss


class TestSnapshot(unittest.TestCase):
    """
    Test ANDES snapshot.
    """

    def test_save_load_ss(self):
        """
        Test loading a snapshot and continuing the simulation
        """

        # --- save a snapshot ---
        ss = andes.run(andes.get_case("kundur/kundur_full.xlsx"),
                       no_output=True,
                       default_config=True,  # remove if outside tests
                       )

        ss.TDS.config.tf = 2.0
        ss.TDS.run()

        path = save_ss('kundur_full_2s.pkl', ss)

        # --- load a snapshot ---
        ss = load_ss(path)
        os.remove(path)

        # set a new simulation end time
        ss.TDS.config.tf = 3.0
        ss.TDS.run()

        np.testing.assert_almost_equal(ss.GENROU.omega.v,
                                       np.array([1.00524852, 1.00508697, 1.00423421, 1.0039603]),
                                       decimal=4)
