"""
Test ANDES snapshot based on dill.
"""

import os
import unittest
import numpy as np

import andes
from andes.utils.snapshot import save_ss, load_ss


class TestSnapshot(unittest.TestCase):
    """
    Test ANDES snapshot.
    """

    def test_save_ss(self):
        """
        Test saving a snapshot.
        """

        ss = andes.run(andes.get_case("kundur/kundur_full.xlsx"),
                       default_config=True,  # remove if outside tests
                       )

        ss.TDS.config.tf = 2
        ss.TDS.run()

        save_ss('test_ss.pkl', ss)
        os.remove('test_ss.pkl')

    def load_ss(self):
        """
        Test loading a snapshot and continuing the simulation.
        """

        # load a snapshot
        test_dir = os.path.dirname(__file__)
        ss = load_ss(os.path.join(test_dir, 'kundur_full_2s.pkl'))

        # set a new simulation end time
        ss.TDS.config.tf = 2
        ss.TDS.run()

        np.testing.assert_almost_equal(ss.GENROU.omega.v,
                                       np.array([1.00474853, 1.00456209, 1.00316554, 1.00298933]))
