"""
Test variable names in the DAE name arrays.
"""
import unittest

import andes


class TestDAENames(unittest.TestCase):
    """
    Test names in DAE.
    """

    def setUp(self):
        self.ss = andes.run(andes.get_case("kundur/kundur_full.json"),
                            default_config=True,
                            no_output=True,
                            )

    def test_dae_names(self):
        """
        Test if DAE names are non-empty.
        """

        self.ss.TDS.init()
        for item in self.ss.dae.y_name:
            self.assertNotEqual(len(item), 0)
        for item in self.ss.dae.x_name:
            self.assertNotEqual(len(item), 0)
