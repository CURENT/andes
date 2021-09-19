"""
Test variable names in the DAE name arrays.
"""
import andes
import unittest


class TestDAENames(unittest.TestCase):
    """
    Test names in DAE.
    """

    def setUp(self):
        self.ss = andes.run(andes.get_case("kundur/kundur_full.xlsx"))
