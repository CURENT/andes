"""
Test PLBVFU1 model.
"""
import unittest
import andes


class TestPLBVFU1(unittest.TestCase):
    """
    Class for testing PLBVFU1.
    """

    def test_PLBVFU1(self):
        """
        Test PLVBFU1 model.
        """

        ss = andes.run(andes.get_case("ieee14/ieee14_plbvfu1.xlsx"),
                       default_config=True,
                       no_output=True,
                       )
        ss.TDS.config.tf = 3.0
        ss.TDS.config.criteria = 0
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0)
