"""
Test output selection.
"""

import unittest
import andes


class TestOutput(unittest.TestCase):
    """
    Class for Output test.
    """

    def setUp(self):
        pass

    def test_output_xyname(self):
        """
        Test x_name and y_name for Output
        """

        ss = andes.load(andes.get_case("5bus/pjm5bus.json"),
                        no_output=True,
                        setup=False,
                        default_config=True)

        ss.add("Output", {"model": "GENCLS", "varname": "omega"})
        ss.add("Output", {"model": "GENCLS", "varname": "delta", "dev": 2})
        ss.add("Output", {"model": "Bus"})

        ss.setup()

        ss.PFlow.run()
        ss.TDS.config.tf = 0.1
        ss.TDS.run()

        nx = 5
        ny = 10

        self.assertEqual(len(ss.dae.x_name_output), nx)
        self.assertEqual(len(ss.dae.y_name_output), ny)

        self.assertEqual(ss.dae.ts.x.shape[1], nx)
        self.assertEqual(ss.dae.ts.y.shape[1], ny)
