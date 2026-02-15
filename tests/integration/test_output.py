"""
Test output selection.
"""
import os

import numpy as np
import unittest

import andes


class TestOutput(unittest.TestCase):
    """
    Class for Output test.
    """

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
        ss.TDS.load_plotter()

        nt = len(ss.dae.ts.t)
        nx = 5
        ny = 10

        self.assertEqual(len(ss.dae.x_name_output), nx)
        self.assertEqual(len(ss.dae.y_name_output), ny)

        self.assertEqual(ss.dae.ts.x.shape[1], nx)
        self.assertEqual(ss.dae.ts.y.shape[1], ny)

        np.testing.assert_array_equal(ss.Output.xidx, [1, 4, 5, 6, 7])
        np.testing.assert_array_equal(ss.Output.yidx, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # test loaded data by plot
        self.assertEqual(ss.TDS.plt._data.shape, (nt, nx + ny + 1))
        np.testing.assert_array_equal(
            ss.TDS.plt._process_yidx(ss.GENCLS.omega, a=None),
            [2, 3, 4, 5])

        np.testing.assert_array_equal(
            ss.TDS.plt._process_yidx(ss.GENCLS.delta, a=None),
            [1])

        np.testing.assert_array_equal(
            ss.TDS.plt._process_yidx(ss.TG2.pout, a=None),
            [])

        # test `DAE.ts.get_data`
        np.testing.assert_equal(
            ss.dae.ts.get_data(ss.GENCLS.omega, a=None).shape[1],
            4)

        np.testing.assert_equal(
            ss.dae.ts.get_data(ss.GENCLS.delta, a=None).shape[1],
            1)

        np.testing.assert_equal(
            ss.dae.ts.get_data(ss.TG2.pout, a=None).shape[1],
            0)

    def test_from_csv(self):
        """
        Test from_csv when loading selected output from csv file.
        """
        case = andes.get_case("5bus/pjm5bus.json")
        ss = andes.load(case,
                        no_output=True,
                        setup=False,
                        default_config=True)

        ss.add("Output", {"model": "Bus", "varname": "v"})

        ss.setup()

        ss.PFlow.run()
        ss.TDS.config.tf = 0.1
        ss.TDS.run()
        ss.TDS.load_plotter()

        ss.TDS.plt.export_csv("pjm5bus_selec_out.csv")

        # Test assign CSV in TDS.run()
        ss2 = andes.load(case,
                         no_output=True,
                         setup=False,
                         default_config=True)

        ss2.add("Output", {"model": "Bus", "varname": "v"})

        ss2.setup()

        ss2.PFlow.run()
        ss2.TDS.run(from_csv="pjm5bus_selec_out.csv")

        self.assertTrue(ss2.TDS.converged)

        # Test assign CSV in andes.load()
        ss3 = andes.load(case,
                         no_output=True,
                         setup=False,
                         default_config=True,
                         from_csv="pjm5bus_selec_out.csv")
        ss3.add("Output", {"model": "Bus", "varname": "v"})

        ss3.setup()

        ss3.PFlow.run()
        ss3.TDS.run()

        self.assertTrue(ss3.TDS.converged)

        os.remove("pjm5bus_selec_out.csv")
