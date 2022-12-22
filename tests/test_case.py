import os
import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestSMIB(unittest.TestCase):
    """
    Tests for SMIB.
    """

    def test_pflow(self):
        """
        Test power flow for SMIB.
        """
        ss = andes.run(
            andes.get_case("smib/SMIB.json"),
            default_config=True,
            no_output=True,
        )
        np.testing.assert_array_almost_equal(ss.Bus.v.v, [1.05, 1, 1.01699103])
        np.testing.assert_array_almost_equal(
            ss.Bus.a.v, [3.04692663e-01, 1.43878247e-22, 1.77930079e-01])


class Test5Bus(unittest.TestCase):
    """
    Tests for the 5-bus system.
    """

    def setUp(self) -> None:
        self.ss = andes.main.load(
            get_case('5bus/pjm5bus.json'),
            default_config=True,
            no_output=True,
        )

    def test_essential(self):
        """
        Test essential functionalities of Model and System.
        """

        # --- test model names
        self.assertTrue('Bus' in self.ss.models)
        self.assertTrue('PQ' in self.ss.models)

        # --- test device counts
        self.assertEqual(self.ss.Bus.n, 5)
        self.assertEqual(self.ss.PQ.n, 3)
        self.assertEqual(self.ss.PV.n, 3)
        self.assertEqual(self.ss.Slack.n, 1)
        self.assertEqual(self.ss.Line.n, 7)
        self.assertEqual(self.ss.GENCLS.n, 4)
        self.assertEqual(self.ss.TG2.n, 4)

        # test idx values
        self.assertSequenceEqual(self.ss.Bus.idx.v, [0, 1, 2, 3, 4])
        self.assertSequenceEqual(self.ss.Area.idx.v, [1, 2, 3])

        # test cache refreshing
        self.ss.Bus.cache.refresh()

        # test conversion to dataframe
        self.ss.Bus.as_df()
        self.ss.Bus.as_df(vin=True)

        # test model initialization sequence
        self.ss.Bus.get_init_order()

    def test_pflow_reset(self):
        """
        Test resetting power flow.
        """

        self.ss.PFlow.run()
        if self.ss.PFlow.config.init_tds == 0:
            self.ss.reset()
            self.ss.PFlow.run()

    def test_alter_param(self):
        """
        Test altering parameter for power flow.
        """

        self.ss.PV.alter('v0', 2, 0.98)
        self.assertEqual(self.ss.PV.v0.v[1], 0.98)
        self.ss.PFlow.run()


class TestKundur2AreaEIG(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """

    def test_xlsx_eig_run(self):
        """
        Test eigenvalue run for Kundur using xlsx data.
        """
        self.xlsx = get_case("kundur/kundur_full.xlsx")
        ss = andes.run(
            self.xlsx,
            default_config=True,
            no_output=True,
        )

        ss.EIG.run()


class TestKundur2AreaPSSE(unittest.TestCase):
    """
    Test Kundur's 2-area system in PSS/E format
    """

    def setUp(self) -> None:
        raw = get_case("kundur/kundur.raw")
        dyr = get_case("kundur/kundur_full.dyr")
        self.ss_psse = andes.run(
            raw,
            addfile=dyr,
            default_config=True,
            no_output=True,
        )

    def test_psse_tds_run_with_stats(self):
        self.ss_psse.config.save_stats = 1
        self.ss_psse.TDS.config.tf = 3
        self.ss_psse.TDS.run()

        self.assertEqual(self.ss_psse.exit_code, 0, "Exit code is not 0.")

    def test_psse_eig_run(self):
        self.ss_psse.EIG.run()

        self.assertEqual(self.ss_psse.exit_code, 0, "Exit code is not 0.")

    def test_kundur_psse2xlsx(self):
        output_name = 'test_kundur_convert.xlsx'
        andes.io.dump(self.ss_psse, 'xlsx', full_path=output_name)
        os.remove(output_name)


class TestNPCCRAW(unittest.TestCase):
    """
    Test NPCC system in the RAW format.
    """

    def test_npcc_raw(self):
        andes.run(
            get_case('npcc/npcc.raw'),
            default_config=True,
            no_output=True,
        )

    def test_npcc_raw_tds(self):
        ss = andes.run(
            get_case('npcc/npcc.raw'),
            verbose=50,
            routine='tds',
            no_output=True,
            profile=True,
            tf=10,
            default_config=True,
        )

        ss.dae.print_array('f')
        ss.dae.print_array('g')
        ss.dae.print_array('f', tol=1e-4)
        ss.dae.print_array('g', tol=1e-4)

    def test_npcc_raw_convert(self):
        ss = andes.run(
            get_case('npcc/npcc.raw'),
            convert=True,
            default_config=True,
        )

        os.remove(ss.files.dump)
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_npcc_raw2json_convert(self):
        ss = andes.run(
            get_case('npcc/npcc.raw'),
            convert='json',
            default_config=True,
        )

        ss2 = andes.run(
            'npcc.json',
            default_config=True,
            no_output=True,
        )

        os.remove(ss.files.dump)
        self.assertEqual(ss2.exit_code, 0, "Exit code is not 0.")

    def test_read_json_from_memory(self):
        fd = open(get_case('ieee14/ieee14_zip.json'), 'r')

        ss = andes.main.System(
            default_config=True,
            no_output=True,
        )
        andes.io.json.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_mpc_from_memory(self):
        fd = open(get_case('matpower/case14.m'), 'r')

        ss = andes.main.System(
            default_config=True,
            no_output=True,
        )
        andes.io.matpower.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_psse_from_memory(self):
        fd_raw = open(get_case('npcc/npcc.raw'), 'r')
        fd_dyr = open(get_case('npcc/npcc_full.dyr'), 'r')

        ss = andes.main.System(
            default_config=True,
            no_output=True,
        )
        # suppress out-of-normal info
        ss.config.warn_limits = 0
        ss.config.warn_abnormal = 0

        andes.io.psse.read(ss, fd_raw)
        andes.io.psse.read_add(ss, fd_dyr)
        ss.setup()
        ss.PFlow.run()
        ss.TDS.init()

        fd_raw.close()
        fd_dyr.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")


class TestPlot(unittest.TestCase):

    def test_kundur_plot(self):
        import matplotlib
        matplotlib.use('Agg')  # use headless plot for testing

        ss = andes.run(
            get_case('kundur/kundur_full.json'),
            routine='tds',
            tf=2.0,
            no_output=True,
            default_config=True,
        )

        ss.TDS.load_plotter()

        ss.TDS.plt.plot(
            ss.Bus.v,
            ylabel="Bus Voltages [pu]",
            title='Bus Voltage Plot',
            left=0.2,
            right=1.5,
            ymin=0.95,
            ymax=1.05,
            legend=True,
            grid=True,
            greyscale=True,
            hline=[1.01, 1.02],
            vline=[0.5, 0.8],
            dpi=80,
            line_width=1.2,
            font_size=11,
            show=False,
        )

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")


class TestCOI(unittest.TestCase):

    def test_kundur_COI(self):
        ss = get_case('kundur/kundur_coi.json')
        exit_code = andes.run(
            ss,
            routine='tds',
            no_output=True,
            tf=0.1,
            cli=True,
            default_config=True,
        )

        self.assertEqual(exit_code, 0, "Exit code is not 0.")

    def test_kundur_COI_empty(self):
        ss = get_case('kundur/kundur_coi_empty.json')

        exit_code = andes.run(
            ss,
            routine='tds',
            no_output=True,
            tf=0.1,
            cli=True,
            default_config=True,
        )

        self.assertEqual(exit_code, 0, "Exit code is not 0.")


class TestVSC(unittest.TestCase):
    """Test case =for VSC power flow model"""

    def test_kundur_vsc(self):
        """Test power flow exit code"""

        ss = get_case('kundur/kundur_vsc.json')
        exit_code = andes.run(
            ss,
            routine='pflow',
            no_output=True,
            cli=True,
            default_config=True,
        )

        self.assertEqual(exit_code, 0, "Exit code is not 0.")


class TestShuntSw(unittest.TestCase):
    """Test class for switched shunt."""

    def test_shuntsw(self):
        """
        Test `ShuntSw` class.
        """

        case = get_case('ieee14/ieee14_shuntsw.json')
        ss = andes.run(
            case,
            no_output=True,
            default_config=True,
        )

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

        np.testing.assert_almost_equal(ss.ShuntSw.beff.v, [0.1, 0.1])
        np.testing.assert_almost_equal(ss.ShuntSw.beff.bcs[0],
                                       [0., 0.025, 0.05, 0.075, 0.1, 0.125])
        np.testing.assert_almost_equal(ss.ShuntSw.beff.bcs[1],
                                       [0., 0.05, 0.1, 0.15, 0.2, 0.25])


class TestIslands(unittest.TestCase):
    """Test power flow with two islands"""

    def test_islands(self):
        ss = andes.run(get_case('kundur/kundur_islands.json'),
                       no_output=True,
                       default_config=True)

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(len(ss.Bus.islands), 2)


class TestCaseInit(unittest.TestCase):
    """
    Test if initializations pass.
    """

    def test_pvd1_init(self):
        """
        Test if PVD1 model initialization works.
        """
        ss = andes.run(
            get_case('ieee14/ieee14_pvd1.json'),
            no_output=True,
            default_config=True,
        )
        ss.config.warn_limits = 0
        ss.config.warn_abnormal = 0

        ss.TDS.init()

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_exac1_init(self):
        """
        Test EXAC1 initialization with one TGOV1 at lower limit.
        """
        ss = andes.load(
            get_case('ieee14/ieee14_exac1.json'),
            no_output=True,
            default_config=True,
        )
        ss.PV.config.pv2pq = 1
        ss.PFlow.run()

        # suppress EXAC1 warning from select
        np.seterr(invalid='ignore')

        ss.config.warn_limits = 0
        ss.config.warn_abnormal = 0

        ss.TDS.init()

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")
