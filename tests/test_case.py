import unittest
import andes
import os
import numpy as np

from andes.utils.paths import get_case


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = andes.main.load(get_case('5bus/pjm5bus.xlsx'),
                                  default_config=True,
                                  no_output=True,
                                  )

    def test_names(self):
        self.assertTrue('Bus' in self.ss.models)
        self.assertTrue('PQ' in self.ss.models)

    def test_count(self):
        self.assertEqual(self.ss.Bus.n, 5)
        self.assertEqual(self.ss.PQ.n, 3)
        self.assertEqual(self.ss.PV.n, 3)
        self.assertEqual(self.ss.Slack.n, 1)
        self.assertEqual(self.ss.Line.n, 7)
        self.assertEqual(self.ss.GENCLS.n, 4)
        self.assertEqual(self.ss.TG2.n, 4)

    def test_idx(self):
        self.assertSequenceEqual(self.ss.Bus.idx.v, [0, 1, 2, 3, 4])
        self.assertSequenceEqual(self.ss.Area.idx.v, [1, 2, 3])

    def test_cache_refresh(self):
        self.ss.Bus.cache.refresh()

    def test_as_df(self):
        self.ss.Bus.as_df()
        self.ss.Bus.as_df(vin=True)

    def test_init_order(self):
        self.ss.Bus.get_init_order()

    def test_pflow_reset(self):
        self.ss.PFlow.run()
        if self.ss.PFlow.config.init_tds == 0:
            self.ss.reset()
            self.ss.PFlow.run()

    def test_alter_param(self):
        self.ss.PV.alter('v0', 2, 0.98)
        self.assertEqual(self.ss.PV.v0.v[1], 0.98)
        self.ss.PFlow.run()


class TestKundur2AreaEIG(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """

    def test_xlsx_eig_run(self):
        self.xlsx = get_case("kundur/kundur_full.xlsx")
        self.ss = andes.run(self.xlsx, default_config=True)

        self.ss.EIG.run()

        os.remove(self.ss.files.txt)
        os.remove(self.ss.files.eig)


class TestKundur2AreaPSSE(unittest.TestCase):
    """
    Test Kundur's 2-area system in PSS/E format
    """

    def setUp(self) -> None:
        raw = get_case("kundur/kundur.raw")
        dyr = get_case("kundur/kundur_full.dyr")
        self.ss_psse = andes.run(raw, addfile=dyr, default_config=True)

    def test_psse_tds_run(self):
        self.ss_psse.TDS.config.tf = 10
        self.ss_psse.TDS.run()
        os.remove(self.ss_psse.files.txt)
        os.remove(self.ss_psse.files.lst)
        os.remove(self.ss_psse.files.npz)

        self.assertEqual(self.ss_psse.exit_code, 0, "Exit code is not 0.")

    def test_psse_eig_run(self):
        self.ss_psse.EIG.run()
        os.remove(self.ss_psse.files.txt)
        os.remove(self.ss_psse.files.eig)

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
        self.ss = andes.run(get_case('npcc/npcc.raw'),
                            default_config=True,
                            )

        os.remove(self.ss.files.txt)

    def test_npcc_raw_tds(self):
        self.ss = andes.run(get_case('npcc/npcc.raw'),
                            verbose=50,
                            routine='tds',
                            no_output=True,
                            profile=True,
                            tf=10,
                            default_config=True,
                            )

        self.ss.dae.print_array('f')
        self.ss.dae.print_array('g')
        self.ss.dae.print_array('f', tol=1e-4)
        self.ss.dae.print_array('g', tol=1e-4)

    def test_npcc_raw_convert(self):
        self.ss = andes.run(get_case('npcc/npcc.raw'),
                            convert=True,
                            default_config=True,
                            )

        os.remove(self.ss.files.dump)
        self.assertEqual(self.ss.exit_code, 0, "Exit code is not 0.")

    def test_npcc_raw2json_convert(self):
        self.ss = andes.run(get_case('npcc/npcc.raw'),
                            convert='json',
                            default_config=True,
                            )

        self.ss2 = andes.run('npcc.json',
                             default_config=True,
                             no_output=True,
                             )

        os.remove(self.ss.files.dump)
        self.assertEqual(self.ss2.exit_code, 0, "Exit code is not 0.")

    def test_read_json_from_memory(self):
        fd = open(get_case('ieee14/ieee14_zip.json'), 'r')

        ss = andes.main.System(default_config=True,
                               no_output=True,
                               )
        ss.undill()
        andes.io.json.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_mpc_from_memory(self):
        fd = open(get_case('matpower/case14.m'), 'r')

        ss = andes.main.System(default_config=True,
                               no_output=True,
                               )
        ss.undill()
        andes.io.matpower.read(ss, fd)
        ss.setup()
        ss.PFlow.run()

        fd.close()
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_read_psse_from_memory(self):
        fd_raw = open(get_case('npcc/npcc.raw'), 'r')
        fd_dyr = open(get_case('npcc/npcc_full.dyr'), 'r')

        ss = andes.main.System(default_config=True,
                               no_output=True,
                               )
        # suppress out-of-normal info
        ss.config.warn_limits = 0
        ss.config.warn_abnormal = 0

        ss.undill()
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
        ss = andes.run(get_case('kundur/kundur_full.xlsx'),
                       routine='tds',
                       tf=2.0,
                       no_output=True,
                       default_config=True,
                       )

        ss.TDS.load_plotter()

        ss.TDS.plt.plot(ss.Bus.v, ylabel="Bus Voltages [pu]",
                        title='Bus Voltage Plot',
                        left=0.2, right=1.5,
                        ymin=0.95, ymax=1.05, legend=True, grid=True, greyscale=True,
                        hline1=1.01, hline2=1.02, vline1=0.5, vline2=0.8,
                        dpi=80, line_width=1.2, font_size=11, show=False,
                        )

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")


class TestCOI(unittest.TestCase):
    def test_kundur_COI(self):
        ss = get_case('kundur/kundur_coi.xlsx')
        exit_code = andes.run(ss,
                              routine='tds',
                              no_output=True,
                              tf=0.1,
                              cli=True,
                              default_config=True,
                              )

        self.assertEqual(exit_code, 0, "Exit code is not 0.")

    def test_kundur_COI_empty(self):
        ss = get_case('kundur/kundur_coi_empty.xlsx')

        exit_code = andes.run(ss,
                              routine='tds',
                              no_output=True,
                              tf=0.1,
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

        case = get_case('ieee14/ieee14_shuntsw.xlsx')
        ss = andes.run(case,
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
        ss = andes.run(get_case('kundur/kundur_islands.xlsx'),
                       no_output=True, default_config=True)

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")
        self.assertEqual(len(ss.Bus.islands), 2)
