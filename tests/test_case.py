import unittest
import andes
import os
from andes.utils.paths import get_case


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = andes.System()
        self.ss.undill()

        # load from excel file
        andes.io.xlsx.read(self.ss, get_case('5bus/pjm5bus.xlsx'))
        self.ss.setup()

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

    def test_cache_refresn(self):
        self.ss.Bus.cache.refresh()

    def test_as_df(self):
        self.ss.Bus.as_df()
        self.ss.Bus.as_df_in()

    def test_init_order(self):
        self.ss.Bus.get_init_order()

    def test_pflow(self):
        self.ss.PFlow.run()
        self.ss.PFlow.newton_krylov()

    def test_pflow_reset(self):
        self.ss.PFlow.run()
        self.ss.reset()
        self.ss.PFlow.run()

    def test_tds_init(self):
        self.ss.PFlow.run()
        self.ss.TDS.config.tf = 10
        self.ss.TDS.run()

    def test_alter_param(self):
        self.ss.PV.alter('v0', 2, 0.98)
        self.assertEqual(self.ss.PV.v0.v[1], 0.98)
        self.ss.PFlow.run()


class TestKundur2AreaXLSX(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """
    def setUp(self) -> None:
        xlsx = get_case("kundur/kundur_full.xlsx")
        self.ss = andes.run(xlsx)

    def test_xlsx_tds_run(self):
        self.ss.TDS.config.tf = 10
        self.ss.TDS.run()
        andes.main.misc(clean=True)

    def test_xlsx_eig_run(self):
        self.ss.EIG.run()
        andes.main.misc(clean=True)


class TestKundur2AreaPSSE(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """
    def setUp(self) -> None:
        raw = get_case("kundur/kundur_full.raw")
        dyr = get_case("kundur/kundur_full.dyr")
        self.ss_psse = andes.run(raw, addfile=dyr)

    def test_psse_tds_run(self):
        self.ss_psse.TDS.config.tf = 10
        self.ss_psse.TDS.run()
        andes.main.misc(clean=True)

    def test_psse_eig_run(self):
        self.ss_psse.EIG.run()
        andes.main.misc(clean=True)

    def test_kundur_psse2xlsx(self):
        output_name = 'test_kundur_convert.xlsx'
        andes.io.dump(self.ss_psse, 'xlsx', full_path=output_name)
        os.remove(output_name)


class TestNPCCRAW(unittest.TestCase):
    """
    Test NPCC system in the RAW format.
    """
    def test_npcc_raw(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'))
        andes.main.misc(clean=True)

    def test_npcc_raw_tds(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'),
                            verbose=50,
                            routine='tds',
                            no_output=True,
                            profile=True,
                            tf=10,
                            )
        self.ss.dae.print_array('f')
        self.ss.dae.print_array('g')
        self.ss.dae.print_array('f', tol=1e-4)
        self.ss.dae.print_array('g', tol=1e-4)

    def test_npcc_raw_convert(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'), convert=True)
        os.remove(self.ss.files.dump)

    def test_npcc_raw2json_convert(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'),
                            convert='json')
        self.ss2 = andes.run('npcc48.json')
        os.remove(self.ss.files.dump)
        andes.main.misc(clean=True)
