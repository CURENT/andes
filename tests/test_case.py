import unittest
import andes
from andes.system import System
from andes.io import xlsx
from andes.utils.paths import get_case


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = System()
        self.ss.undill()

        # load from excel file
        xlsx.read(self.ss, get_case('5bus/pjm5bus.xlsx'))
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
        self.assertSequenceEqual(self.ss.Bus.idx, [0, 1, 2, 3, 4])
        self.assertSequenceEqual(self.ss.Area.idx, [1, 2, 3])

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
        self.ss.TDS.run([0, 20])


class TestKundur2Area(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """
    def setUp(self) -> None:
        self.ss = andes.run(get_case('kundur/kundur_full.xlsx'))

    def test_tds_run(self):
        self.ss.TDS.run([0, 20])
        andes.main.misc(clean=True)

    def test_eig_run(self):
        self.ss.EIG.run()
        andes.main.misc(clean=True)


class TestNPCCRAW(unittest.TestCase):
    """
    Test NPCC system in the RAW format
    """
    def test_npcc_raw(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'))
        andes.main.misc(clean=True)

    def test_npcc_raw_tds(self):
        self.ss = andes.run(get_case('npcc/npcc48.raw'), routine='TDS', no_output=True, profile=True)
