import unittest
from andes.system import System
from andes.io import xlsx
from andes.utils.paths import get_case


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = System()
        self.ss.undill_calls()

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

    def test_pflow(self):
        self.ss.PFlow.run()
        self.ss.PFlow.newton_krylov()

    def test_tds_init(self):
        self.ss.PFlow.run()
        self.ss.TDS.run([0, 20])


class TestKundur2Area(unittest.TestCase):
    """
    Test Kundur's 2-area system
    """
    def setUp(self) -> None:
        self.ss = System()
        self.ss.undill_calls()
        xlsx.read(self.ss, get_case('kundur/kundur_full.xlsx'))
        self.ss.setup()

    def test_tds_init(self):
        self.ss.PFlow.run()
        self.ss.TDS.run([0, 20])
