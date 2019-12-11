import unittest
import dill

import numpy as np  # NOQA
from andes.system import SystemNew
dill.settings['recurse'] = True


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = SystemNew()
        system = self.ss
        system.undill_calls()

        system.add("Bus", {"idx": "0", "name": "Bus 1", "Vn": "110", "area": 1})
        system.add("Bus", {"idx": "1", "name": "Bus 2", "Vn": "110", "area": 1})
        system.add("Bus", {"idx": "2", "name": "Bus 3", "Vn": "110", "area": 2})
        system.add("Bus", {"idx": "3", "name": "Bus 4", "Vn": "110", "area": 2})
        system.add("Bus", {"idx": "4", "name": "Bus 5", "Vn": "110", "area": 3})

        system.add("PQ", {"idx": "0", "name": "PQ 1", "bus": "1", "p0": "3", "q0": "0.9861"})
        system.add("PQ", {"idx": "1", "name": "PQ 2", "bus": "2", "p0": "3", "q0": "0.9861"})
        system.add("PQ", {"idx": "2", "name": "PQ 3", "bus": "3", "p0": "4", "q0": "1.3147"})

        system.add("Line", {"idx": "0", "name": "Line 1-2", "bus1": "0", "bus2": "1",
                            "r": "0.00281", "x": "0.0281", "b": "0.00712"})
        system.add("Line", {"idx": "1", "name": "Line 1-4", "bus1": "0", "bus2": "3",
                            "r": "0.00304", "x": "0.0304", "b": "0.00658"})
        system.add("Line", {"idx": "2", "name": "Line 1-5", "bus1": "0", "bus2": "4",
                            "r": "0.00064", "x": "0.0064", "b": "0.03126"})
        system.add("Line", {"idx": "3", "name": "Line 2-3", "bus1": "1", "bus2": "2",
                            "r": "0.00108", "x": "0.0108", "b": "0.01852"})
        system.add("Line", {"idx": "4", "name": "Line 3-4", "bus1": "2", "bus2": "3",
                            "r": "0.00297", "x": "0.0297", "b": "0.00674"})
        system.add("Line", {"idx": "5", "name": "Line 4-5", "bus1": "3", "bus2": "4",
                            "r": "0.00297", "x": "0.0297", "b": "0.00674"})
        system.add("Line", {"idx": "6", "name": "Line 1-2 (2)", "bus1": "0", "bus2": "1",
                            "r": "0.00281", "x": "0.0281", "b": "0.00712"})

        system.add("PV", {"idx": "0", "name": "PV 1", "bus": "0", "p0": "2.1", "v0": "1"})
        system.add("PV", {"idx": "2", "name": "PV 3", "bus": "2", "p0": "3.2349", "v0": "1"})
        system.add("PV", {"idx": "4", "name": "PV 5", "bus": "4", "p0": "4.6651", "v0": "1"})

        system.add("Slack", {"idx": "3", "name": "Slack 1", "bus": "3", "v0": "1", "a0": "0"})

        system.add("GENCLS", {"idx": "0", "xq": 1.7, "gen": "0", "bus": "0", "M": 4})
        system.add("GENCLS", {"idx": "2", "xq": 1.7, "gen": "2", "bus": "2", "M": 4})
        system.add("GENCLS", {"idx": "3", "xq": 1.7, "gen": "3", "bus": "3", "M": 4})
        system.add("GENCLS", {"idx": "4", "xq": 1.7, "gen": "4", "bus": "4", "M": 4})

        system.add("TG2", {'idx': '1', 'syn': "0", "pmax": 2.103})
        system.add("TG2", {'idx': '2', 'syn': "2"})
        system.add("TG2", {'idx': '3', 'syn': "3"})
        system.add("TG2", {'idx': '4', 'syn': "4"})

        system.add("Area", {"idx": 1})
        system.add("Area", {"idx": 2})
        system.add("Area", {"idx": 3})

        system.add("Toggler", {'idx': "0", 'model': 'Line', 'dev': '6', 't': 2})
        system.setup()

    def test_names(self):
        self.assertEqual(self.ss.Bus.n, 5)
        self.assertEqual(self.ss.PQ.n, 3)

        self.assertTrue('Bus' in self.ss.models)
        self.assertTrue('PQ' in self.ss.models)

    def test_idx(self):
        self.assertSequenceEqual(self.ss.Bus.idx, ['0', '1', '2', '3', '4'])
        self.assertSequenceEqual(self.ss.Area.idx, [1, 2, 3])

    def test_pflow(self):
        self.ss.PFlow.nr()
        self.ss.PFlow.newton_krylov()

    def test_tds_init(self):
        self.ss.PFlow.nr()
        self.ss.TDS.run_implicit([0, 5])
