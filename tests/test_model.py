import unittest

import numpy as np  # NOQA
from andes.core.model import Model, ModelData  # NOQA
from andes.devices.bus import Bus  # NOQA
from andes.system import SystemNew
import dill
dill.settings['recurse'] = True


class Test5Bus(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = SystemNew()
        self.ss.prepare()

        self.ss.add("Bus", {"idx": "0", "name": "Bus 1", "Vn": "110"})
        self.ss.add("Bus", {"idx": "1", "name": "Bus 2", "Vn": "110"})
        self.ss.add("Bus", {"idx": "2", "name": "Bus 3", "Vn": "110"})
        self.ss.add("Bus", {"idx": "3", "name": "Bus 4", "Vn": "110"})
        self.ss.add("Bus", {"idx": "4", "name": "Bus 5", "Vn": "110"})

        self.ss.add("PQ", {"idx": "0", "name": "PQ 1", "bus": "1", "p0": "3", "q0": "0.9861"})
        self.ss.add("PQ", {"idx": "1", "name": "PQ 2", "bus": "2", "p0": "3", "q0": "0.9861"})
        self.ss.add("PQ", {"idx": "2", "name": "PQ 3", "bus": "3", "p0": "4", "q0": "1.3147"})

        self.ss.add("Line", {"idx": "0", "name": "Line 1-2", "bus1": "0", "bus2": "1", "r": "0.00281",
                             "x": "0.0281",
                             "b": "0.00712"})
        self.ss.add("Line", {"idx": "1", "name": "Line 1-4", "bus1": "0", "bus2": "3", "r": "0.00304",
                             "x": "0.0304",
                             "b": "0.00658"})
        self.ss.add("Line", {"idx": "2", "name": "Line 1-5", "bus1": "0", "bus2": "4", "r": "0.00064",
                             "x": "0.0064",
                             "b": "0.03126"})
        self.ss.add("Line", {"idx": "3", "name": "Line 2-3", "bus1": "1", "bus2": "2", "r": "0.00108",
                             "x": "0.0108",
                             "b": "0.01852"})
        self.ss.add("Line", {"idx": "4", "name": "Line 3-4", "bus1": "2", "bus2": "3", "r": "0.00297",
                             "x": "0.0297",
                             "b": "0.00674"})
        self.ss.add("Line", {"idx": "5", "name": "Line 4-5", "bus1": "3", "bus2": "4", "r": "0.00297",
                             "x": "0.0297",
                             "b": "0.00674"})

        self.ss.add("PV", {"idx": "0", "name": "PV 1", "bus": "0", "p0": "0.4", "v0": "1", 'qmax': "0.1"})
        self.ss.add("PV", {"idx": "1", "name": "PV 2", "bus": "0", "p0": "1.7", "v0": "1"})
        self.ss.add("PV", {"idx": "2", "name": "PV 3", "bus": "2", "p0": "3.2349", "v0": "1"})
        self.ss.add("PV", {"idx": "3", "name": "PV 5", "bus": "4", "p0": "4.6651", "v0": "1"})

        self.ss.add("Slack", {"idx": "4", "name": "Slack 1", "bus": "3", "v0": "1", "a0": "0"})

        self.ss.add("GEN2Axis", {"idx": "1", "xd": 1.7, "xq1": 0.5,
                                 "gen": "0", "bus": "0", 'Td10': 8, 'Tq10': 0.8})

        self.ss.setup()

    def test_names(self):
        self.assertEqual(self.ss.Bus.n, 5)
        self.assertEqual(self.ss.PQ.n, 3)

        self.assertTrue('Bus' in self.ss.models)
        self.assertTrue('PQ' in self.ss.models)

    def test_pflow(self):
        self.ss.PFlow.nr()
        self.ss.PFlow.newton_krylov()

    def test_tds_init(self):
        self.ss.PFlow.nr()
        self.ss.TDS._initialize()
