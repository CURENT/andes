from andes.core.model import Model, ModelData  # NOQA
from andes.devices.bus import Bus  # NOQA
from andes.system import SystemNew

import numpy as np  # NOQA
import unittest


class TestSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = SystemNew()
        self.n_bus = 10000
        for i in range(self.n_bus):
            self.ss.add('Bus', Vn=100, idx=i)
            self.ss.add('PQ', bus=i, idx=1)

        self.ss._set_address()
        self.ss._finalize_add()
        self.ss.link_external()
        self.ss.PQ.generate_equations()
        self.ss.PQ.generate_jacobians()

    def test_names(self):
        self.assertEqual(self.ss.Bus.n, self.n_bus)
        self.assertEqual(self.ss.PQ.n, self.n_bus)

        self.assertTrue('Bus' in self.ss.models)
        self.assertTrue('PQ' in self.ss.models)

    def test_variable_address(self):
        self.assertSequenceEqual(self.ss.Bus.a.a.tolist(), list(range(self.n_bus)))
        self.assertSequenceEqual(self.ss.Bus.v.a.tolist(), list(range(self.n_bus, 2 * self.n_bus)))

        self.assertSequenceEqual(self.ss.PQ.a.a.tolist(), list(range(self.n_bus)))
        self.assertSequenceEqual(self.ss.PQ.v.a.tolist(), list(range(self.n_bus, 2 * self.n_bus)))
