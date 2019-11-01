from andes.models.base import Model  # NOQA
from andes.models.base import ModelData  # NOQA
from andes.models.bus import BusNew  # NOQA
from andes.system import SystemNew

import unittest


class TestSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = SystemNew()
        self.ss.add('BusNew', Vn=100, idx=1)
        self.ss.add('BusNew', Vn=100, idx=2)

        self.ss.add('PQNew', bus=1, idx=1)
        self.ss.add('PQNew', bus=2, idx=2)

        self.ss.set_address()
        self.ss.link_external()

    def test_names(self):
        self.assertTrue('BusNew' in self.ss.models)
        self.assertTrue('PQNew' in self.ss.models)

        self.assertSequenceEqual(self.ss.BusNew.a.a.tolist(), [0, 1])
        self.assertSequenceEqual(self.ss.BusNew.v.a.tolist(), [2, 3])
