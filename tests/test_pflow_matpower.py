import unittest
import os
import andes
from andes.utils.paths import get_case

andes.main.config_logger(stream_level=30, file=True)


class TestMATPOWER(unittest.TestCase):

    def setUp(self) -> None:
        self.cases = ('case5.m', 'case14.m', 'case300.m', 'case118.m')

    def test_pflow_mpc_pool(self):
        case_path = [get_case(os.path.join('matpower', item)) for item in self.cases]
        andes.run(case_path, no_output=True, ncpu=2, pool=True, verbose=40, default_config=True)

    def test_pflow_mpc_process(self):
        case_path = [get_case(os.path.join('matpower', item)) for item in self.cases]
        andes.run(case_path, no_output=True, ncpu=2, pool=False, verbose=40, default_config=True)
