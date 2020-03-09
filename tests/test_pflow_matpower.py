import unittest
import os
import andes
from andes.utils.paths import get_case

andes.main.config_logger(stream_level=30, file=False)


class TestMATPOWER(unittest.TestCase):

    def setUp(self) -> None:
        self.cases = ('case5.m', 'case14.m', 'case300.m', 'case118.m')
        # self.cases = ('*.m',)

    def test_pflow_mpc(self):
        for case in self.cases:
            case_path = get_case(os.path.join('matpower', case))
            andes.main.run(case_path, no_output=True, ncpu=2)
