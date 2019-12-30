import unittest
import os
import andes
from andes.utils.paths import get_case


class TestMATPOWER(unittest.TestCase):

    def setUp(self) -> None:
        self.cases = ('case5.m', 'case14.m', 'case300.m', 'case9241pegase.m')

    def test_pflow_mpc(self):
        for case in self.cases:
            case_path = get_case(os.path.join('matpower', case))
            andes.main.run(case_path, no_output=True)
