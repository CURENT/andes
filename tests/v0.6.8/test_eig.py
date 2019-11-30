import unittest
import andes


class TestEIGCases(unittest.TestCase):
    def setUp(self) -> None:
        andes.main.config_logger(file=False)

    @staticmethod
    def test_ieee14_tds():
        andes.main.run('cases/ieee14/ieee14_syn.dm', routine=['pflow', 'eig'], tf=0.1, no_output=True)
