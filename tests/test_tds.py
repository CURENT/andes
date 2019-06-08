import unittest
import andes


class TestTDSCases(unittest.TestCase):
    def setUp(self) -> None:
        andes.main.config_logger(file=False)

    @staticmethod
    def test_ieee14_tds():
        andes.main.run('cases/ieee14/ieee14_syn.dm', routine=['pflow', 'tds'], tf=0.1)

    @staticmethod
    def test_wecc_wind0():
        andes.main.run('cases/curent/WECC_WIND0.dm', routine=['pflow', 'tds'], tf=0.1, no_output=True)

    @staticmethod
    def test_wecc_wind50():
        andes.main.run('cases/curent/WECC_WIND50.dm', routine=['pflow', 'tds'], tf=0.1, no_output=True)

    @staticmethod
    def test_npcc_raw_dyr():
        andes.main.run('cases/npcc/npcc48.raw', addfile='npcc48.dyr',
                       routine=['pflow', 'tds'], tf=0.1, no_output=True)
