import unittest
import andes


class TestTDSCases(unittest.TestCase):
    def setUp(self) -> None:
        andes.main.config_logger(file=False)

    def test_ieee14_tds(self):
        andes.main.run('cases/ieee14/ieee14_syn.dm', routine=['pflow', 'tds'], tf=0.1)

    def test_wecc_wind0(self):
        andes.main.run('cases/curent/WECC_WIND0.dm', routine=['pflow', 'tds'], tf=0.1, no_output=True)

    def test_wecc_wind50(self):
        andes.main.run('cases/curent/WECC_WIND50.dm', routine=['pflow', 'tds'], tf=0.1, no_output=True)

    def test_npcc_raw_dyr(self):
        andes.main.run('cases/npcc/npcc48.raw', addfile='npcc48.dyr',
                       routine=['pflow', 'tds'], tf=0.1, no_output=True)
