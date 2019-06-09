import unittest
import andes


class TestTDSCases(unittest.TestCase):
    def setUp(self) -> None:
        andes.main.config_logger(file=False)

    @staticmethod
    def test_filter_card():
        andes.main.run('cards/AVR1.andc', exit_now=True)
