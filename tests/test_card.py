import unittest
import andes


class TestTDSCases(unittest.TestCase):
    def setUp(self) -> None:
        andes.main.config_logger(file=False)

    @staticmethod
    def test_filter_card():
        try:
            andes.main.run('cards/AVR1.andc', no_output=True, exit_now=True)
        except ImportError:
            pass
