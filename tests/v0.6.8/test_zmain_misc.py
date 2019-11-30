import unittest
import andes


class TestMainMisc(unittest.TestCase):

    def setUp(self) -> None:
        self.parser = andes.main.cli_parser()

    def test_clean(self):
        parsed = self.parser.parse_args(['--clean'])
        andes.main.main(parsed)
