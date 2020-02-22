import unittest
from andes.utils.paths import list_cases


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.kundur = 'kundur/'
        self.matpower = 'matpower/'

    def test_tree(self):
        list_cases(self.kundur, no_print=True)
        list_cases(self.matpower, no_print=True)
