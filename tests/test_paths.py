import unittest
from andes.utils.paths import list_cases
import andes
import os


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.kundur = 'kundur/'
        self.matpower = 'matpower/'

    def test_tree(self):
        list_cases(self.kundur, no_print=True)
        list_cases(self.matpower, no_print=True)

    def test_addfile_path(self):
        ieee14 = andes.get_case("ieee14/ieee14.raw")
        path, case = os.path.split(ieee14)
        andes.load('ieee14.raw', addfile='ieee14.dyr', input_path=path)

        andes.run('ieee14.raw', addfile='ieee14.dyr', input_path=path,
                  no_output=True)
