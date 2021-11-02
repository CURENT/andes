import os
import unittest

import andes
from andes.utils.paths import list_cases


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.kundur = 'kundur/'
        self.matpower = 'matpower/'
        self.ieee14 = andes.get_case("ieee14/ieee14.raw")

    def test_tree(self):
        list_cases(self.kundur, no_print=True)
        list_cases(self.matpower, no_print=True)

    def test_addfile_path(self):
        path, case = os.path.split(self.ieee14)
        ss = andes.load('ieee14.raw', addfile='ieee14.dyr',
                        input_path=path, default_config=True,
                        )
        self.assertNotEqual(ss, None)

        ss = andes.run('ieee14.raw', addfile='ieee14.dyr',
                       input_path=path,
                       no_output=True, default_config=True,
                       )
        self.assertNotEqual(ss, None)

    def test_relative_path(self):
        ss = andes.run('ieee14.raw',
                       input_path=andes.get_case('ieee14/', check=False),
                       no_output=True, default_config=True,
                       )
        self.assertNotEqual(ss, None)

    def test_pert_file(self):
        """Test path of pert file"""
        path, case = os.path.split(self.ieee14)

        # --- with pert file ---
        ss = andes.run('ieee14.raw', pert='pert.py',
                       input_path=path, no_output=True, default_config=True,
                       )
        ss.TDS.init()
        self.assertIsNotNone(ss.TDS.callpert)

        # --- without pert file ---
        ss = andes.run('ieee14.raw',
                       input_path=path, no_output=True, default_config=True,
                       )
        ss.TDS.init()
        self.assertIsNone(ss.TDS.callpert)
