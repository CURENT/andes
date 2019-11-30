import unittest
import andes


class TestAndesHelp(unittest.TestCase):
    def test_andeshelp_model_format(self):
        self.assertEqual(andes.main.andeshelp(model_format='all'), True)

        self.assertEqual(andes.main.andeshelp(model_format='Bus'), True)

    def test_andeshelp_model_var(self):
        self.assertEqual(andes.main.andeshelp(model_var='Bus.voltage'), True)

    def test_andeshelp_quick_help(self):
        self.assertEqual(andes.main.andeshelp(quick_help='Bus'), True)

    def test_andeshelp_help_config(self):
        self.assertEqual(andes.main.andeshelp(help_config='all'), True)

    def test_andeshelp_data_example(self):
        self.assertEqual(andes.main.andeshelp(data_example='Bus'), True)

    def test_andeshelp_group(self):
        self.assertEqual(andes.main.andeshelp(group='Synchronous'), True)
