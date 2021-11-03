import unittest

import andes


class TestCLI(unittest.TestCase):
    def test_main_doc(self):
        andes.main.doc('Bus')
        andes.main.doc(list_supported=True)

    def test_misc(self):
        andes.main.misc(show_license=True)
        andes.main.misc(save_config=None, overwrite=True)
