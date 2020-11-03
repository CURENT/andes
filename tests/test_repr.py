"""
Test __repr__ of modeling elements.
"""

import unittest
import andes
import contextlib


class TestRepr(unittest.TestCase):
    """Test __repr__"""
    def setUp(self):
        self.ss = andes.run(andes.get_case("ieee14/ieee14_linetrip.xlsx"),
                            no_output=True,
                            default_config=True,
                            )

    def test_print_repr(self):
        """
        Print out ``cache``'s fields and values.
        """
        with contextlib.redirect_stdout(None):
            for model in self.ss.models.values():
                print(model.cache.__dict__)
