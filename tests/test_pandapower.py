"""
Test ANDES-pandapower interface.
"""

import unittest


try:
    import pandapower as pp
    HAVE_PANDAPOWER = True
except ImportError:
    HAVE_PANDAPOWER = False


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestPandapower(unittest.TestCase):
    """
    Tests for the ANDES-pandapower interface.
    """

    def setUp(self) -> None:
        """
        Test setup. This is executed before each test case.
        """
        pass

    def test_to_pandapower(self):
        pass
