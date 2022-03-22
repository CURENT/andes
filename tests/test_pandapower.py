"""
Test pandapower interface.
"""

import unittest


try:
    import pandapower
    HAVE_PANDAPOWER = True
except ImportError:
    HAVE_PANDAPOWER = False


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
def test_to_pandapower():
    pass