"""
Test the ANDES-gridcal interface.
"""

import unittest

import andes
from andes.interop.gridcal import to_gridcal, _verify_pf

try:
    import GridCal.Engine as gc
    getattr(gc, '__name__')
    HAVE_GRIDCAL = True
except (ImportError, AttributeError):
    HAVE_GRIDCAL = False


@unittest.skipUnless(HAVE_GRIDCAL, "gridcal not available")
class TestGridcal(unittest.TestCase):
    """
    Tests for the ANDES-gridcal interface.
    """

    cases = [
        'ieee14/ieee14_ieeet1.xlsx',
        'ieee14/ieee14_pvd1.xlsx',
        'ieee39/ieee39.xlsx',
        'npcc/npcc.xlsx',
    ]

    def setUp(self) -> None:
        """
        Test setup. This is executed before each test case.
        """

    def test_to_gridcal(self):
        """
        Test `andes.interop.gridcal.to_gridcal` with cases
        """
        for case_file in self.cases:
            case = andes.get_case(case_file)
            match = _test_to_gridcal_single(case)

            self.assertEqual(match, True)


def _test_to_gridcal_single(case, **kwargs):
    """
    Test `andes.interop.gridcal.to_gridcal` with a single case
    """

    sa = andes.load(case, setup=True, no_output=True, default_config=True)
    sgc = to_gridcal(sa, verify=False, **kwargs)
    return _verify_pf(sa, sgc)
