"""
Test the ANDES-gridcal interface.
"""

import unittest
import numpy as np

import andes
from andes.interop.gridcal import to_gridcal


try:
    import GridCal as gc
    getattr(gc, '__name__')
    HAVE_GRIDCAL = True
except (ImportError, AttributeError):
    HAVE_GRIDCAL = False


@unittest.skipUnless(HAVE_GRIDCAL, "gridcal not available")
class TestGridcal(unittest.TestCase):
    """
    Tests for the ANDES-gridcal interface.
    """

    cases = ['ieee14/ieee14_ieeet1.xlsx',
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

            _test_to_gridcal_single(case, tol=1e-3)


def _test_to_gridcal_single(case, **kwargs):
    """
    Test `andes.interop.gridcal.to_gridcal` with a single case
    """

    sa = andes.load(case, setup=True, no_output=True, default_config=True)
    sp = to_gridcal(sa, **kwargs)

    sa.PFlow.run()
    
    # GridCal invoke power flow call
    options = gc.Engine.PowerFlowOptions(gc.Engine.SolverType.NR, verbose=False)
    pf = gc.Engine.PowerFlowDriver(sp, options)
    pf.run()
    # 

    v_andes = sa.Bus.v.v
    a_andes = sa.Bus.a.v
    v_gc = np.abs(pf.results.voltage)
    a_gc = np.angle(pf.results.voltage) * 180 / np.pi

    # return np.testing.assert_almost_equal(v_andes, v_gc, decimal=3)


if __name__ == "__main__":

    cases = ['ieee14/ieee14_ieeet1.xlsx',
             'ieee14/ieee14_pvd1.xlsx',
             'ieee39/ieee39.xlsx',
             'npcc/npcc.xlsx',
             ]

    for case_file in cases:
        case = andes.get_case(case_file)

        _test_to_gridcal_single(case, tol=1e-3)


