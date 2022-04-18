"""
Test the ANDES-pandapower interface.
"""

import unittest
import numpy as np

import andes
from andes.shared import deg2rad
from andes.interop.pandapower import to_pandapower, make_link_table, make_GSF

try:
    import pandapower as pp
    getattr(pp, '__version__')
    HAVE_PANDAPOWER = True
except (ImportError, AttributeError):
    HAVE_PANDAPOWER = False


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestPandapower(unittest.TestCase):
    """
    Tests for the ANDES-pandapower interface.
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

    def test_to_pandapower(self):
        """
        Test `andes.interop.pandapower.to_pandapower` with cases
        """
        for case_file in self.cases:
            case = andes.get_case(case_file)

            _test_to_pandapower_single(case, tol=1e-3)

    def test_make_link_table(self):
        """
        Test `andes.interop.pandapower.make_link_table`
        """

        sa14 = andes.load(andes.get_case('ieee14/ieee14_ieeet1.xlsx'),
                          setup=True,
                          no_output=True,
                          default_config=True)
        link_table = make_link_table(sa14)
        ridx = link_table[link_table['syg_idx'] == 'GENROU_1'].index
        c_bus = link_table['bus_name'].iloc[ridx].astype(str) == 'BUS1'
        c_exc = link_table['exc_idx'].iloc[ridx].astype(str) == 'ESST3A_2'
        c_stg = link_table['stg_idx'].iloc[ridx].astype(str) == '1'
        c_gov = link_table['gov_idx'].iloc[ridx].astype(str) == 'TGOV1_1'
        c = c_bus.values[0] and c_exc.values[0] and c_stg.values[0] and c_gov.values[0]
        return c

    def test_make_GSF(self):
        """
        Test `andes.interop.pandapower.make_GSF with ieee39`
        """

        sa39 = andes.load(andes.get_case('ieee39/ieee39.xlsx'),
                          setup=True,
                          no_output=True,
                          default_config=True)
        sp39 = to_pandapower(sa39)
        make_GSF(sp39)
        return True


def _test_to_pandapower_single(case, **kwargs):
    """
    Test `andes.interop.pandapower.to_pandapower` with a single case
    """

    sa = andes.load(case, setup=True, no_output=True, default_config=True)
    sp = to_pandapower(sa, **kwargs)

    sa.PFlow.run()
    pp.runpp(sp)

    v_andes = sa.Bus.v.v
    a_andes = sa.Bus.a.v
    v_pp = sp.res_bus['vm_pu']
    a_pp = sp.res_bus['va_degree'] * deg2rad

    # align ssa angle with slcka bus angle
    rid_slack = np.argmin(np.abs(a_pp))
    a_andes = a_andes - a_andes[rid_slack]

    return np.testing.assert_almost_equal(v_andes, v_pp, decimal=3)
