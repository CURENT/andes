"""
Test the ANDES-pandapower interface.
"""

import unittest
import numpy as np

import andes
from andes.shared import deg2rad
from andes.interop.pandapower import to_pandapower, make_link_table

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

    def test_to_pandapower_ieee14(self):
        """
        Test `andes.interop.pandapower.to_pandapower` with ieee14
        """

        case14 = andes.get_case('ieee14/ieee14_ieeet1.xlsx')
        return test_to_pandapower_single(case14, tol=1e-7)

    def test_to_pandapower_ieee39(self):
        """
        Test `andes.interop.pandapower.to_pandapower` with ieee39
        """

        case39 = andes.get_case('ieee39/ieee39.xlsx')
        return test_to_pandapower_single(case39, tol=1e-6)

    def test_to_pandapower_wecc(self):
        """
        Test `andes.interop.pandapower.to_pandapower` with wecc
        """

        case_wecc = andes.get_case('wecc/wecc_full.xlsx')
        return test_to_pandapower_single(case_wecc, tol=1e-5)

    def test_make_link_table(self):
        """
        Test `andes.interop.pandapower.make_link_table`
        """

        sa14 = andes.load(andes.get_case('ieee14/ieee14_ieeet1.xlsx'),
                          setup=True,
                          no_output=True,
                          default_config=True)
        link_table = make_link_table(sa14)
        ridx = link_table[link_table['syn_idx'] == 'GENROU_1'].index
        c_bus = link_table['bus_name'].iloc[ridx].astype(str) == 'BUS1'
        c_exc = link_table['exc_idx'].iloc[ridx].astype(str) == 'ESST3A_2'
        c_stg = link_table['stg_idx'].iloc[ridx].astype(str) == '1'
        c_gov = link_table['gov_idx'].iloc[ridx].astype(str) == 'TGOV1_1'
        c = c_bus.values[0] and c_exc.values[0] and c_stg.values[0] and c_gov.values[0]
        return c


def test_to_pandapower_single(case, **kwargs):
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

    return np.testing.assert_almost_equal(v_andes, v_pp, decimal=5)
