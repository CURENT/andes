"""
Test the ANDES-pandapower interface.
"""

import unittest
import numpy as np
import pandas as pd

import andes
from andes.shared import deg2rad
from andes.interop.pandapower import (
    to_pandapower, make_link_table, make_GSF,
    build_group_table, _rename, _prep_line_data,
    _to_pp_bus, _to_pp_line, _to_pp_trafo,
    _to_pp_load, _to_pp_shunt, _to_pp_gen,
)

try:
    import pandapower as pp
    getattr(pp, '__version__')
    HAVE_PANDAPOWER = True
except (ImportError, AttributeError):
    HAVE_PANDAPOWER = False


def _load_case(case_name):
    return andes.load(
        andes.get_case(case_name),
        setup=True, no_output=True, default_config=True,
    )


def _make_empty_pp(ssa):
    """Create an empty pandapower net from an ANDES system."""
    return pp.create_empty_network(
        f_hz=ssa.config.freq,
        sn_mva=ssa.config.mva,
    )


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

        sa14 = _load_case('ieee14/ieee14_ieeet1.xlsx')
        link_table = make_link_table(sa14)
        ridx = link_table[link_table['syg_idx'] == 'GENROU_1'].index
        c_bus = link_table['bus_name'].iloc[ridx].astype(str) == 'BUS1'
        c_exc = link_table['exc_idx'].iloc[ridx].astype(str) == 'ESST3A_2'
        c_stg = link_table['stg_idx'].iloc[ridx].astype(str) == '1'
        c_gov = link_table['gov_idx'].iloc[ridx].astype(str) == 'TGOV1_1'
        self.assertTrue(c_bus.values[0])
        self.assertTrue(c_exc.values[0])
        self.assertTrue(c_stg.values[0])
        self.assertTrue(c_gov.values[0])

    def test_make_GSF(self):
        """
        Test `andes.interop.pandapower.make_GSF with ieee39`
        """

        sa39 = _load_case('ieee39/ieee39.xlsx')
        sp39 = to_pandapower(sa39)
        gsf = make_GSF(sp39)
        self.assertIsNotNone(gsf)


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestRename(unittest.TestCase):
    """Tests for the _rename helper."""

    def test_no_duplicates(self):
        s = pd.Series(['A', 'B', 'C'])
        result = _rename(s)
        # should return input unchanged (same object)
        self.assertTrue(result is s)

    def test_with_duplicates(self):
        s = pd.Series(['A', 'B', 'A', 'B', 'C'])
        result = _rename(s)
        # first occurrences kept, duplicates get index suffix
        self.assertEqual(result.iloc[0], 'A')
        self.assertEqual(result.iloc[1], 'B')
        self.assertEqual(result.iloc[2], 'A 2')
        self.assertEqual(result.iloc[3], 'B 3')
        self.assertEqual(result.iloc[4], 'C')
        # no duplicates remain
        self.assertFalse(result.duplicated().any())

    def test_triple_duplicates(self):
        s = pd.Series(['X', 'X', 'X'])
        result = _rename(s)
        self.assertEqual(result.iloc[0], 'X')
        self.assertEqual(result.iloc[1], 'X 1')
        self.assertEqual(result.iloc[2], 'X 2')


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestBuildGroupTable(unittest.TestCase):
    """Tests for build_group_table."""

    @classmethod
    def setUpClass(cls):
        cls.sa14 = _load_case('ieee14/ieee14_ieeet1.xlsx')

    def test_all_models_in_group(self):
        df = build_group_table(self.sa14, 'StaticGen', ['idx', 'u', 'bus'])
        # ieee14 has PV + Slack generators
        n_pv = self.sa14.PV.n
        n_slack = self.sa14.Slack.n
        self.assertEqual(len(df), n_pv + n_slack)
        self.assertListEqual(list(df.columns), ['idx', 'u', 'bus'])

    def test_specific_models(self):
        df = build_group_table(
            self.sa14, 'SynGen', ['idx', 'bus', 'gen'],
            mdl_name=['GENROU'],
        )
        n_genrou = self.sa14.GENROU.n
        self.assertEqual(len(df), n_genrou)

    def test_empty_group(self):
        df = build_group_table(
            self.sa14, 'SynGen', ['idx', 'bus', 'gen'],
            mdl_name=['GENCLS'],
        )
        # ieee14_ieeet1 has no GENCLS
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ['idx', 'bus', 'gen'])


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestComponentConverters(unittest.TestCase):
    """Tests for individual _to_pp_* converter functions.

    Uses ieee14 (14 buses, 16 lines, 4 transformers, 11 loads, 2 shunts)
    and npcc (140 buses, duplicate names) to pin current behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.sa14 = _load_case('ieee14/ieee14_ieeet1.xlsx')
        cls.sa39 = _load_case('ieee39/ieee39.xlsx')
        cls.sa_npcc = _load_case('npcc/npcc.xlsx')

    # --- Bus ---

    def test_bus_count_ieee14(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        self.assertEqual(len(ssp.bus), sa.Bus.n)

    def test_bus_columns(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        for col in ['vn_kv', 'name', 'in_service', 'max_vm_pu', 'min_vm_pu', 'zone', 'type']:
            self.assertIn(col, ssp.bus.columns)

    def test_bus_vn_matches(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        np.testing.assert_array_equal(
            ssp.bus['vn_kv'].values,
            sa.Bus.as_df()['Vn'].values,
        )

    def test_bus_duplicate_names_npcc(self):
        """NPCC has duplicate bus names; verify _rename deduplicates them."""
        sa = self.sa_npcc
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        self.assertFalse(ssp.bus['name'].duplicated().any())

    # --- Line and Transformer ---

    def _prep_lines(self, sa):
        """Helper: prepare pp net with bus and line/trafo data."""
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        ssa_line = _prep_line_data(sa, ssp, ssa_bus)
        _to_pp_line(ssa_line, ssp)
        _to_pp_trafo(ssa_line, ssp)
        return ssp

    def test_line_count_ieee14(self):
        sa = self.sa14
        ssp = self._prep_lines(sa)
        line_df = sa.Line.as_df()
        n_lines = (line_df['trans'] == 0).sum()
        n_trans = (line_df['trans'] == 1).sum()
        self.assertEqual(len(ssp.line), n_lines)
        if n_trans >= 1:
            self.assertEqual(len(ssp.trafo), n_trans)

    def test_line_columns(self):
        ssp = self._prep_lines(self.sa14)
        for col in ['name', 'from_bus', 'to_bus', 'in_service',
                    'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km',
                    'length_km', 'max_i_ka']:
            self.assertIn(col, ssp.line.columns, f"Missing line column: {col}")

    def test_trafo_columns_ieee14(self):
        ssp = self._prep_lines(self.sa14)
        for col in ['name', 'hv_bus', 'lv_bus', 'sn_mva', 'vn_hv_kv',
                    'vn_lv_kv', 'vk_percent', 'vkr_percent',
                    'tap_side', 'tap_pos', 'in_service']:
            self.assertIn(col, ssp.trafo.columns, f"Missing trafo column: {col}")

    def test_line_positive_impedance(self):
        ssp = self._prep_lines(self.sa14)
        self.assertTrue((ssp.line['r_ohm_per_km'] >= 0).all())
        self.assertTrue((ssp.line['x_ohm_per_km'] > 0).all())

    def test_line_count_ieee39(self):
        """ieee39 has 12 transformers -- tests larger transformer set."""
        sa = self.sa39
        ssp = self._prep_lines(sa)
        line_df = sa.Line.as_df()
        n_lines = (line_df['trans'] == 0).sum()
        n_trans = (line_df['trans'] == 1).sum()
        self.assertEqual(len(ssp.line), n_lines)
        self.assertEqual(len(ssp.trafo), n_trans)

    # --- Load ---

    def test_load_count_ieee14(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_load(sa, ssp, ssa_bus)
        self.assertEqual(len(ssp.load), sa.PQ.n)

    def test_load_power_values(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_load(sa, ssp, ssa_bus)
        # p_mw should equal p0 * sn_mva
        pq_df = sa.PQ.as_df()
        expected_p = pq_df['p0'].values * ssp.sn_mva
        np.testing.assert_array_almost_equal(
            ssp.load['p_mw'].values, expected_p,
        )

    def test_load_columns(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_load(sa, ssp, ssa_bus)
        for col in ['name', 'bus', 'p_mw', 'q_mvar', 'in_service',
                    'sn_mva', 'scaling', 'controllable']:
            self.assertIn(col, ssp.load.columns, f"Missing load column: {col}")

    # --- Shunt ---

    def test_shunt_count_ieee14(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_shunt(sa, ssp, ssa_bus)
        self.assertEqual(len(ssp.shunt), sa.Shunt.n)

    def test_shunt_columns(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_shunt(sa, ssp, ssa_bus)
        for col in ['name', 'bus', 'p_mw', 'q_mvar', 'in_service', 'vn_kv',
                    'step', 'max_step']:
            self.assertIn(col, ssp.shunt.columns, f"Missing shunt column: {col}")

    # --- Generator ---

    def test_gen_count_ieee14(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_gen(sa, ssp)
        n_expected = sa.PV.n + sa.Slack.n
        self.assertEqual(len(ssp.gen), n_expected)

    def test_gen_has_slack(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_gen(sa, ssp)
        self.assertEqual(ssp.gen['slack'].sum(), 1)

    def test_gen_columns(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_gen(sa, ssp)
        for col in ['name', 'bus', 'p_mw', 'vm_pu', 'in_service',
                    'controllable', 'slack', 'max_p_mw', 'min_p_mw',
                    'max_q_mvar', 'min_q_mvar', 'slack_weight', 'sn_mva']:
            self.assertIn(col, ssp.gen.columns, f"Missing gen column: {col}")

    def test_gen_vm_matches_v0(self):
        sa = self.sa14
        ssp = _make_empty_pp(sa)
        ssa_bus = sa.Bus.as_df()
        ssa_bus['name'] = _rename(ssa_bus['name'])
        ssa_bus['bus'] = ssa_bus.index
        _to_pp_bus(ssp, ssa_bus)
        _to_pp_gen(sa, ssp)
        # vm_pu should match v0 from StaticGen
        stg = build_group_table(sa, 'StaticGen', ['v0'])
        np.testing.assert_array_almost_equal(
            ssp.gen['vm_pu'].values, stg['v0'].values,
        )


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestEndToEnd(unittest.TestCase):
    """End-to-end conversion and power flow verification."""

    @classmethod
    def setUpClass(cls):
        cls.sa14 = _load_case('ieee14/ieee14_ieeet1.xlsx')
        cls.sa39 = _load_case('ieee39/ieee39.xlsx')

    def test_pf_match_ieee14(self):
        sp = to_pandapower(self.sa14, verify=False)
        self.sa14.PFlow.run()
        pp.runpp(sp)
        np.testing.assert_almost_equal(
            self.sa14.Bus.v.v,
            sp.res_bus['vm_pu'].values,
            decimal=3,
        )

    def test_pf_match_ieee39(self):
        sp = to_pandapower(self.sa39, verify=False)
        self.sa39.PFlow.run()
        pp.runpp(sp)
        np.testing.assert_almost_equal(
            self.sa39.Bus.v.v,
            sp.res_bus['vm_pu'].values,
            decimal=3,
        )

    def test_no_verify_skips_pf(self):
        """verify=False should not run power flow."""
        sp = to_pandapower(self.sa14, verify=False)
        # res_bus should be empty when PF hasn't been run
        self.assertTrue(sp.res_bus.empty or sp.res_bus['vm_pu'].isna().all())

    def test_roundtrip_bus_count(self):
        sp = to_pandapower(self.sa14, verify=False)
        self.assertEqual(len(sp.bus), self.sa14.Bus.n)
        self.assertEqual(len(sp.load), self.sa14.PQ.n)
        self.assertEqual(len(sp.shunt), self.sa14.Shunt.n)
        self.assertEqual(len(sp.gen), self.sa14.PV.n + self.sa14.Slack.n)


@unittest.skipUnless(HAVE_PANDAPOWER, "pandapower not available")
class TestLinkTable(unittest.TestCase):
    """Tests for make_link_table."""

    @classmethod
    def setUpClass(cls):
        cls.sa14 = _load_case('ieee14/ieee14_ieeet1.xlsx')

    def test_link_table_columns(self):
        lt = make_link_table(self.sa14)
        expected_cols = [
            'stg_name', 'stg_u', 'stg_idx', 'bus_idx',
            'dg_idx', 'rg_idx', 'rexc_idx',
            'syg_idx', 'exc_idx', 'gov_idx',
            'bus_name', 'gammap', 'gammaq',
        ]
        self.assertListEqual(list(lt.columns), expected_cols)

    def test_link_table_row_count(self):
        lt = make_link_table(self.sa14)
        # should have one row per StaticGen
        n_stg = self.sa14.PV.n + self.sa14.Slack.n
        self.assertEqual(len(lt), n_stg)

    def test_gammap_gammaq_positive(self):
        lt = make_link_table(self.sa14)
        self.assertTrue((lt['gammap'] > 0).all())
        self.assertTrue((lt['gammaq'] > 0).all())


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
