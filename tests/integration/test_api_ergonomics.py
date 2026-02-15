"""
Tests for API ergonomics improvements:
- TDS.get_timeseries() for all variable types
- System.add() error messages for unrecognized fields
- System.add() kwargs support
"""

import unittest

import numpy as np
import pandas as pd

import andes
from andes.utils.paths import get_case


class TestGetTimeseries(unittest.TestCase):
    """
    Test TDS.get_timeseries() on all variable types.
    Uses ieee14_esst3a which provides State, Algeb, ExtState, ExtAlgeb,
    and Observable variables in active models.
    """

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(
            get_case('ieee14/ieee14_esst3a.xlsx'),
            default_config=True,
            no_output=True,
        )
        cls.ss.PFlow.run()
        cls.ss.TDS.config.tf = 0.5
        cls.ss.TDS.run()

    def test_internal_state(self):
        """get_timeseries on State variable (GENROU.omega)."""
        df = self.ss.TDS.get_timeseries(self.ss.GENROU.omega)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], len(self.ss.dae.ts.t))
        self.assertEqual(df.shape[1], self.ss.GENROU.n)
        self.assertEqual(list(df.columns), list(self.ss.GENROU.idx.v))

        # Values should match direct DAE access
        addr = self.ss.GENROU.omega.a
        np.testing.assert_array_equal(df.values, self.ss.dae.ts.x[:, addr])

    def test_internal_algeb(self):
        """get_timeseries on Algeb variable (Bus.v)."""
        df = self.ss.TDS.get_timeseries(self.ss.Bus.v)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[1], self.ss.Bus.n)
        self.assertEqual(list(df.columns), list(self.ss.Bus.idx.v))

        addr = self.ss.Bus.v.a
        np.testing.assert_array_equal(df.values, self.ss.dae.ts.y[:, addr])

    def test_ext_algeb(self):
        """get_timeseries on ExtAlgeb variable (GENROU.v -> Bus.v)."""
        df = self.ss.TDS.get_timeseries(self.ss.GENROU.v)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[1], self.ss.GENROU.n)
        self.assertEqual(list(df.columns), list(self.ss.GENROU.idx.v))

        addr = self.ss.GENROU.v.a
        np.testing.assert_array_equal(df.values, self.ss.dae.ts.y[:, addr])

    def test_ext_state(self):
        """get_timeseries on ExtState variable (ESST3A.omega -> GENROU.omega)."""
        df = self.ss.TDS.get_timeseries(self.ss.ESST3A.omega)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[1], self.ss.ESST3A.n)

        addr = self.ss.ESST3A.omega.a
        np.testing.assert_array_equal(df.values, self.ss.dae.ts.x[:, addr])

    def test_observable(self):
        """get_timeseries on Observable variable (ESST3A.VB_y, v_code='b')."""
        df = self.ss.TDS.get_timeseries(self.ss.ESST3A.VB_y)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[1], self.ss.ESST3A.n)

        addr = self.ss.ESST3A.VB_y.a
        np.testing.assert_array_equal(df.values, self.ss.dae.ts.b[:, addr])

    def test_time_index(self):
        """DataFrame index should be the time vector."""
        df = self.ss.TDS.get_timeseries(self.ss.GENROU.omega)
        np.testing.assert_array_equal(df.index.values, self.ss.dae.ts.t)

    def test_non_variable_raises_typeerror(self):
        """get_timeseries on a parameter, service, or non-variable raises TypeError."""
        with self.assertRaises(TypeError):
            self.ss.TDS.get_timeseries(self.ss.GENROU.M)  # NumParam

        with self.assertRaises(TypeError):
            self.ss.TDS.get_timeseries('omega')  # string

        with self.assertRaises(TypeError):
            self.ss.TDS.get_timeseries(None)

    def test_no_tds_data_raises(self):
        """get_timeseries before TDS.run() should raise ValueError."""
        ss2 = andes.load(
            get_case('ieee14/ieee14_esst3a.xlsx'),
            default_config=True,
            no_output=True,
        )
        ss2.PFlow.run()

        with self.assertRaises(ValueError):
            ss2.TDS.get_timeseries(ss2.GENROU.omega)


class TestAddKwargs(unittest.TestCase):
    """Test System.add() kwargs support and error messages."""

    def test_add_with_kwargs(self):
        """add() accepts keyword arguments."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )
        idx = ss.add('Fault', bus=1, tf=1.0, tc=1.1)
        self.assertIsNotNone(idx)
        self.assertEqual(ss.Fault.n, 1)

    def test_add_with_dict(self):
        """add() accepts dictionary (existing behavior)."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )
        idx = ss.add('Fault', {'bus': 1, 'tf': 1.0, 'tc': 1.1})
        self.assertIsNotNone(idx)

    def test_add_mixed_dict_kwargs(self):
        """add() merges dict and kwargs."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )
        idx = ss.add('Fault', {'bus': 1}, tf=1.0, tc=1.1)
        self.assertIsNotNone(idx)

    def test_unrecognized_field_warns(self):
        """Unrecognized field names produce warnings."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )
        with self.assertLogs('andes.core.model.modeldata', level='WARNING') as cm:
            ss.add('Fault', {'bus': 1, 'tf': 1.0, 'tc': 1.1, 'foobar': 99})

        msgs = [m for m in cm.output if 'unrecognized' in m.lower()]
        self.assertEqual(len(msgs), 1)
        self.assertIn('foobar', msgs[0])

    def test_typo_suggests_close_match(self):
        """Typo in field name triggers 'Did you mean' suggestion."""
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )
        with self.assertLogs('andes.core.model.modeldata', level='WARNING') as cm:
            ss.add('Fault', {'bus': 1, 'tf': 1.0, 'tc': 1.1, 'busx': 5})

        msgs = [m for m in cm.output if 'Did you mean' in m]
        self.assertEqual(len(msgs), 1)
        self.assertIn('bus', msgs[0])


if __name__ == '__main__':
    unittest.main()
