"""
Tests for ModelCache: decorator registration, combined fields, and refresh.
"""

import unittest
import warnings
from collections import OrderedDict

import andes


class TestCacheFields(unittest.TestCase):
    """
    Verify cache fields on a loaded system.
    """

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(andes.get_case('ieee14/ieee14.json'),
                            no_output=True)

    def test_combined_fields(self):
        """
        Combined cache fields match manual concatenation of underlying dicts.
        """
        from andes.core.model.model import Model

        for mdl in self.ss.models.values():
            if mdl.n == 0:
                continue
            for field, attrs in Model._CACHE_COMBINED.items():
                expected = OrderedDict()
                for attr in attrs:
                    expected.update(getattr(mdl, attr, OrderedDict()))
                actual = getattr(mdl.cache, field)
                self.assertEqual(list(expected.keys()), list(actual.keys()),
                                 f"{mdl.class_name}.cache.{field}")

    def test_filter_fields_exist(self):
        """
        Filter-based cache fields (v_getters, e_adders, etc.) are registered.
        """
        mdl = self.ss.Bus
        for field in ('v_getters', 'v_adders', 'v_setters',
                      'e_adders', 'e_setters', 'iter_vars',
                      'input_vars', 'output_vars'):
            self.assertIn(field, mdl.cache._callbacks,
                          f"Missing cache field: {field}")

    def test_deprecated_fields_warn(self):
        """
        Accessing cache.df, cache.df_in, etc. emits FutureWarning
        but still returns correct data.
        """
        import pandas as pd

        mdl = self.ss.Bus
        for field in ('dict', 'df', 'dict_in', 'df_in'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = getattr(mdl.cache, field)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, FutureWarning))
                self.assertIn('v3.0.0', str(w[0].message))
            if field.startswith('df'):
                self.assertIsInstance(result, pd.DataFrame)

    def test_refresh_overwrites_stale(self):
        """
        refresh() overwrites previously cached values in __dict__.
        """
        mdl = self.ss.Bus

        # Access to populate __dict__
        _ = mdl.cache.all_vars
        self.assertIn('all_vars', mdl.cache.__dict__)

        old_id = id(mdl.cache.__dict__['all_vars'])
        mdl.cache.refresh('all_vars')
        new_id = id(mdl.cache.__dict__['all_vars'])

        # refresh() should produce a new OrderedDict object
        self.assertNotEqual(old_id, new_id)


class TestCacheTDS(unittest.TestCase):
    """
    Verify cache correctness through PFlow + TDS.
    """

    def test_pflow_tds(self):
        """
        PFlow + TDS succeed with the refactored cache.
        """
        ss = andes.load(andes.get_case('ieee14/ieee14_linetrip.xlsx'),
                        no_output=True, default_config=True)
        ss.setup()

        self.assertTrue(ss.PFlow.run(), "PFlow did not converge")

        ss.TDS.config.tf = 1.0
        self.assertTrue(ss.TDS.run(), "TDS did not complete")


if __name__ == '__main__':
    unittest.main()
