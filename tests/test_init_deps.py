"""
Tests for discrete flag dependency tracking in init sequence.

Tests that the dependency resolver correctly discovers dependencies
introduced by discrete component flags in v_str expressions.
"""

import unittest
from collections import OrderedDict

import sympy as sp
from andes.core.discrete import LessThan, Limiter
from andes.core.param import NumParam
from andes.core.var import Algeb, BaseVar
from andes.core.symprocessor import _store_deps, resolve_deps


class TestDiscreteDepTracking(unittest.TestCase):
    """Unit tests for discrete flag dependency tracking in _store_deps."""

    def test_flag_in_v_str_adds_var_dep(self):
        """When v_str contains `lim_zi` and the limiter's input is a BaseVar,
        the input var should appear as a dependency."""
        x_var = Algeb(info='input var')
        x_var.name = 'x'

        lower = NumParam()
        upper = NumParam()
        lim = Limiter(x_var, lower, upper, name='lim')

        # Build flag_deps from the discrete component
        flag_deps = {}
        for flag_name in lim.get_names():  # ['lim_zi', 'lim_zl', 'lim_zu']
            dep_var_names = []
            for inp in lim.input_list:
                if isinstance(inp, BaseVar):
                    dep_var_names.append(inp.name)
            flag_deps[flag_name] = dep_var_names

        # Expression: y = lim_zi * some_param
        x_sym = sp.Symbol('x', real=True)
        lim_zi = sp.Symbol('lim_zi', real=True)
        K = sp.Symbol('K', real=True)

        vars_int_dict = {'x': x_sym}
        expr = lim_zi * K  # y depends on lim_zi but NOT directly on x

        deps = OrderedDict()
        _store_deps('y', expr, vars_int_dict, deps, flag_deps=flag_deps)

        self.assertIn('x', deps['y'],
                      "Discrete flag lim_zi should introduce dependency on limiter input var 'x'")

    def test_param_input_not_in_deps(self):
        """When a discrete component's input is a NumParam (not BaseVar),
        no variable dependency should be added."""
        param_input = NumParam()
        param_input.name = 'MODE'

        bound = NumParam()
        lt = LessThan(param_input, bound, name='SW')

        flag_deps = {}
        for flag_name in lt.get_names():
            dep_var_names = []
            for inp in lt.input_list:
                if isinstance(inp, BaseVar) and hasattr(inp, 'name'):
                    dep_var_names.append(inp.name)
            flag_deps[flag_name] = dep_var_names

        # Expression: y = SW_z0 * something
        SW_z0 = sp.Symbol('SW_z0', real=True)
        K = sp.Symbol('K', real=True)

        vars_int_dict = {}  # no internal vars
        expr = SW_z0 * K

        deps = OrderedDict()
        _store_deps('y', expr, vars_int_dict, deps, flag_deps=flag_deps)

        self.assertEqual(deps['y'], [],
                         "NumParam input to discrete should NOT add variable dependencies")

    def test_multiple_flags_union_deps(self):
        """When v_str references flags from two different discrete components,
        dependencies from both should be included."""
        a_var = Algeb(info='var a')
        a_var.name = 'a'
        b_var = Algeb(info='var b')
        b_var.name = 'b'

        lower = NumParam()
        upper = NumParam()

        lim1 = Limiter(a_var, lower, upper, name='lim1')
        lim2 = Limiter(b_var, lower, upper, name='lim2')

        flag_deps = {}
        for lim in [lim1, lim2]:
            for flag_name in lim.get_names():
                dep_var_names = []
                for inp in lim.input_list:
                    if isinstance(inp, BaseVar):
                        dep_var_names.append(inp.name)
                flag_deps[flag_name] = dep_var_names

        # Expression: z = lim1_zi * a + lim2_zu * b
        a_sym = sp.Symbol('a', real=True)
        b_sym = sp.Symbol('b', real=True)
        lim1_zi = sp.Symbol('lim1_zi', real=True)
        lim2_zu = sp.Symbol('lim2_zu', real=True)

        vars_int_dict = {'a': a_sym, 'b': b_sym}
        expr = lim1_zi * a_sym + lim2_zu * b_sym

        deps = OrderedDict()
        _store_deps('z', expr, vars_int_dict, deps, flag_deps=flag_deps)

        # 'a' appears directly AND via lim1_zi; 'b' appears directly AND via lim2_zu
        self.assertIn('a', deps['z'])
        self.assertIn('b', deps['z'])

    def test_flag_dep_no_duplicate(self):
        """If a variable is already a direct dependency AND a flag dependency,
        it should appear only once."""
        x_var = Algeb(info='input var')
        x_var.name = 'x'

        lower = NumParam()
        upper = NumParam()
        lim = Limiter(x_var, lower, upper, name='lim')

        flag_deps = {}
        for flag_name in lim.get_names():
            flag_deps[flag_name] = ['x']

        # Expression: y = x + lim_zi (both direct dep on x and flag dep on x)
        x_sym = sp.Symbol('x', real=True)
        lim_zi = sp.Symbol('lim_zi', real=True)

        vars_int_dict = {'x': x_sym}
        expr = x_sym + lim_zi

        deps = OrderedDict()
        _store_deps('y', expr, vars_int_dict, deps, flag_deps=flag_deps)

        # 'x' should appear exactly once
        self.assertEqual(deps['y'].count('x'), 1,
                         "Variable should not appear as duplicate dependency")

    def test_resolve_ordering_with_flag_deps(self):
        """resolve_deps should produce correct order when discrete flag deps
        are included in the dependency graph."""
        # Scenario: x has no deps, y depends on x (via discrete flag)
        deps = OrderedDict()
        deps['x'] = []
        deps['y'] = ['x']  # y depends on x through a discrete flag

        seq = resolve_deps(deps)

        # x must come before y
        x_idx = seq.index('x')
        y_idx = seq.index('y')
        self.assertLess(x_idx, y_idx,
                        "x should be initialized before y in the sequence")

    def test_backward_compat_no_flag_deps(self):
        """When flag_deps is None (default), _store_deps should behave
        exactly as before — only direct variable dependencies."""
        x_sym = sp.Symbol('x', real=True)
        lim_zi = sp.Symbol('lim_zi', real=True)

        vars_int_dict = {'x': x_sym}
        expr = x_sym + lim_zi

        deps = OrderedDict()
        _store_deps('y', expr, vars_int_dict, deps)  # no flag_deps

        # Only direct dep on x, NOT lim_zi (which is not in vars_int_dict)
        self.assertEqual(deps['y'], ['x'])


class TestObservableInInitSeq(unittest.TestCase):
    """Integration tests for Observable appearing in init_seq."""

    def test_ieeest_init_seq_includes_observable(self):
        """IEEEST model should include Vks_y (Observable from Gain block)
        in its init_seq after the dependency resolver runs."""
        import andes

        ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            no_output=True,
            default_config=True,
        )

        # Check that IEEEST has observables
        self.assertGreater(len(ss.IEEEST.observables), 0,
                           "IEEEST should have at least one Observable (Vks_y)")

        # Check that the Observable name appears in init_seq
        init_seq_flat = []
        for item in ss.IEEEST.calls.init_seq:
            if isinstance(item, str):
                init_seq_flat.append(item)
            elif isinstance(item, list):
                init_seq_flat.extend(item)

        self.assertIn('Vks_y', init_seq_flat,
                      "Observable Vks_y should appear in init_seq")


if __name__ == '__main__':
    unittest.main()
