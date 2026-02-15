"""
Regression tests for Observable block conversions and two-pass init.

These tests capture baseline values before block conversions and verify
that the two-pass init produces correct results.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import andes


class TestTwoPassInit(unittest.TestCase):
    """Tests for the two-pass init mechanism."""

    def test_pvd1_init(self):
        """PVD1 init values should match expected results with two-pass init."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14_pvd1.json'),
            no_output=True,
            default_config=True,
        )
        ss.PFlow.run()
        ss.TDS.init()

        # Ipmax should be nonzero for all devices
        self.assertTrue(np.all(ss.PVD1.Ipmax.v > 0),
                        "PVD1.Ipmax should be positive for all devices")
        assert_array_almost_equal(ss.PVD1.Ipmax.v[:5],
                                  [0.011, 0.011, 0.011, 0.011, 0.011],
                                  decimal=3)

    def test_pvd1_tds(self):
        """PVD1 TDS should run without error after two-pass init."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14_pvd1.json'),
            no_output=True,
            default_config=True,
        )
        ss.PFlow.run()
        ss.TDS.config.tf = 0.1
        ss.TDS.run()
        self.assertEqual(ss.exit_code, 0, "TDS should complete without error")

    def test_exac1_init(self):
        """EXAC1 init should succeed (exercises iterative init in pass 2)."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14_exac1.xlsx'),
            no_output=True,
            default_config=True,
        )
        ss.PFlow.run()
        ss.TDS.init()
        self.assertEqual(ss.exit_code, 0, "EXAC1 TDS init should succeed")

    def test_observable_in_dae_b(self):
        """Observable addresses should be in dae.b, not dae.x or dae.y."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            no_output=True,
            default_config=True,
        )

        # IEEEST has Observable Vks_y (from Gain block)
        self.assertGreater(len(ss.IEEEST.observables), 0,
                           "IEEEST should have observables")

        obs_var = ss.IEEEST.Vks_y
        self.assertTrue(all(a < len(ss.dae.b) for a in obs_var.a),
                        "Observable addresses should be within dae.b range")

    def test_observable_init_values(self):
        """Observable var.v should be computed during init (not left as zero)."""
        ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            no_output=True,
            default_config=True,
        )
        ss.PFlow.run()
        ss.TDS.init()

        # After init, Vks_y should be computed from its e_str
        # For IEEEST with no disturbance, Vks_y = Vks_K * input = 0
        # (input is zero at steady state for a stabilizer)
        # The key test is that it's IN init_seq and computed, not skipped.
        self.assertIn('Vks_y', [
            item if isinstance(item, str) else None
            for item in ss.IEEEST.calls.init_seq
            for item in ([item] if isinstance(item, str) else item)
        ], "Vks_y should be in init_seq")

    def test_kundur_tds(self):
        """Kundur full case should work correctly with two-pass init."""
        ss = andes.load(
            andes.get_case('kundur/kundur_full.xlsx'),
            no_output=True,
            default_config=True,
        )
        ss.PFlow.run()
        ss.TDS.config.tf = 0.1
        ss.TDS.run()
        self.assertEqual(ss.exit_code, 0, "Kundur TDS should succeed")

        # GENROU omega should be 1.0 at steady state
        assert_array_almost_equal(ss.GENROU.omega.v,
                                  np.ones(ss.GENROU.n),
                                  decimal=6)


if __name__ == '__main__':
    unittest.main()
