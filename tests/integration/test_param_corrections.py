"""
Tests for deferred parameter correction reporting.
"""

import unittest

import andes
from andes.utils.paths import get_case


class TestParamCorrections(unittest.TestCase):
    """
    Test that parameter violations are collected and reported
    in batch during setup, rather than one-by-one during add.
    """

    def test_non_zero_correction_grouped(self):
        """
        Test that non_zero violations are grouped into a single warning per (param, violation).
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        # Add two Shunt devices with Vn=0 (violates non_zero).
        ss.add('Shunt', {'bus': 1, 'Vn': 0.0, 'g': 0.0, 'b': 0.05})
        ss.add('Shunt', {'bus': 2, 'Vn': 0.0, 'g': 0.0, 'b': 0.05})

        # Corrections should be accumulated but not yet logged.
        self.assertTrue(len(ss.Shunt._param_corrections) > 0)

        with self.assertLogs('andes.core.model.modeldata', level='WARNING') as cm:
            ss.setup()

        # Should have exactly one warning line for (Vn, non_zero),
        # listing both device indices.
        vn_warnings = [msg for msg in cm.output if 'non_zero' in msg and '<Vn>' in msg]
        self.assertEqual(len(vn_warnings), 1, f"Expected 1 grouped Vn warning, got: {vn_warnings}")
        self.assertIn('2 device(s)', vn_warnings[0])

    def test_no_correction_no_warning(self):
        """
        Test that no warnings are emitted when all parameters are valid.
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        # Add a Shunt with valid parameters (non-zero Vn and Sn).
        ss.add('Shunt', {'bus': 1, 'Vn': 110, 'Sn': 100, 'g': 0.0, 'b': 0.05})

        # No corrections should have been accumulated for this device.
        shunt_corrections = ss.Shunt._param_corrections
        self.assertEqual(len(shunt_corrections), 0)

    def test_correction_clears_after_report(self):
        """
        Test that _param_corrections is cleared after report_corrections is called.
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        ss.add('Shunt', {'bus': 1, 'Vn': 0.0, 'g': 0.0, 'b': 0.05})

        self.assertTrue(len(ss.Shunt._param_corrections) > 0)

        ss.setup()

        # After setup (which calls report_corrections), the dict should be cleared.
        self.assertEqual(len(ss.Shunt._param_corrections), 0)

    def test_multiple_violation_types(self):
        """
        Test that different violation types on the same model produce separate grouped warnings.
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        # Vn=0 and Sn=0 both trigger non_zero.
        ss.add('Shunt', {'bus': 1, 'Vn': 0.0, 'Sn': 0.0, 'g': 0.0, 'b': 0.05})

        corrections = ss.Shunt._param_corrections
        # Should have entries for both Vn and Sn non_zero violations.
        self.assertIn(('Vn', 'non_zero'), corrections)
        self.assertIn(('Sn', 'non_zero'), corrections)

    def test_idx_recorded_in_corrections(self):
        """
        Test that the correct device idx values are recorded in corrections.
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        idx1 = ss.add('Shunt', {'bus': 1, 'Vn': 0.0, 'g': 0.0, 'b': 0.05})
        idx2 = ss.add('Shunt', {'bus': 2, 'Vn': 0.0, 'g': 0.0, 'b': 0.05})

        vn_idxes = ss.Shunt._param_corrections[('Vn', 'non_zero')]
        self.assertIn(idx1, vn_idxes)
        self.assertIn(idx2, vn_idxes)

    def test_non_negative_correction_on_busfreq(self):
        """
        Test that a BusFreq added with a negative time constant (Tf)
        gets its correction reported during setup.

        This exercises the reporting path for devices added late
        (e.g. by DeviceFinder during find_devices), since
        _report_param_corrections runs after find_devices.
        """
        ss = andes.load(
            get_case('ieee14/ieee14.json'),
            setup=False,
            default_config=True,
            no_output=True,
        )

        ss.add('BusFreq', {'bus': 1, 'Tf': -0.05})

        self.assertIn(('Tf', 'non_negative'), ss.BusFreq._param_corrections)

        with self.assertLogs('andes.core.model.modeldata', level='WARNING') as cm:
            ss.setup()

        tf_warnings = [m for m in cm.output if 'non_negative' in m and '<Tf>' in m]
        self.assertEqual(len(tf_warnings), 1)
        self.assertIn('1 device(s)', tf_warnings[0])
