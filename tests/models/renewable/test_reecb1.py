import itertools
import json
import os
import tempfile
import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestREECB1(unittest.TestCase):
    """
    Tests for the REECB1 renewable energy electrical control model.
    """

    def test_init_flags(self):
        """
        Test REECB1 initialization with all flag combinations.

        Iterates over PFFLAG, VFLAG, QFLAG, PQFLAG in {0, 1} and
        verifies that TDS initialization succeeds for each combination.
        """
        case_path = get_case('ieee14/ieee14_reecb1.json')
        with open(case_path) as f:
            data = json.load(f)

        for pfflag, vflag, qflag, pqflag in itertools.product([0, 1], repeat=4):
            for row in data['REECB1']:
                if isinstance(row, dict):
                    row['PFFLAG'] = float(pfflag)
                    row['VFLAG'] = float(vflag)
                    row['QFLAG'] = float(qflag)
                    row['PQFLAG'] = float(pqflag)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f)
                tmpfile = f.name

            try:
                ss = andes.load(tmpfile, no_output=True, default_config=True)
                ss.PFlow.run()
                ss.TDS.init()

                label = f"PF={pfflag} VF={vflag} QF={qflag} PQF={pqflag}"
                self.assertTrue(ss.TDS.initialized,
                                f"REECB1 init failed for {label}")
            finally:
                os.unlink(tmpfile)

    def test_eig_with_zero_tf_states_not_at_end(self):
        """
        Test eigenvalue analysis when zero-Tf states (from ESST3A with TA=0)
        are not at the end of the state array.

        The REECB1 case adds renewable model states (REGCA1, REECB1, REPCA1)
        after ESST3A, pushing ESST3A's zero-Tf states to the middle of the
        state vector. This exercises the folding of zero-Tf states into the
        algebraic system in EIG.

        Regression test for a bug where zero-Tf states in the middle of the
        state array caused a singular state matrix.
        """
        ss = andes.run(
            get_case('ieee14/ieee14_reecb1.json'),
            routine='eig',
            default_config=True,
            no_output=True,
        )

        self.assertEqual(ss.exit_code, 0, "EIG should complete without error")

        # Zero-Tf states must be in the middle (not at the end) for this
        # test to exercise the bug. Verify that assumption holds.
        zidx = ss.EIG.zstate_idx
        self.assertGreater(len(zidx), 0, "Case must have zero-Tf states")
        self.assertTrue(
            zidx[-1] < ss.dae.n - 1,
            "Zero-Tf states should not be at the end of the state array "
            "(other models must add states after them)")

        # Basic sanity: no positive eigenvalues for a stable system
        self.assertEqual(ss.EIG.n_positive, 0,
                         "Stable system should have no positive eigenvalues")

    def test_vref0_zero_convention(self):
        """
        Test that Vref0=0 is replaced with the initial bus voltage.
        """
        case_path = get_case('ieee14/ieee14_reecb1.json')
        with open(case_path) as f:
            data = json.load(f)

        for row in data['REECB1']:
            if isinstance(row, dict):
                row['Vref0'] = 0.0

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            tmpfile = f.name

        try:
            ss = andes.load(tmpfile, no_output=True, default_config=True)
            ss.PFlow.run()
            ss.TDS.init()

            self.assertTrue(ss.TDS.initialized,
                            "REECB1 init failed with Vref0=0")

            np.testing.assert_array_almost_equal(
                ss.REECB1.Vref0r.v, ss.REECB1.v.v,
                err_msg="Vref0r should equal bus voltage when Vref0=0")
        finally:
            os.unlink(tmpfile)


if __name__ == '__main__':
    unittest.main()
