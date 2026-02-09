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
                ss.setup()
                ss.PFlow.run()
                ss.TDS.init()

                label = f"PF={pfflag} VF={vflag} QF={qflag} PQF={pqflag}"
                self.assertTrue(ss.TDS.initialized,
                                f"REECB1 init failed for {label}")
            finally:
                os.unlink(tmpfile)

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
            ss.setup()
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
