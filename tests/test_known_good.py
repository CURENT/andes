import os
import unittest

import dill
import numpy as np

import andes
from andes.utils import get_case


class TestKnownResults(unittest.TestCase):

    sets = (('kundur/kundur_aw.xlsx', 'kundur_aw_10s.pkl'),
            ('kundur/kundur_full.xlsx', 'kundur_full_10s.pkl'),
            ('kundur/kundur_ieeeg1.xlsx', 'kundur_ieeeg1_10s.pkl'),
            ('kundur/kundur_ieeest.xlsx', 'kundur_ieeest_10s.pkl')
            )

    def tnc(self, case_path, pkl_path):
        """
        Test and compare
        """
        ss = compare_results(case_path, pkl_path)
        andes.main.misc(clean=True)
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_known(self):
        for (case, pkl) in self.sets:
            self.tnc(case, pkl)


def compare_results(case, pkl_name, tf=10):
    case_path = get_case(case)
    ss = andes.run(case_path)

    ss.TDS.config.tf = tf
    ss.TDS.config.tstep = 1/30
    ss.TDS.run()

    test_dir = os.path.dirname(__file__)
    f = open(os.path.join(test_dir, pkl_name), 'rb')
    results = dill.load(f)
    f.close()

    indices = np.hstack((ss.GENROU.omega.a,
                         ss.dae.n + ss.GENROU.tm.a,
                         ss.dae.n + ss.GENROU.vf.a))

    np.testing.assert_almost_equal(ss.dae.xy[indices],
                                   results,
                                   err_msg=f"{case} test error")
    return ss
