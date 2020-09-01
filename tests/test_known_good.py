import os
import dill
import unittest
import numpy as np

import andes
from andes.utils import get_case


class TestKnownResults(unittest.TestCase):

    sets = (('kundur/kundur_aw.xlsx', 'kundur_aw_10s.pkl', 10),
            ('kundur/kundur_full.xlsx', 'kundur_full_10s.pkl', 10),
            ('kundur/kundur_ieeeg1.xlsx', 'kundur_ieeeg1_10s.pkl', 10),
            ('kundur/kundur_ieeest.xlsx', 'kundur_ieeest_10s.pkl', 10),
            (('ieee14/ieee14.raw', 'ieee14/ieee14.dyr'), 'ieee14_2s.pkl', 2)
            )

    def tnc(self, case_path, pkl_path, tf):
        """
        Test and compare
        """
        addfile = None
        if isinstance(case_path, (tuple, list)):
            addfile = get_case(case_path[1])
            case_path = get_case(case_path[0])
        else:
            case_path = get_case(case_path)

        ss = compare_results(case_path, pkl_path, tf=tf, addfile=addfile)
        andes.main.misc(clean=True)
        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_known(self):
        for case, pkl, tf in self.sets:
            self.tnc(case, pkl, tf)


def compare_results(case, pkl_name, addfile=None, tf=10):
    ss = andes.load(case, addfile=addfile, default_config=True)

    ss.config.warn_limits = 0
    ss.config.warn_abnormal = 0
    ss.PFlow.run()

    ss.TDS.config.tf = tf
    ss.TDS.config.tstep = 1/30
    ss.TDS.config.tol = 1e-6
    ss.TDS.config.fixt = 1
    ss.TDS.config.shrinkt = 0
    ss.TDS.config.honest = 0
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
                                   err_msg=f"{case} test error",
                                   decimal=3)
    return ss
