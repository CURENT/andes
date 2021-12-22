import os
import logging
import unittest

import dill
import numpy as np

import andes
from andes.utils import get_case

logger = logging.getLogger(__name__)


class TestKnownResults(unittest.TestCase):

    sets = (('kundur/kundur_aw.json', 'kundur_aw_10s.pkl', 10),
            ('kundur/kundur_full.json', 'kundur_full_10s.pkl', 10),
            ('kundur/kundur_ieeeg1.json', 'kundur_ieeeg1_10s.pkl', 10),
            ('kundur/kundur_ieeest.json', 'kundur_ieeest_10s.pkl', 10),
            ('ieee14/ieee14_fault.json', 'ieee14_fault_2s.pkl', 2),
            (('ieee14/ieee14.raw', 'ieee14/ieee14.dyr'), 'ieee14_2s.pkl', 2),
            )

    def tnc(self, case_path, pkl_path, tf, pkl_prefix='pkl'):
        """
        Test and compare
        """
        addfile = None
        if isinstance(case_path, (tuple, list)):
            addfile = get_case(case_path[1])
            case_path = get_case(case_path[0])
        else:
            case_path = get_case(case_path)

        ss = compare_results(case_path,
                             os.path.join(pkl_prefix, pkl_path),
                             tf=tf,
                             addfile=addfile)

        self.assertEqual(ss.exit_code, 0, "Exit code is not 0.")

    def test_known(self):
        for case, pkl, tf in self.sets:
            self.tnc(case, pkl, tf)


def compare_results(case, pkl_name, addfile=None, tf=10):
    ss = andes.load(case, addfile=addfile, tf=tf,
                    default_config=True, no_output=True,
                    )

    ss.config.warn_limits = 0
    ss.config.warn_abnormal = 0
    ss.PFlow.run()

    ss.TDS.config.tstep = 1/30
    ss.TDS.config.tol = 1e-4
    ss.TDS.config.fixt = 1
    ss.TDS.config.shrinkt = 0
    ss.TDS.run()

    test_dir = os.path.dirname(__file__)
    f = open(os.path.join(test_dir, pkl_name), 'rb')
    results = dill.load(f)
    f.close()

    indices = np.hstack((ss.GENROU.omega.a,
                         ss.dae.n + ss.GENROU.tm.a,
                         ss.dae.n + ss.GENROU.vf.a))
    logger.info(case)
    logger.info("This results: %s", ss.dae.xy[indices])
    logger.info("Known results: %s", results)

    np.testing.assert_almost_equal(ss.dae.xy[indices],
                                   results,
                                   err_msg=f"{case} test error",
                                   decimal=2,)
    # actual algeb. errors will be `tstep` times greater.
    # combined with a tol of 1e-4, accuracy is to the 2nd decimal place.

    return ss
