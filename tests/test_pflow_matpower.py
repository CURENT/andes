import os
import unittest

import andes
import numpy as np
from andes.utils.paths import get_case
from andes.shared import deg2rad

andes.main.config_logger(30, file=True)

try:
    from matpower import start_instance
    m = start_instance()
    m.exit()
    MATPOWER_WORKING = True
except (ImportError, OSError):
    MATPOWER_WORKING = False


class TestRunMATPOWER(unittest.TestCase):
    """
    Test parsing and running matpower cases.
    """

    def setUp(self) -> None:
        self.cases = ('case5.m', 'case14.m', 'case118.m')

    def test_pflow_mpc_pool(self):
        case_path = [get_case(os.path.join('matpower', item)) for item in self.cases]
        andes.run(case_path, no_output=True, ncpu=2, pool=True, verbose=40, default_config=True)

    def test_pflow_mpc_process(self):
        case_path = [get_case(os.path.join('matpower', item)) for item in self.cases]
        andes.run(case_path, no_output=True, ncpu=2, pool=False, verbose=40, default_config=True)


@unittest.skipUnless(MATPOWER_WORKING, "MATPOWER not available")
class TestMATPOWEROct2Py(unittest.TestCase):

    def test_pflow_against_matpower_extra_test(self):

        m = start_instance()
        cases = ('case5.m', 'case14.m', 'case118.m')

        for name in cases:

            ss = andes.run(andes.get_case(os.path.join("matpower", name)),
                           no_output=True,
                           default_config=True)

            m.eval('clear mpc')
            andes.interop.matpower.to_matpower(m, 'mpc', ss)
            m.eval("mpopt = mpoption('verbose', 0, 'out.all', 0);")
            m.eval('mpc = runpf(mpc, mpopt);')

            v_mpc = np.ravel(m.eval('mpc.bus(:,8);'))
            a_mpc = np.ravel(m.eval('mpc.bus(:,9);')) * deg2rad

            v_andes = ss.Bus.v.v
            a_andes = ss.Bus.a.v

            np.testing.assert_almost_equal(v_mpc, v_andes, decimal=5)
            np.testing.assert_almost_equal(a_mpc, a_andes, decimal=5)
