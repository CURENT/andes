
import unittest
import logging

from oct2py import Oct2Py
from andes.main import run

VM = 7
VA = 8


class TestMATPOWER(unittest.TestCase):

    def setUp(self):
        self.mp = Oct2Py(logger=logging.getLogger())

        self.opts = self.mp.mpoption("pf.alg", "NR", "verbose", 0, "out.all", 0)

    def testPowerFlow(self):
        for name in ["case9"]:
            mpc = self.mp.loadcase(name)
            out = self.mp.runpf(mpc, self.opts)

            mpVm = out.bus[:, VM]
            mpVa = out.bus[:, VA]

            system = run("/usr/local/matpower/data/" + name + ".m", routine=["pflow"])

            idx, names, Vm, Va, Pg, Qg, Pl, Ql = system.get_busdata()

            for i in range(len(mpVm)):
                self.assertAlmostEqual(Vm[i], mpVm[i], msg="{} - Vm[{}]".format(name, i))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    unittest.main()
