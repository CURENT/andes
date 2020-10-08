# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import os
import andes
import andes.io


def base_path():
    path, _ = os.path.split(__file__)
    return os.path.join(path, '../../andes/cases/')


def load_system(case_path):
    """
    Load a test case and return an ``andes.system.System`` object.
    """
    ss = andes.system.System(case_path)
    ss.undill()
    andes.io.parse(ss)
    ss.setup()
    ss.files.no_output = True
    ss.TDS.config.tol = 1e-6
    return ss


def prep():
    """
    Generate numerical code.
    """
    ss = andes.System()
    try:
        try:
            try:
                ss.prepare(incremental=True)
                return
            except TypeError:
                pass
            ss.prepare(incremental=True, quick=True)
            return
        except TypeError:
            pass
        ss.prepare(quick=True)
        return
    except TypeError:
        pass

    ss.prepare()


class TimeCodeGen:
    def setup(self):
        prep()

    def time_codegen(self):
        """
        Empty function to invoke its setup.
        """
        pass


class TimePFlow:
    def setup(self):
        case = "kundur/kundur_full.xlsx"
        self.ss = load_system(os.path.join(base_path(), case))

    def time_kundur_power_flow(self):
        self.ss.PFlow.run()


class TimePFlowMPC:
    def setup(self):
        case = "../../../matpower/data/case9241pegase.m"
        self.ss = load_system(os.path.join(base_path(), case))

    def time_9241_power_flow(self):
        self.ss.PFlow.run()


class TimeTDS:
    def setup(self):
        case = "kundur/kundur_full.xlsx"
        self.ss = load_system(os.path.join(base_path(), case))
        self.ss.PFlow.run()

    def time_kundur_time_domain(self):
        self.ss.TDS.run()


class TimeTDSAW:
    def setup(self):
        case = "kundur/kundur_aw.xlsx"
        self.ss = load_system(os.path.join(base_path(), case))
        self.ss.PFlow.run()

    def time_kundur_time_domain_aw(self):
        self.ss.TDS.run()
