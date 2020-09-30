# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import os
import andes
import andes.io.xlsx


def prep():
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
    def time_codegen(self):
        prep()


class TimePFlow:
    def setup(self):
        cases = 'kundur/kundur_full.xlsx'
        path, _ = os.path.split(__file__)
        path = os.path.join(path, '../../andes/cases/')

        self.ss = andes.main.System()
        self.ss.undill()
        andes.io.xlsx.read(self.ss, os.path.join(path, cases))
        self.ss.setup()
        self.ss.files.no_output = True

    def time_power_flow(self):
        self.ss.PFlow.run()


class TimeTDS:
    def setup(self):
        cases = 'kundur/kundur_full.xlsx'
        path, _ = os.path.split(__file__)
        path = os.path.join(path, '../../andes/cases/')

        self.ss = andes.main.System()
        self.ss.undill()
        andes.io.xlsx.read(self.ss, os.path.join(path, cases))
        self.ss.setup()
        self.ss.files.no_output = True
        self.ss.PFlow.run()

    def time_time_domain(self):
        self.ss.TDS.run()
