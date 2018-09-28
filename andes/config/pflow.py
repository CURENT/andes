from . import ConfigBase
from ..utils.cached import cached


class Pflow(ConfigBase):
    def __init__(self, **kwargs):
        self.flatstart = False
        self.tol = 1e-6
        self.maxit = 100
        self.pv2pq = False
        self.ipv2pq = 4
        self.npv2pq = 1
        self.iter = 0
        self.report = 'default'
        self.show = True
        self.method = 'NR'
        self.method_alt = ['NR', 'FDPF', 'FDBX', 'FDXB']
        self.sortbuses = 'data'
        self.sortbuses_alt = ['data', 'idx']
        self.static = False
        self.switch2nr = False
        self.units = 'pu'
        self.units_alt = ['pu', 'nominal']
        self.usedegree = True
        super(Pflow, self).__init__(**kwargs)

    @cached
    def config_descr(self):
        descriptions = {
            'flatstart': 'flat start for power flow problem',
            'maxit': 'the maximum iteration number',
            'pv2pq': 'check Q limit and convert PV to PQ',
            'ipv2pq': 'the interation from which to convert PV to PQ',
            'npv2pq': 'the maximum number of PVs to convert in one iteration',
            'method': 'power flow routine solver type',
            'sortbuses': 'index to sort buses',
            'switch2nr': 'switch to Newton Raphson method if non-convergence',
            'units': 'the unit for the power flow report',
            'usedegree': 'use degree in the power flow report',
        }
        return descriptions

    def check(self):
        if self.method not in self.method_alt:
            self.method = 'NR'
        return True
