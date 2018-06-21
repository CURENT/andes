from ..settings.base import SettingsBase
from ..utils.cached import cached


class SPF(SettingsBase):
    def __init__(self):
        self.flatstart = False
        self.maxit = 100
        self.pv2pq = False
        self.ipv2pq = 4
        self.npv2pq = 1
        self.iter = 0
        self.report = 'default'
        self.show = True
        self.solver = 'NR'
        self.solve_alt = ['NR', 'FDBX', 'FDXB']
        self.sortbuses = 'data'
        self.sortbuses_alt = ['data', 'idx']
        self.static = False
        self.switch2nr = False
        self.units = 'pu'
        self.units_alt = ['pu', 'nominal']
        self.usedegree = True
        self.solved = False

    @cached
    def doc_help(self):
        descriptions = {'flatstart': 'flat start for power flow problem',
                        'maxit': 'the maximum iteration number',
                        'pv2pq': 'check Q limit and convert PV to PQ',
                        'ipv2pq': 'the interation from which to convert PV to PQ',
                        'npv2pq': 'the maximum number of PVs to convert in one iteration',
                        'solver': 'power flow routine solver type',
                        'sortbuses': 'index to sort buses',
                        'switch2nr': 'switch to Newton Raphson method if non-convergence',
                        'units': 'the unit for the power flow report',
                        'usedegree': 'use degree in the power flow report',
                        }
        return descriptions
