from ..settings.base import SettingsBase
from ..utils.cached import cached


class TDS(SettingsBase):
    def __init__(self):
        self.fixt = True
        self.tstep = 0.02
        self.method = 'trapezoidal'
        self.method_alt = ['euler', 'trapezoidal','fwdeuler']
        self.t0 = 0.0
        self.tf = 20
        self.deltat = 0.01
        self.deltatmax = 1
        self.deltatmin = 0.0002
        self.maxit = 30
        self.tol = 1e-06
        self.disturbance = False
        self.error = 1
        self.method_desc = {'euler': 'Implicit Euler',
                            'trapezoidal': 'Implicit Trapezoidal',
                            'fwdeuler': 'Explicit Euler'}

    @cached
    def doc_help(self):
        descriptions = {'fixt': 'use fixed time step size',
                        'tstep': 'time step size',
                        'method': 'time domain integration method',
                        't0': 'starting simulation time',
                        'tf': 'ending simulation time',
                        'maxit': 'maximum iteration number for each integration step',
                        'tol': 'iteration error tolerance',
                        }
        return descriptions
