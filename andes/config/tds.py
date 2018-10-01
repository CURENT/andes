from . import ConfigBase
from ..utils.cached import cached


class Tds(ConfigBase):
    def __init__(self, **kwargs):
        self.fixt = True
        self.tstep = 1 / 30
        self.method = 'trapezoidal'
        self.method_alt = ['euler', 'trapezoidal', 'fwdeuler']
        self.t0 = 0.0
        self.tf = 20
        self.deltat = 0.01
        self.deltatmax = 1
        self.deltatmin = 0.0002
        self.maxit = 30
        self.tol = 1e-4
        self.disturbance = False
        self.error = 1
        self.qrt = False
        self.kqrt = 1
        self.compute_flows = True
        self.max_cache = 0
        super(Tds, self).__init__(**kwargs)

    @cached
    def config_descr(self):
        descriptions = {
            'fixt':
            'use fixed time step size',
            'tstep':
            'time step size',
            'method':
            'time domain integration method',
            't0':
            'starting simulation time',
            'tf':
            'ending simulation time',
            'maxit':
            'maximum iteration number for each integration step',
            'tol':
            'iteration error tolerance',
            'qrt':
            'quasi-real-time simulation speed',
            'kqrt':
            'quasi-rt runs at kqrt seconds per simulated second',
            'compute_flows':
            'post-compute bus injections and line flows at each step',
            'max_cache':
            'maximum allowed steps in varout memory, save to disk if reached, '
            '0 for unlimited memory cache',
        }
        return descriptions
