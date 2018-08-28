from . import SettingsBase
from ..consts import *
from ..utils.cached import cached


class Settings(SettingsBase):
    def __init__(self):
        self.verbose = INFO
        self.verbose_alt = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.freq = 60.0
        self.mva = 100.0
        self.distrsw = False
        self.sparselib = 'klu'
        self.sparselib_alt = ['klu', 'umfpack']
        self.export = 'txt'
        self.export_alt = ['txt', 'latex']
        self.coi = False
        self.connectivity = False
        self.error = 1
        self.tol = 1e-6
        self.static = 0
        self.nseries = 0
        self.forcepq = False
        self.forcez = False
        self.base = True
        self.dime_enable = False
        self.dime_name = 'sim'
        # self.dime_server = 'tcp://127.0.0.1:5000'
        # self.dime_server = 'tcp://10.129.132.192:9999'
        # self.dime_server = 'tcp://160.36.56.211:9900'
        # self.dime_server = 'tcp://160.36.58.82:8898'
        self.dime_server = 'ipc:///tmp/dime'
        self.progressbar = False

    @property
    def wb(self):
        return 2 * pi * self.freq

    @cached
    def descr(self):
        descriptions = {'verbose': 'program logging level',
                        'freq': 'system base frequency',
                        'mva': 'system base MVA',
                        'distrsw': 'use distributed slack bus mode',
                        'sparselib': 'sparse matrix library name',
                        'export': 'help documentation export format',
                        'coi': 'using Center of Inertia',
                        'connectivity': 'connectivity check during TDS',
                        'tol': 'iteration error tolerance',
                        'forcepq': 'force to use constant PQ load',
                        'forcez': 'force to convert load to impedance',
                        'base': 'convert model parameters to the system base',
                        }
        return descriptions
