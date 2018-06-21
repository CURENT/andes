from ..settings.base import SettingsBase
from ..consts import *
from ..utils.cached import cached


class Settings(SettingsBase):
    def __init__(self):
        self.verbose = INFO
        self.verbose_alt = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.freq = 60.0
        self.mva = 100.0
        self.distrsw = False
        self.sparselib = 'umfpack'
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
        self.forcez = True
        self.base = True
        self.progressbar = False

    @property
    def wb(self):
        return 2 * pi * self.freq

    @cached
    def doc_help(self):
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
                        'base': 'per-unitize parameters to the common base',
                        }
        return descriptions
