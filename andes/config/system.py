from . import ConfigBase
from ..utils.cached import cached
import logging

logger = logging.getLogger(__name__)
try:
    from cvxoptklu import klu  # NOQA
    KLU = True
except ImportError:
    KLU = False


class System(ConfigBase):
    def __init__(self, **kwargs):
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
        self.static = 0
        self.nseries = 0
        self.forcepq = False
        self.forcez = False
        self.base = True
        super(System, self).__init__(**kwargs)

    @cached
    def config_descr(self):
        descriptions = {
            'freq': 'system base frequency',
            'mva': 'system base MVA',
            'distrsw': 'use distributed slack bus mode',
            'sparselib': 'sparse matrix library name',
            'export': 'help documentation export format',
            'coi': 'using Center of Inertia',
            'connectivity': 'connectivity check during TDS',
            'forcepq': 'force to use constant PQ load',
            'forcez': 'force to convert load to impedance',
            'base': 'convert  parameters to the system base',
        }
        return descriptions

    def check(self):
        """
        Check config data consistency

        Returns
        -------

        """
        if self.sparselib not in self.sparselib_alt:
            logger.warning("Invalid sparse library <{}>".format(self.sparselib))
            self.sparselib = 'umfpack'

        if self.sparselib == 'klu' and not KLU:
            logger.info("cvxoptklu import error. Fall back to umfpack".format(self.sparselib))
            self.sparselib = 'umfpack'

        return True
