import numpy as np  # NOQA
from collections import OrderedDict
from andes.programs.base import ProgramBase
from cvxopt import matrix, sparse  # NOQA

import logging
logger = logging.getLogger(__name__)


class TDS(ProgramBase):

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('tf', 20.0),
                                     ('fixt', 1),
                                     ('tstep', 1/30))))
        self.tds_models = system.get_models_with_flag('tds')
        self.pflow_tds_models = system.get_models_with_flag(('tds', 'pflow'))

    def _initialize(self):
        system = self.system

        system.set_address(models=self.tds_models)
        system.set_dae_names(models=self.tds_models)
        system.dae.resize_array()
        system.link_external(models=self.tds_models)
        system.store_adder_setter()
        return system.initialize(self.tds_models, tds=True)

    def f_update(self):
        system = self.system
        # evaluate limiters, differential, algebraic, and jacobians
        system.vars_to_models()
        system.e_clear(models=self.pflow_tds_models)
        system.l_update_var(models=self.pflow_tds_models)
        system.f_update(models=self.pflow_tds_models)
        system.l_update_eq(models=self.pflow_tds_models)

    def g_update(self):
        system = self.system
        system.g_update(models=self.pflow_tds_models)
