import logging
import numpy as np

import slycot
import control as ctl
from andes.routines.base import BaseRoutine

logger = logging.getLogger(__name__)
__cli__ = 'red'


class RED(BaseRoutine):
    def __init__(self, system, config):
        BaseRoutine.__init__(self, system, config)
        self.As = None

    def summary(self):
        logger.info("-> Model Reduction")

    def run(self):
        self.summary()
        self.system.TDS._initialize()
        self.As = self.system.EIG.calc_state_matrix()

        # assume the actuator is the EXDC2
        # modify VREF for stabilizing

        system = self.system
        dae = self.system.dae
        EXDC2 = self.system.EXDC2
        GENROU = self.system.GENROU

        nu = EXDC2.n

        B = np.zeros((dae.n, nu))
        for ii in range(EXDC2.n):
            B[EXDC2.LA_x.a[ii], ii] = (EXDC2.LA.lim.zi[ii] * EXDC2.TC.v[ii] * EXDC2.KA.v[ii]) / \
                                      (EXDC2.TB.v[ii] * EXDC2.TA.v[ii])

        C = np.zeros((GENROU.n, dae.n))
        for ii in range(GENROU.n):
            C[ii, GENROU.omega.a[ii]] = 1

        D = np.zeros((GENROU.n, nu))

        sys_full = ctl.ss(self.As, B, C, D)

        return sys_full