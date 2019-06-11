import logging

from cvxopt import mul, div, matrix

from andes.consts import Gx, Fy0, Gy0
from andes.models.base import ModelBase
from andes.utils.math import zeros

logger = logging.getLogger(__name__)


class BArea(ModelBase):
    """
    Balancing area class. This class defines power balancing area on top of the `Area` class for calculating
    center of inertia frequency, total inertia, expected power and area control error.
    """
    def __init__(self, system, name):
        super(BArea, self).__init__(system, name)
        self._group = 'Calculation'
        self._data.update({
            'area': None,
            'syn': None,
            'beta': 0,
        })
        self._descr.update({'area': 'Idx of Area',
                            'beta': 'Beta coefot multiply by the pu freq. deviation',
                            'syn': 'Indices of generators for computing COI'
                            })
        self._units.update({'syn': 'list'})
        self._mandatory.extend(['area', 'syn', 'beta'])
        self._algebs.extend(['Pexp', 'fcoi', 'ace'])
        self.calls.update({
            'gcall': True,
            'init1': True,
            'jac0': True,
        })
        self._service.extend(['P0', 'Mtot', 'M', 'usyn', 'wsyn'])
        self._fnamey.extend(['P_{exp}', 'f_{coi}', 'ace'])
        self._params.extend(['beta'])
        self._init()

    def init1(self, dae):

        for item in self._service:
            self.__dict__[item] = [[]] * self.n

        # Start with frequency
        for idx, item in enumerate(self.syn):
            self.M[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
            self.Mtot[idx] = sum(self.M[idx])
            self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
            self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
            dae.y[self.fcoi[idx]] = sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        # Get BA Export Power
        self.copy_data_ext('Area', field='area_P0', dest='P0', idx=self.area)
        dae.y[self.Pexp] = self.P0
        dae.y[self.ace] = 0

    def gcall(self, dae):

        # the value below gets updated at each iteration in `seriesflow`
        P = self.read_data_ext('Area', field='area_P0', idx=self.area)
        dae.g[self.Pexp] = dae.y[self.Pexp] - P

        for idx, item in enumerate(self.syn):
            self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
            dae.g[self.fcoi[idx]] = dae.y[self.fcoi[idx]] - \
                sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        ACE = (P - self.P0) - mul(self.beta, (1 - dae.y[self.fcoi]))

        dae.g[self.ace] = dae.y[self.ace] + ACE

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.Pexp, self.Pexp)
        dae.add_jac(Gy0, 1, self.fcoi, self.fcoi)
        dae.add_jac(Gy0, 1, self.ace, self.ace)
        dae.add_jac(Gy0, 1, self.ace, self.Pexp)
        dae.add_jac(Gy0, self.beta, self.ace, self.fcoi)


class AGCBase(ModelBase):
    """
    Base AGC class. The allocation of Pagc will be based on inverse droop (iR)
    """
    def __init__(self, system, name):
        super(AGCBase, self).__init__(system, name)
        self._group = 'AGCGroup'
        self._data.update({'BArea': None,
                           'Ki': 0.05,
                           })
        self._descr.update({'BArea': 'Idx of BArea',
                            'Ki': 'Integral gain of ACE',
                            })
        self._mandatory.extend(['BArea', 'Ki'])
        self._states.extend(['Pagc'])
        self.calls.update({'init1': True,
                           'gcall': True,
                           'fcall': True,
                           'jac0': True,
                           'gycall': True
                           })
        self._service.extend(['ace', 'iR', 'iRtot'])
        self._fnamex.extend(['P_{agc}^{total}'])
        self._params.extend(['Ki'])

    def init1(self, dae):
        self.copy_data_ext('BArea', field='ace', idx=self.BArea)

    def fcall(self, dae):
        dae.f[self.Pagc] = mul(self.Ki, dae.y[self.ace])

    def gcall(self, dae):
        pass

    def jac0(self, dae):
        dae.add_jac(Fy0, self.Ki, self.Pagc, self.ace)

    def gycall(self, dae):
        pass


class AGCSyn(AGCBase):
    """AGC for synchronous generators. This class changes the setpoints by modifying the generator pm."""
    def __init__(self, system, name):
        super(AGCSyn, self).__init__(system, name)
        self._data.update({'syn': None})
        self._descr.update({'syn': 'Indices of synchronous generators for AGC'})
        self._units.update({'syn': 'list'})
        self._mandatory.extend(['syn'])
        self._service.extend(['pm', 'usyn'])
        self._init()

    def init1(self, dae):
        super(AGCSyn, self).init1(dae)
        self.pm = [[]] * self.n
        self.iR = [[]] * self.n
        self.usyn = [[]] * self.n
        self.iRtot = [[]] * self.n

        for idx, item in enumerate(self.syn):
            self.pm[idx] = self.read_data_ext('Synchronous', field='pm', idx=item)
            self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
            self.iR[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
            self.iRtot[idx] = sum(mul(self.usyn[idx], self.iR[idx]))

    def gcall(self, dae):
        super(AGCSyn, self).gcall(dae)

        # Kgen and each item in `self.pm`, `self.usyn`, and `self.Pagc` is a list
        #   Do not get rid of the `for` loop, since each of them is a matrix operation

        for idx, item in enumerate(self.syn):
            Kgen = div(self.iR[idx], self.iRtot[idx])
            dae.g[self.pm[idx]] -= mul(self.usyn[idx], Kgen, dae.x[self.Pagc[idx]])

    def gycall(self, dae):
        super(AGCSyn, self).gycall(dae)

        # Do not get rid of the for loop; for each `idx` it is a matrix operation

        for idx, item in enumerate(self.syn):
            Kgen = div(self.iR[idx], self.iRtot[idx])
            dae.add_jac(Gx, -mul(self.usyn[idx], Kgen), self.pm[idx], self.Pagc[idx])


class AGC(AGCSyn):
    """Alias for class <AGCSyn>"""
    pass


class AGCTG(AGCBase):
    """AGC class that modifies the turbine governor power reference. Links to TG1 only."""
    def __init__(self, system, name):
        super(AGCTG, self).__init__(system, name)
        self._data.update({'tg': None})
        self._mandatory.extend(['tg'])
        self._descr.update({'tg': 'Indices of turbine governors for AGC'})
        self._units.update({'tg': 'list'})
        self._service.extend(['pin', 'R', 'iR', 'iRtot'])

        self._init()

    def init1(self, dae):
        super(AGCTG, self).init1(dae)
        self.pin = [[]] * self.n
        self.R = [[]] * self.n
        self.iR = [[]] * self.n
        self.iRtot = [[]] * self.n

        for idx, item in enumerate(self.tg):
            self.pin[idx] = self.read_data_ext(model='Governor', field='pin', idx=item)
            self.R[idx] = self.read_data_ext(model='Governor', field='R', idx=item)
            self.iR[idx] = div(1, self.R[idx])
            self.iRtot[idx] = sum(self.iR[idx])

    def gcall(self, dae):
        super(AGCTG, self).gcall(dae)
        for idx, item in enumerate(self.tg):
            Ktg = div(self.iR[idx], self.iRtot[idx])
            dae.g[self.pin[idx]] += mul(Ktg, dae.x[self.Pagc[idx]])

    def gycall(self, dae):
        super(AGCTG, self).gycall(dae)
        for idx, item in enumerate(self.tg):
            Ktg = div(self.iR[idx], self.iRtot[idx])
            dae.add_jac(Gx, Ktg, self.pin[idx], self.Pagc[idx])


class AGCVSCBase(object):
    """
    Base class for AGC using VSC. Modifies the ref1 for PV or PQ-controlled VSCs. This class must be
    inherited with subclasses of AGCBase
    """
    def __init__(self, system, name):
        self.system = system
        self._data.update({'vsc': None,
                           'Rvsc': None,
                           })
        self._descr.update({'vsc': 'Indices of VSCs to control',
                            'Rvsc': 'Droop coefficients for the VSCs'})
        self._units.update({'tg': 'list',
                            'Rvsc': 'list'})
        self._mandatory.extend(['vsc', 'Rvsc'])
        self._service.extend(['uvsc', 'ref1'])
        self._init()

    def init1(self, dae):
        self.ref1 = [[]] * self.n
        self.uvsc = [[]] * self.n

        # manually convert self.Rvsc to a list of matrices
        self.Rvsc = [matrix(item) for item in self.Rvsc]
        self.iRvsc = [div(1, item) for item in self.Rvsc]

        # Only PV or PQ-controlled VSCs are acceptable
        for agc_idx, item in enumerate(self.vsc[:]):
            pv_or_pq = self.read_data_ext('VSCgroup', field="PV", idx=item) + \
                        self.read_data_ext('VSCgroup', field='PQ', idx=item)

            valid_vsc_list = list()
            valid_vsc_R = list()
            for i, (vsc_idx, valid) in enumerate(zip(item, pv_or_pq)):
                if valid:
                    valid_vsc_list.append(vsc_idx)
                    # TODO: fix the hard-coded `vsc_Idx` below
                    valid_vsc_R.append(self.Rvsc[agc_idx][i])
                else:
                    logger.warning('VSC <{}> is not a PV or PQ type, thus cannot be used for AGC.'.format(vsc_idx))
            self.vsc[agc_idx] = valid_vsc_list

        for agc_idx, item in enumerate(self.vsc):
            # skip elements that contain no valid VSC index
            if len(item) == 0:
                continue

            # retrieve status `uvsc`
            self.uvsc[agc_idx] = self.read_data_ext('VSCgroup', field='u', idx=item)
            self.ref1[agc_idx] = self.read_data_ext('VSCgroup', field='ref1', idx=item)
            # Add `Rvsc` to Mtot
            self.iRtot[agc_idx] += sum(mul(self.uvsc[agc_idx], self.iRvsc[agc_idx]))

    def gcall(self, dae):
        for agc_idx, item in enumerate(self.vsc):
            if len(item) == 0:
                continue

            Kvsc = div(self.iRvsc[agc_idx], self.iRtot[agc_idx])
            dae.g[self.ref1[agc_idx]] -= mul(self.uvsc[agc_idx], Kvsc, dae.x[self.Pagc[agc_idx]])

    def gycall(self, dae):

        for agc_idx, item in enumerate(self.vsc):
            if len(item) == 0:
                continue

            Kvsc = div(self.iRvsc[agc_idx], self.iRtot[agc_idx])
            dae.add_jac(Gx, -mul(self.uvsc[agc_idx], Kvsc), self.ref1[agc_idx], self.Pagc[agc_idx])


class AGCTGVSC(AGCTG, AGCVSCBase):
    """AGC class that modifies the turbine governor and VSC pref"""
    def __init__(self, system, name):
        AGCTG.__init__(self, system, name)
        AGCVSCBase.__init__(self, system, name)
        self._init()

    def init1(self, dae):
        AGCTG.init1(self, dae)
        AGCVSCBase.init1(self, dae)

    def jac0(self, dae):
        AGCTG.jac0(self, dae)

    def gcall(self, dae):
        AGCTG.gcall(self, dae)
        AGCVSCBase.gcall(self, dae)

    def gycall(self, dae):
        AGCTG.gycall(self, dae)
        AGCVSCBase.gycall(self, dae)

    def fcall(self, dae):
        AGCTG.fcall(self, dae)


class AGCSynVSC(AGCSyn, AGCVSCBase):
    """AGC class that modifies Synchronous pm and VSC pref"""
    def __init__(self, system, name):
        AGCSyn.__init__(self, system, name)
        AGCVSCBase.__init__(self, system, name)
        self._init()

    def init1(self, dae):
        AGCSyn.init1(self, dae)
        AGCVSCBase.init1(self, dae)

    def jac0(self, dae):
        AGCSyn.jac0(self, dae)

    def gcall(self, dae):
        AGCSyn.gcall(self, dae)
        AGCVSCBase.gcall(self, dae)

    def gycall(self, dae):
        AGCSyn.gycall(self, dae)
        AGCVSCBase.gycall(self, dae)

    def fcall(self, dae):
        AGCSyn.fcall(self, dae)


class eAGC(ModelBase):
    def __init__(self, system, name):
        super(eAGC, self).__init__(system, name)
        self._group = 'Control'
        self._data.update({
            'cl': None,
            'tl': 0,
            'Pl': None,
            'BA': None,
        })
        self._descr.update({
            'cl': 'Loss sharing coefficient (vector)',
            'tl': 'Time of generator loss',
            'Pl': 'Loss of power generation in pu (vector)',
            'BA': 'Balancing Area that support the Gen loss',
        })
        self._mandatory.extend(['cl', 'tl', 'Pl', 'BA'])
        self.calls.update({
            'gcall': True,
            'init1': True,
            'jac0': False,
            'fcall': False,
        })
        self._service.extend(['ace', 'en'])
        self._params.extend(['cl', 'tl', 'Pl'])
        self._init()

    def init1(self, dae):
        self.ace = [[]] * self.n
        for idx, item in enumerate(self.BA):
            self.ace[idx] = self.read_data_ext('BArea', field='ace', idx=item)

        self.en = zeros(self.n, 1)

    def switch(self):
        """Switch if time for eAgc has come"""
        t = self.system.dae.t
        for idx in range(0, self.n):
            if t >= self.tl[idx]:
                if self.en[idx] == 0:
                    self.en[idx] = 1
                    logger.info('Extended ACE <{}> activated at t = {}.'.format(self.idx[idx], t))

    def gcall(self, dae):
        self.switch()

        for idx in range(0, self.n):
            dae.g[self.ace[idx]] -= mul(self.en[idx], self.cl[:, idx],
                                        self.Pl[idx])
