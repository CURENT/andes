from cvxopt import matrix, spmatrix  # NOQA
from cvxopt import mul, div, exp  # NOQA
from andes.consts import Fx, Fy, Gx, Gy, Fx0, Fy0, Gx0, Gy0  # NOQA
from .base import ModelBase
from andes.utils.math import zeros
import logging

logger = logging.getLogger(__name__)


class BArea(ModelBase):
    def __init__(self, system, name):
        super(BArea, self).__init__(system, name)
        self._group = 'Calculation'
        self._data.update({
            'area': None,
            'syn': None,
            'beta': 0,
        })
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
            self.M[idx] = self.read_data_ext(
                'Synchronous', field='M', idx=item)
            self.Mtot[idx] = sum(self.M[idx])
            self.usyn[idx] = self.read_data_ext(
                'Synchronous', field='u', idx=item)
            self.wsyn[idx] = self.read_data_ext(
                'Synchronous', field='omega', idx=item)
            dae.y[self.fcoi[idx]] = sum(
                mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        # Get BA Export Power
        self.copy_data_ext('Area', field='area_P0', dest='P0', idx=self.area)
        dae.y[self.Pexp] = self.P0
        dae.y[self.ace] = 0

    def gcall(self, dae):

        P = self.read_data_ext('Area', field='area_P0', idx=self.area)
        dae.g[self.Pexp] = dae.y[self.Pexp] - P

        for idx, item in enumerate(self.syn):
            self.wsyn[idx] = self.read_data_ext(
                'Synchronous', field='omega', idx=item)
            dae.g[self.fcoi[idx]] = dae.y[self.fcoi[idx]] - sum(
                mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        ACE = (P - self.P0) - mul(self.beta, (1 - dae.y[self.fcoi]))

        dae.g[self.ace] = dae.y[self.ace] + ACE

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.Pexp, self.Pexp)
        dae.add_jac(Gy0, 1, self.fcoi, self.fcoi)
        dae.add_jac(Gy0, 1, self.ace, self.ace)
        dae.add_jac(Gy0, 1, self.ace, self.Pexp)
        dae.add_jac(Gy0, self.beta, self.ace, self.fcoi)


class AGC(ModelBase):
    def __init__(self, system, name):
        super(AGC, self).__init__(system, name)
        self._group = 'Control'
        self._data.update({
            'BArea': None,
            'syn': None,
            'Ki': 0.05,
        })
        self._mandatory.extend(['BArea', 'syn', 'Ki'])
        self._states.extend(['Pagc'])
        self.calls.update({
            'gcall': True,
            'init1': True,
            'jac0': True,
            'fcall': True,
        })
        self._service.extend(['ace', 'pm', 'M', 'usyn', 'Mtot'])
        self._fnamex.extend(['P_{agc tot}'])
        self._params.extend(['Ki'])
        self._init()

    def init1(self, dae):
        self.pm = [[]] * self.n
        self.M = [[]] * self.n
        self.usyn = [[]] * self.n
        self.Mtot = [[]] * self.n

        for idx, item in enumerate(self.syn):
            self.pm[idx] = self.read_data_ext(
                'Synchronous', field='pm', idx=item)
            self.usyn[idx] = self.read_data_ext(
                'Synchronous', field='u', idx=item)
            self.M[idx] = self.read_data_ext(
                'Synchronous', field='M', idx=item)
            self.Mtot[idx] = sum(mul(self.usyn[idx], self.M[idx]))

        self.copy_data_ext('BArea', field='ace', idx=self.BArea)

    def fcall(self, dae):
        dae.f[self.Pagc] = mul(self.Ki, dae.y[self.ace])

    def gcall(self, dae):
        for idx, item in enumerate(self.syn):
            Kgen = div(self.M[idx], self.Mtot[idx])
            dae.g[self.pm[idx]] -= mul(self.usyn[idx], Kgen,
                                       dae.x[self.Pagc[idx]])

    def jac0(self, dae):
        dae.add_jac(Fy0, self.Ki, self.Pagc, self.ace)

    def gycall(self, dae):
        for idx, item in enumerate(self.syn):
            Kgen = div(self.M[idx], self.Mtot[idx])
            dae.add_jac(Gx, -mul(self.usyn[idx], Kgen), self.pm[idx],
                        self.Pagc[idx])


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
                    logger.info(
                        'Extended ACE <{}> activated at t = {}.'.format(
                            self.idx[idx], t))

    def gcall(self, dae):
        self.switch()

        for idx in range(0, self.n):
            dae.g[self.ace[idx]] -= mul(self.en[idx], self.cl[:, idx],
                                        self.Pl[idx])
