from .base import ModelBase
from cvxopt import spmatrix, spdiag, matrix
from ..utils.math import zeros
import logging

logger = logging.getLogger(__name__)


class Zone(ModelBase):
    """Zone class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Topology'
        self._name = 'Zone'
        self._params.extend(['pdes', 'ptol', 'isw'])
        self._descr.update({
            'pdes':
            "Desired net interchange leaving this area",
            'ptol':
            'Interchange tolerance bandwidth',
            'isw':
            'Area slack bus idx',
        })
        self._units.update({
            'pdes': 'MW',
            'ptol': 'MW',
        })
        self._data.update({
            'pdes': 0.0,
            'ptol': 10.0,
            'isw': 0,
        })
        self._powers.extend(['pdes', 'ptol'])
        self.calls.update({'pflow': True})
        self._service.extend(['uname', 'fname'])
        self._init()

        # shape of `self.incidence` is (self.n, self.n) with
        #   the diagonals are the number of lines in the area, and
        #   the off-diagonals are the number of tie-lines between area  `row`
        #  and `col`

        self.incidence = None
        self.tielines = dict()  # [tielines, boarder bus] in each area
        self.buses = dict()  # buses in each area
        # self.area_flows = dict()  # tieline outgoings [P, Q] for each area
        self.area_P0 = []
        self.area_Q0 = []
        self.interchange = dict(
        )  # a sparse matrix with the interchange values
        self.inter_flows = dict()
        self.inter_varout = matrix([])

        self.inter_uname = list()
        self.inter_fname = list()
        self.n_combination = 0

    def setup(self):
        # TODO: account for >1 area/region/zone
        super().setup()
        var = self._name.lower()
        for idx, uid in self.system.Bus.uid.items():
            code = self.system.Bus.__dict__[var][uid]
            if code and code not in self.idx:
                logger.warning('{} <{}> not defined.'.format(
                    self._name, code))
            if code not in self.buses.keys():
                self.buses[code] = list()
            self.buses[code].append(idx)

        x, a, b = list(), list(), list()
        for idx, uid in self.system.Line.uid.items():
            bus1 = self.system.Line.bus1[uid]
            bus2 = self.system.Line.bus2[uid]
            code1 = self.system.Bus.__dict__[var][self.system.Bus.uid[bus1]]
            code2 = self.system.Bus.__dict__[var][self.system.Bus.uid[bus2]]

            if code1 == 0 or code2 == 0:
                continue
            if code1 != code2:
                if code1 not in self.tielines.keys():
                    self.tielines[code1] = list()
                if code2 not in self.tielines.keys():
                    self.tielines[code2] = list()

                self.tielines[code1].append([idx, bus1])
                self.tielines[code2].append([idx, bus2])

            if code1 != code2:
                if code1 not in self.interchange:
                    self.interchange[code1] = dict()
                if code2 not in self.interchange:
                    self.interchange[code2] = dict()
                if code2 not in self.interchange[code1]:
                    self.interchange[code1][code2] = []
                if code1 not in self.interchange[code2]:
                    self.interchange[code2][code1] = []

                self.interchange[code1][code2].append([idx, bus1])
                self.interchange[code2][code1].append([idx, bus2])

            int_code1 = self.uid[code1]
            int_code2 = self.uid[code2]

            x.append(0.5) if int_code1 == int_code2 else x.append(1)
            a.append(int_code1)
            b.append(int_code2)

        half_incidence = spmatrix(x, a, b, (self.n, self.n), 'd')
        self.incidence = half_incidence + half_incidence.T

        incidence = self.incidence - spdiag(self.incidence[0::self.n + 1])
        I, J, V = incidence.I, incidence.J, incidence.V

        self.n_combination = 0
        for i, j, v in zip(I, J, V):
            if not v:
                continue
            self.n_combination += 1
            self.uname.append('{}-{}'.format(self.name[i], self.name[j]))
            self.fname.append('{}-{}'.format(self.name[i], self.name[j]))

        self.area_P0 = zeros(self.n, 1)
        self.area_Q0 = zeros(self.n, 1)

    def seriesflow(self, dae):
        for code in self.idx:
            int_idx = self.uid[code]
            pairs = self.tielines.get(code, None)
            if not pairs:
                continue
            lines = [i for i, _ in pairs]
            buses = [j for _, j in pairs]

            P0, Q0 = self.system.Line.get_flow_by_idx(idx=lines, bus=buses)
            self.area_P0[int_idx] = sum(P0)
            self.area_Q0[int_idx] = sum(Q0)

        for fr_area, pairs in self.interchange.items():
            if fr_area not in self.inter_flows.keys():
                self.inter_flows[fr_area] = dict()

            for to_area, vals in pairs.items():
                lines = [i for i, _ in vals]
                buses = [j for _, j in vals]
                P, Q = self.system.Line.get_flow_by_idx(idx=lines, bus=buses)
                self.inter_flows[fr_area][to_area] = [sum(P), sum(Q)]

        self.interchange_varout()

    def _varname_inter(self):
        if not self.n:
            return
        mpql = self.system.dae.m + 2 * self.system.Bus.n + \
            4 * self.system.Line.n

        # P_ic
        xy_idx = range(mpql, mpql + self.n_combination)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='P_ic',
            element_name=self.uname)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='P_{ic}',
            element_name=self.fname)

        # Q_ic
        xy_idx = range(mpql + self.n_combination,
                       mpql + 2 * self.n_combination)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Q_ic',
            element_name=self.uname)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='Q_{ic}',
            element_name=self.fname)

    def interchange_varout(self):
        if not self.n:
            return
        incidence = self.incidence - spdiag(self.incidence[0::self.n + 1])
        I, J, V = incidence.I, incidence.J, incidence.V
        P, Q = [], []
        for i, j, v in zip(I, J, V):
            if v == 0.:
                continue
            P.append(self.inter_flows[self.idx[i]][self.idx[j]][0])
            Q.append(self.inter_flows[self.idx[i]][self.idx[j]][1])

        self.P = matrix(P)
        self.Q = matrix(Q)
        self.inter_varout = matrix(P + Q)


class Area(Zone):
    """Area class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'Area'
        self._init()

    def setup(self):
        # TODO: account for >1 area/region/zone
        super().setup()


class Region(Zone):
    """Region class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'Region'
        self._params.extend(['Ptol', 'slack'])
        self._descr.update({
            'Ptol': 'Total transfer capacity',
            'slack': 'slack bus idx',
        })
        self._data.update({
            'Ptol': None,
            'slack': None,
        })
        self._powers.extend(['Ptol'])
        self._init()

    def setup(self):
        super().setup()
