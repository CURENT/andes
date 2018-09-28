from cvxopt import matrix, spmatrix, sparse, spdiag  # NOQA
from cvxopt import mul, div  # NOQA

from .base import ModelBase

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA

from ..utils.math import polar, conj
from ..consts import deg2rad
import logging
logger = logging.getLogger(__name__)


class Line(ModelBase):
    """AC transmission line lumped model"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Series'
        self._name = 'Line'
        self._data.update({
            'r': 0.0,
            'x': 1e-6,
            'b': 0.0,
            'g': 0.0,
            'b1': 0.0,
            'g1': 0.0,
            'b2': 0.0,
            'g2': 0.0,
            'bus1': None,
            'bus2': None,
            'Vn2': 110.0,
            'xcoord': None,
            'ycoord': None,
            'trasf': False,
            'tap': 1.0,
            'phi': 0,
            'fn': 60,
            'owner': 0,
            'rate_a': 0,
        })
        self._units.update({
            'r': 'pu',
            'x': 'pu',
            'b': 'pu',
            'g': 'pu',
            'b1': 'pu',
            'g1': 'pu',
            'b2': 'pu',
            'g2': 'pu',
            'bus1': 'na',
            'bus2': 'na',
            'Vn2': 'kV',
            'xcoord': 'deg',
            'ycoord': 'deg',
            'trasf': 'na',
            'tap': 'na',
            'phi': 'deg',
            'fn': 'Hz',
            'owner': 'na',
            'rate_a': 'pu',
        })
        self._descr.update({
            'r': 'connection line resistance',
            'x': 'connection line reactance',
            'g': 'shared shunt conductance',
            'b': 'shared shunt susceptance',
            'g1': 'from-side conductance',
            'b1': 'from-side susceptance',
            'g2': 'to-side conductance',
            'b2': 'to-side susceptance',
            'bus1': 'idx of from bus',
            'bus2': 'idx of to bus',
            'Vn2': 'rated voltage of bus2',
            'xcoord': 'x coordinates',
            'ycoord': 'y coordinates',
            'trasf': 'transformer branch flag',
            'tap': 'transformer branch tap ratio',
            'phi': 'transformer branch phase shift in rad',
            'fn': 'rated frequency',
            'owner': 'owner code',
            'rate_a': 'phase a power flow limit',
        })
        self._params.extend([
            'r', 'x', 'b', 'g', 'b1', 'g1', 'b2', 'g2', 'tap', 'phi', 'fn',
            'rate_a'
        ])
        self._service.extend([
            'a', 'v', 'a1', 'a2', 'S1', 'S2', 'nb', 'gy_store', 'y1', 'y2',
            'y12', 'm', 'm2', 'mconj'
        ])
        self.calls.update({
            'gcall': True,
            'gycall': True,
            'init0': True,
            'pflow': True,
            'series': True,
            'flows': True
        })

        self._ac = {'bus1': ['a1', 'v1'], 'bus2': ['a2', 'v2']}

        self._config['is_series'] = True

        self.rebuild = True
        self.Y = []
        self.C = []
        self.Bp = []
        self.Bpp = []
        self._init()

    def setup(self):
        self._param_to_matrix()

        self.nb = int(self.system.Bus.n)
        self.system.config.nseries += self.n

        self.r += 1e-10
        self.b += 1e-10

        self.g1 += 0.5 * self.g
        self.b1 += 0.5 * self.b
        self.g2 += 0.5 * self.g
        self.b2 += 0.5 * self.b

    def build_y(self):
        """Build transmission line admittance matrix into self.Y"""
        if not self.n:
            return
        self.y1 = mul(self.u, self.g1 + self.b1 * 1j)
        self.y2 = mul(self.u, self.g2 + self.b2 * 1j)
        self.y12 = div(self.u, self.r + self.x * 1j)
        self.m = polar(self.tap, self.phi * deg2rad)
        self.m2 = abs(self.m)**2
        self.mconj = conj(self.m)

        # build self and mutual admittances into Y
        self.Y = spmatrix(
            div(self.y12 + self.y1, self.m2), self.a1, self.a1,
            (self.nb, self.nb), 'z')
        self.Y -= spmatrix(
            div(self.y12, self.mconj), self.a1, self.a2, (self.nb, self.nb),
            'z')
        self.Y -= spmatrix(
            div(self.y12, self.m), self.a2, self.a1, (self.nb, self.nb), 'z')
        self.Y += spmatrix(self.y12 + self.y2, self.a2, self.a2,
                           (self.nb, self.nb), 'z')

        # avoid singularity
        # for item in range(self.nb):
        #     if abs(self.Y[item, item]) == 0:
        #         self.Y[item, item] = 1e-6 + 0j

    def build_b(self):
        """build Bp and Bpp for fast decoupled method"""
        if not self.n:
            return
        method = self.system.pflow.config.method.lower()

        # Build B prime matrix
        y1 = mul(
            self.u, self.g1
        )  # y1 neglects line charging shunt, and g1 is usually 0 in HV lines
        y2 = mul(
            self.u, self.g2
        )  # y2 neglects line charging shunt, and g2 is usually 0 in HV lines
        m = polar(1.0, self.phi * deg2rad)  # neglected tap ratio
        self.mconj = conj(m)
        m2 = matrix(1.0, (self.n, 1), 'z')
        if method == 'fdxb':
            # neglect line resistance in Bp in XB method
            y12 = div(self.u, self.x * 1j)
        else:
            y12 = div(self.u, self.r + self.x * 1j)
        self.Bp = spmatrix(
            div(y12 + y1, m2), self.a1, self.a1, (self.nb, self.nb), 'z')
        self.Bp -= spmatrix(
            div(y12, conj(m)), self.a1, self.a2, (self.nb, self.nb), 'z')
        self.Bp -= spmatrix(
            div(y12, m), self.a2, self.a1, (self.nb, self.nb), 'z')
        self.Bp += spmatrix(y12 + y2, self.a2, self.a2, (self.nb, self.nb),
                            'z')
        self.Bp = self.Bp.imag()

        # Build B double prime matrix
        y1 = mul(
            self.u, self.g1 + self.b1 * 1j
        )  # y1 neglected line charging shunt, and g1 is usually 0 in HV lines
        y2 = mul(
            self.u, self.g2 + self.b2 * 1j
        )  # y2 neglected line charging shunt, and g2 is usually 0 in HV lines
        m = self.tap + 0j  # neglected phase shifter
        m2 = abs(m)**2 + 0j

        if method in ('fdbx', 'fdpf'):
            # neglect line resistance in Bpp in BX method
            y12 = div(self.u, self.x * 1j)
        else:
            y12 = div(self.u, self.r + self.x * 1j)
        self.Bpp = spmatrix(
            div(y12 + y1, m2), self.a1, self.a1, (self.nb, self.nb), 'z')
        self.Bpp -= spmatrix(
            div(y12, conj(m)), self.a1, self.a2, (self.nb, self.nb), 'z')
        self.Bpp -= spmatrix(
            div(y12, m), self.a2, self.a1, (self.nb, self.nb), 'z')
        self.Bpp += spmatrix(y12 + y2, self.a2, self.a2, (self.nb, self.nb),
                             'z')
        self.Bpp = self.Bpp.imag()

        for item in range(self.nb):
            if abs(self.Bp[item, item]) == 0:
                self.Bp[item, item] = 1e-6 + 0j
            if abs(self.Bpp[item, item]) == 0:
                self.Bpp[item, item] = 1e-6 + 0j

    def incidence(self):
        """Build incidence matrix into self.C"""
        self.C = \
            spmatrix(self.u, range(self.n), self.a1, (self.n, self.nb), 'd') -\
            spmatrix(self.u, range(self.n), self.a2, (self.n, self.nb), 'd')

    def connectivity(self, bus):
        """check connectivity of network using Goderya's algorithm"""
        if not self.n:
            return
        n = self.nb
        fr = self.a1
        to = self.a2
        os = [0] * self.n

        # find islanded buses
        diag = list(
            matrix(
                spmatrix(self.u, to, os, (n, 1), 'd') +
                spmatrix(self.u, fr, os, (n, 1), 'd')))
        nib = bus.n_islanded_buses = diag.count(0)
        bus.islanded_buses = []
        for idx in range(n):
            if diag[idx] == 0:
                bus.islanded_buses.append(idx)

        # find islanded areas
        temp = spmatrix(
            list(self.u) * 4, fr + to + fr + to, to + fr + fr + to, (n, n),
            'd')
        cons = temp[0, :]
        nelm = len(cons.J)
        conn = spmatrix([], [], [], (1, n), 'd')
        bus.island_sets = []
        idx = islands = 0
        enum = 0

        while 1:
            while 1:
                cons = cons * temp
                new_nelm = len(cons.J)
                if new_nelm == nelm:
                    break
                nelm = new_nelm
            cons = sparse(cons)  # remove zero values
            if len(cons.J) == n:  # all buses are interconnected
                return
            bus.island_sets.append(list(cons.J))
            conn += cons
            islands += 1
            nconn = len(conn.J)
            if nconn >= (n - nib):
                bus.island_sets = [i for i in bus.island_sets if i != []]
                break

            for element in conn.J[idx:]:
                if not diag[idx]:
                    enum += 1  # skip islanded buses
                if element <= enum:
                    idx += 1
                    enum += 1
                else:
                    break

            cons = temp[enum, :]

    def init0(self, dae):
        self.copy_data_ext('Bus', 'a', dest='a', idx=None, astype=list)
        self.copy_data_ext('Bus', 'v', dest='v', idx=None, astype=list)

        method = self.system.pflow.config.method.lower()
        self.build_y()
        self.incidence()
        if method in ('fdpf', 'fdbx', 'fdxb'):
            self.build_b()

    def gcall(self, dae):
        if self.rebuild:
            self.build_y()
            self.rebuild = False
        vc = polar(dae.y[self.v], dae.y[self.a])
        Ic = self.Y * vc
        S = mul(vc, conj(Ic))
        dae.g[self.a] += S.real()
        dae.g[self.v] += S.imag()

    def gycall(self, dae):
        gy = self.build_gy(dae)
        dae.add_jac(Gy, gy.V, gy.I, gy.J)

    def build_gy(self, dae):
        """Build line Jacobian matrix"""
        if not self.n:
            idx = range(dae.m)
            dae.set_jac(Gy, 1e-6, idx, idx)
            return

        Vn = polar(1.0, dae.y[self.a])
        Vc = mul(dae.y[self.v], Vn)
        Ic = self.Y * Vc

        diagVn = spdiag(Vn)
        diagVc = spdiag(Vc)
        diagIc = spdiag(Ic)

        dS = self.Y * diagVn
        dS = diagVc * conj(dS)
        dS += conj(diagIc) * diagVn

        dR = diagIc
        dR -= self.Y * diagVc
        dR = diagVc.H.T * dR

        self.gy_store = sparse([[dR.imag(), dR.real()], [dS.real(),
                                                         dS.imag()]])

        return self.gy_store

    def seriesflow(self, dae):
        """
        Compute the flow through the line after solving PF.

        Compute terminal injections, line losses
        """

        Vm = dae.y[self.v]
        Va = dae.y[self.a]
        V1 = polar(Vm[self.a1], Va[self.a1])
        V2 = polar(Vm[self.a2], Va[self.a2])

        I1 = mul(V1, div(self.y12 + self.y1, self.m2)) - mul(
            V2, div(self.y12, self.mconj))
        I2 = mul(V2, self.y12 + self.y2) - mul(V1, div(self.y12, self.m))
        self.S1 = mul(V1, conj(I1))
        self.S2 = mul(V2, conj(I2))
        self.P1 = self.S1.real()
        self.P2 = self.S2.real()
        self.Q1 = self.S1.imag()
        self.Q2 = self.S2.imag()

        self.chg1 = mul(self.g1 + 1j * self.b1, div(V1**2, self.m2))
        self.chg2 = mul(self.g2 + 1j * self.b2, V2**2)

        self.Pchg1 = self.chg1.real()
        self.Pchg2 = self.chg2.real()

        self.Qchg1 = self.chg1.imag()
        self.Qchg2 = self.chg2.imag()

        self._line_flows = matrix([self.P1, self.P2, self.Q1, self.Q2])

    def switch(self, idx, u):
        """switch the status of Line idx"""
        self.u[self.uid[idx]] = u
        self.rebuild = True
        self.system.dae.factorize = True
        logger.debug('<Line> Status switch to {} on idx {}.'.format(u, idx))

    def build_name_from_bus(self):
        """Rebuild line names from bus names"""
        pass

    def _varname_flow(self):
        """Build variable names for Pij, Pji, Qij, Qji, Sij, Sji"""
        if not self.n:
            return

        mpq = self.system.dae.m + 2 * self.system.Bus.n
        nl = self.n

        # Pij
        xy_idx = range(mpq, mpq + nl)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Pij',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='P_{ij}',
            element_name=self.name)

        # Pji
        xy_idx = range(mpq + nl, mpq + 2 * nl)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Pji',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='P_{ji}',
            element_name=self.name)

        # Qij
        xy_idx = range(mpq + 2 * nl, mpq + 3 * nl)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Qij',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='Q_{ij}',
            element_name=self.name)

        # Qji
        xy_idx = range(mpq + 3 * nl, mpq + 4 * nl)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Qji',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='Q_{ji}',
            element_name=self.name)

        # # Sij
        # xy_idx = range(mpq + 4 * nl, mpq + 5 * nl)
        # self.system.varname.append(
        #     listname='unamey',
        #     xy_idx=xy_idx,
        #     var_name='Sij',
        #     element_name=self.name)
        # self.system.varname.append(
        #     listname='fnamey',
        #     xy_idx=xy_idx,
        #     var_name='S_{ij}',
        #     element_name=self.name)
        #
        # # Qji
        # xy_idx = range(mpq + 5 * nl, mpq + 6 * nl)
        # self.system.varname.append(
        #     listname='unamey',
        #     xy_idx=xy_idx,
        #     var_name='Sji',
        #     element_name=self.name)
        # self.system.varname.append(
        #     listname='fnamey',
        #     xy_idx=xy_idx,
        #     var_name='S_{ji}',
        #     element_name=self.name)

    def get_flow_by_idx(self, idx, bus):
        """Return seriesflow based on the external idx on the `bus` side"""
        P, Q = [], []
        if type(idx) is not list:
            idx = [idx]
        if type(bus) is not list:
            bus = [bus]

        for line_idx, bus_idx in zip(idx, bus):
            line_int = self.uid[line_idx]
            if bus_idx == self.bus1[line_int]:
                P.append(self.P1[line_int])
                Q.append(self.Q1[line_int])
            elif bus_idx == self.bus2[line_int]:
                P.append(self.P2[line_int])
                Q.append(self.Q2[line_int])
        return matrix(P), matrix(Q)
