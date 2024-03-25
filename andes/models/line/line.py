"""
AC transmission line and two-winding transformer line.
"""

import numpy as np

from andes.core import (ModelData, IdxParam, NumParam, DataParam,
                        Model, ExtAlgeb, ConstService)
from andes.shared import spmatrix


class LineData(ModelData):
    """
    Data for Line.
    """

    def __init__(self):
        super().__init__()

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")

        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           non_zero=True,
                           tex_name=r'S_n',
                           unit='MW',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           unit='Hz',
                           )
        self.Vn1 = NumParam(default=110.0,
                            info="AC voltage rating",
                            non_zero=True,
                            tex_name=r'V_{n1}',
                            unit='kV',
                            )
        self.Vn2 = NumParam(default=110.0,
                            info="rated voltage of bus2",
                            non_zero=True,
                            tex_name=r'V_{n2}',
                            unit='kV',
                            )

        self.r = NumParam(default=1e-8,
                          info="line resistance",
                          tex_name='r',
                          z=True,
                          unit='p.u.',
                          )
        self.x = NumParam(default=1e-8,
                          info="line reactance",
                          tex_name='x',
                          z=True,
                          unit='p.u.',
                          non_zero=True,
                          )
        self.b = NumParam(default=0.0,
                          info="shared shunt susceptance",
                          y=True,
                          unit='p.u.',
                          )
        self.g = NumParam(default=0.0,
                          info="shared shunt conductance",
                          y=True,
                          unit='p.u.',
                          )
        self.b1 = NumParam(default=0.0,
                           info="from-side susceptance",
                           y=True,
                           tex_name='b_1',
                           unit='p.u.',
                           )
        self.g1 = NumParam(default=0.0,
                           info="from-side conductance",
                           y=True,
                           tex_name='g_1',
                           unit='p.u.',
                           )
        self.b2 = NumParam(default=0.0,
                           info="to-side susceptance",
                           y=True,
                           tex_name='b_2',
                           unit='p.u.',
                           )
        self.g2 = NumParam(default=0.0,
                           info="to-side conductance",
                           y=True,
                           tex_name='g_2',
                           unit='p.u.',
                           )

        self.trans = NumParam(default=0,
                              info="transformer branch flag",
                              unit='bool',
                              )
        self.tap = NumParam(default=1.0,
                            info="transformer branch tap ratio",
                            tex_name='t_{ap}',
                            non_negative=True,
                            unit='float',
                            )
        self.phi = NumParam(default=0.0,
                            info="transformer branch phase shift in rad",
                            tex_name=r'\phi',
                            unit='radian',
                            )

        self.rate_a = NumParam(default=0.0,
                               info="long-term flow limit (placeholder)",
                               tex_name='R_{ATEA}',
                               unit='MVA',
                               )

        self.rate_b = NumParam(default=0.0,
                               info="short-term flow limit (placeholder)",
                               tex_name='R_{ATEB}',
                               unit='MVA',
                               )

        self.rate_c = NumParam(default=0.0,
                               info="emergency flow limit (placeholder)",
                               tex_name='R_{ATEC}',
                               unit='MVA',
                               )

        self.owner = IdxParam(model='Owner', info="owner code")

        self.xcoord = DataParam(info="x coordinates")
        self.ycoord = DataParam(info="y coordinates")


class Line(LineData, Model):
    """
    AC transmission line model.

    The model is also used for two-winding transformer. Transformers can set the
    tap ratio in ``tap`` and/or phase shift angle ``phi``.

    To reduce the number of variables, line injections are summed at bus
    equations and are not stored. Current injections are not computed.
    """

    def __init__(self, system=None, config=None):
        LineData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACLine'
        self.flags.pflow = True
        self.flags.tds = True

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1, tex_name='a_1',
                           info='phase angle of the from bus',
                           ename='Pij',
                           tex_ename='P_{ij}',
                           )
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2, tex_name='a_2',
                           info='phase angle of the to bus',
                           ename='Pji',
                           tex_ename='P_{ji}',
                           )
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1, tex_name='v_1',
                           info='voltage magnitude of the from bus',
                           ename='Qij',
                           tex_ename='Q_{ij}',
                           )
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2, tex_name='v_2',
                           info='voltage magnitude of the to bus',
                           ename='Qji',
                           tex_ename='Q_{ji}',
                           )

        self.gh = ConstService(tex_name='g_h')
        self.bh = ConstService(tex_name='b_h')
        self.gk = ConstService(tex_name='g_k')
        self.bk = ConstService(tex_name='b_k')

        self.yh = ConstService(tex_name='y_h', vtype=complex)
        self.yk = ConstService(tex_name='y_k', vtype=complex)
        self.yhk = ConstService(tex_name='y_{hk}', vtype=complex)

        self.ghk = ConstService(tex_name='g_{hk}')
        self.bhk = ConstService(tex_name='b_{hk}')

        self.itap = ConstService(tex_name='1/t_{ap}')
        self.itap2 = ConstService(tex_name='1/t_{ap}^2')

        self.gh.v_str = 'g1 + 0.5 * g'
        self.bh.v_str = 'b1 + 0.5 * b'
        self.gk.v_str = 'g2 + 0.5 * g'
        self.bk.v_str = 'b2 + 0.5 * b'

        self.yh.v_str = 'u * (gh + 1j * bh)'
        self.yk.v_str = 'u * (gk + 1j * bk)'
        self.yhk.v_str = 'u/((r+1e-8) + 1j*(x+1e-8))'

        self.ghk.v_str = 're(yhk)'
        self.bhk.v_str = 'im(yhk)'

        self.itap.v_str = '1/tap'
        self.itap2.v_str = '1/tap/tap'

        self.a1.e_str = 'u * (v1 ** 2 * (gh + ghk) * itap2  - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) + \
                                         bhk * sin(a1 - a2 - phi)) * itap)'

        self.v1.e_str = 'u * (-v1 ** 2 * (bh + bhk) * itap2 - \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) - \
                                         bhk * cos(a1 - a2 - phi)) * itap)'

        self.a2.e_str = 'u * (v2 ** 2 * (gh + ghk) - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) - \
                                         bhk * sin(a1 - a2 - phi)) * itap)'

        self.v2.e_str = 'u * (-v2 ** 2 * (bh + bhk) + \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) + \
                                         bhk * cos(a1 - a2 - phi)) * itap)'

    @property
    def istf(self):
        """
        Return an array of booleans to indicate whether or not the device is a
        transformer.

        A transformer has ``tap != 1`` or ``phi != 0``.
        """
        tap_istf = (self.tap.v) != 1
        phi_istf = (self.phi.v) != 0

        return np.logical_or(tap_istf, phi_istf)

    def get_tline_idx(self):
        """
        Return ``idx`` of transmission lines and exclude transformers.
        """

        return np.array(self.idx.v)[np.logical_not(self.istf)]

    def get_tf_idx(self):
        """
        Return ``idx`` of transformers and exclude lines.
        """

        return np.array(self.idx.v)[self.istf]

    def build_y(self):
        """
        Build bus admittance matrix. Store the matrix in ``self.Y``.

        Returns
        -------
        Y : spmatrix
            Bus admittance matrix.
        """

        nb = self.system.Bus.n

        y1 = self.u.v * (self.g1.v + self.b1.v * 1j)
        y2 = self.u.v * (self.g2.v + self.b2.v * 1j)
        y12 = self.u.v / (self.r.v + self.x.v * 1j)
        m = self.tap.v * np.exp(1j * self.phi.v)
        m2 = self.tap.v**2
        mconj = np.conj(m)

        # build self and mutual admittances into Y
        self.Y = spmatrix((y12 + y1 / m2), self.a1.a, self.a1.a, (nb, nb), 'z')
        self.Y -= spmatrix(y12 / mconj, self.a1.a, self.a2.a, (nb, nb), 'z')
        self.Y -= spmatrix(y12 / m, self.a2.a, self.a1.a, (nb, nb), 'z')
        self.Y += spmatrix(y12 + y2, self.a2.a, self.a2.a, (nb, nb), 'z')

        return self.Y

    def build_b(self, method='fdpf'):
        """
        build Bp and Bpp matrices for fast decoupled power flow and DC power flow.

        Results are saved to ``self.Bp`` and ``self.Bpp`` without return.
        """
        self.build_Bp(method)
        self.build_Bpp(method)

    def build_Bp(self, method='fdpf'):
        """
        Function for building B' matrix.

        The result is saved to ``self.Bp`` and returned

        Parameters
        ----------
        method : str
            Method for building B' matrix. Choose from 'fdpf', 'fdbx', 'dcpf'.
        Returns
        -------
        Bp : spmatrix
            B' matrix.

        """
        nb = self.system.Bus.n

        if method not in ("fdpf", "fdbx", "dcpf"):
            raise ValueError(f"Invalid method {method}; choose from 'fdpf', 'fdbx', 'dcpf'")

        # Build B prime matrix -- FDPF
        # `y1`` neglects line charging shunt, and g1 is usually 0 in HV lines
        # `y2`` neglects line charging shunt, and g2 is usually 0 in HV lines
        y1 = self.u.v * self.g1.v
        y2 = self.u.v * self.g2.v

        # `m` neglected tap ratio
        m = np.exp(self.phi.v * 1j)
        mconj = np.conj(m)
        m2 = np.ones(self.n)

        if method in ('fdxb', 'dcpf'):
            # neglect line resistance in Bp in XB method
            y12 = self.u.v / (self.x.v * 1j)
        else:
            y12 = self.u.v / (self.r.v + self.x.v * 1j)

        self.Bdc = spmatrix((y12 + y1) / m2, self.a1.a, self.a1.a, (nb, nb), 'z')
        self.Bdc -= spmatrix(y12 / mconj, self.a1.a, self.a2.a, (nb, nb), 'z')
        self.Bdc -= spmatrix(y12 / m, self.a2.a, self.a1.a, (nb, nb), 'z')
        self.Bdc += spmatrix(y12 + y2, self.a2.a, self.a2.a, (nb, nb), 'z')
        self.Bdc = self.Bdc.imag()

        for item in range(nb):
            if abs(self.Bdc[item, item]) == 0:
                self.Bdc[item, item] = 1e-6 + 0j

        return self.Bdc

    def build_Bpp(self, method='fdpf'):
        """
        Function for building B'' matrix.

        The result is saved to ``self.Bpp`` and returned

        Parameters
        ----------
        method : str
            Method for building B'' matrix. Choose from 'fdpf', 'fdbx', 'dcpf'.

        Returns
        -------
        Bpp : spmatrix
            B'' matrix.
        """

        nb = self.system.Bus.n

        if method not in ("fdpf", "fdbx", "dcpf"):
            raise ValueError(f"Invalid method {method}; choose from 'fdpf', 'fdbx', 'dcpf'")

        # Build B double prime matrix
        # y1 neglected line charging shunt, and g1 is usually 0 in HV lines
        # y2 neglected line charging shunt, and g2 is usually 0 in HV lines
        # m neglected phase shifter
        y1 = self.u.v * (self.g1.v + self.b1.v * 1j)
        y2 = self.u.v * (self.g2.v + self.b2.v * 1j)

        m = self.tap.v
        m2 = abs(m)**2

        if method in ('fdbx', 'fdpf', 'dcpf'):
            # neglect line resistance in Bpp in BX method
            y12 = self.u.v / (self.x.v * 1j)
        else:
            y12 = self.u.v / (self.r.v + self.x.v * 1j)

        self.Bpp = spmatrix((y12 + y1) / m2, self.a1.a, self.a1.a, (nb, nb), 'z')
        self.Bpp -= spmatrix(y12 / np.conj(m), self.a1.a, self.a2.a, (nb, nb), 'z')
        self.Bpp -= spmatrix(y12 / m, self.a2.a, self.a1.a, (nb, nb), 'z')
        self.Bpp += spmatrix(y12 + y2, self.a2.a, self.a2.a, (nb, nb), 'z')
        self.Bpp = self.Bpp.imag()

        for item in range(nb):
            if abs(self.Bpp[item, item]) == 0:
                self.Bpp[item, item] = 1e-6 + 0j

        return self.Bpp

    def build_Bdc(self):
        """
        The MATPOWER-flavor Bdc matrix for DC power flow. Saves results to `self.Bdc`.

        The method neglects line charging and line resistance. It retains tap ratio.

        Returns
        -------
        Bdc : spmatrix
            Bdc matrix.
        """

        nb = self.system.Bus.n

        y12 = self.u.v / (self.x.v * 1j)
        y12 = y12 / self.tap.v

        self.Bdc = spmatrix(y12, self.a1.a, self.a1.a, (nb, nb), 'z')
        self.Bdc -= spmatrix(y12, self.a1.a, self.a2.a, (nb, nb), 'z')
        self.Bdc -= spmatrix(y12, self.a2.a, self.a1.a, (nb, nb), 'z')
        self.Bdc += spmatrix(y12, self.a2.a, self.a2.a, (nb, nb), 'z')
        self.Bdc = self.Bdc.imag()

        for item in range(nb):
            if abs(self.Bdc[item, item]) == 0:
                self.Bdc[item, item] = 1e-6

        return self.Bdc
