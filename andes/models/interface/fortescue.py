"""
Module for Fortescue's symmetric component calculation.
"""

from andes.core.model.modeldata import ModelData
from andes.core.model.model import Model
from andes.core.param import IdxParam, NumParam
from andes.core.service import ConstService
from andes.core.var import Algeb, ExtAlgeb


class FortescueData(ModelData):
    """
    Fortescue data.
    """

    def __init__(self):
        super().__init__()

        self.bus = IdxParam(model='Bus', info="bus idx for the single-phase equivalent", mandatory=True)
        self.busa = IdxParam(model='Bus', info="bus idx for phase a", mandatory=True)
        self.busb = IdxParam(model='Bus', info="bus idx for phase b", mandatory=True)
        self.busc = IdxParam(model='Bus', info="bus idx for phase c", mandatory=True)

        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           non_zero=True,
                           tex_name=r'S_n',
                           unit='MW',
                           )

        self.r = NumParam(default=1e-3,
                          info="resistance",
                          tex_name='r',
                          z=True,
                          unit='p.u.',
                          )
        self.x = NumParam(default=0.025,
                          info="short-circuit reactance",
                          tex_name='x',
                          z=True,
                          unit='p.u.',
                          non_zero=True,
                          )
        self.g = NumParam(default=0.0,
                          info="iron loss",
                          y=True,
                          unit='p.u.',
                          )
        self.b = NumParam(default=0.005,
                          info="magnetizing susceptance",
                          y=True,
                          unit='p.u.',
                          )


class FortescueModel(Model):
    """
    Model implementation.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)

        self.group = 'Interface'
        self.flags.pflow = True
        self.flags.tds = True

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta_1',
                          info='phase angle of single-phase eq. bus',
                          ename='P1',
                          tex_ename='P_1',
                          )
        self.aa = ExtAlgeb(model='Bus', src='a', indexer=self.busa, tex_name=r'\theta_a',
                           info='phase angle of bus for phase a',
                           ename='Pa',
                           tex_ename='P_{a}',
                           diag_eps=True,
                           )
        self.ab = ExtAlgeb(model='Bus', src='a', indexer=self.busb, tex_name=r'\theta_b',
                           info='phase angle of bus for phase b',
                           ename='Pb',
                           tex_ename='P_{b}',
                           diag_eps=True,
                           )
        self.ac = ExtAlgeb(model='Bus', src='a', indexer=self.busc, tex_name=r'\theta_c',
                           info='phase angle of bus for phase c',
                           ename='Pc',
                           tex_ename='P_{c}',
                           diag_eps=True,
                           )

        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V_1',
                          info='voltage of single-phase eq. bus',
                          ename='Q1',
                          tex_ename='Q_1',
                          )
        self.va = ExtAlgeb(model='Bus', src='v', indexer=self.busa, tex_name='V_a',
                           info='voltage of bus for phase a',
                           ename='Qa',
                           tex_ename='Q_{a}',
                           diag_eps=True,
                           )
        self.vb = ExtAlgeb(model='Bus', src='v', indexer=self.busb, tex_name='V_b',
                           info='voltage of bus for phase b',
                           ename='Qb',
                           tex_ename='Q_{b}',
                           diag_eps=True,
                           )
        self.vc = ExtAlgeb(model='Bus', src='v', indexer=self.busc, tex_name='V_c',
                           info='voltage of bus for phase c',
                           ename='Qc',
                           tex_ename='Q_{c}',
                           diag_eps=True,
                           )

        # gh = g, bh = b, yh = (gh + 1j bh), yk = 0, yhk = u/(r+1j*x)
        self.yhk = ConstService(tex_name='y_{hk}', vtype=complex)
        self.ghk = ConstService(tex_name='g_{hk}')
        self.bhk = ConstService(tex_name='b_{hk}')

        self.yhk.v_str = 'u/(r + 1j*x)'
        self.ghk.v_str = 're(yhk)'
        self.bhk.v_str = 'im(yhk)'

        self.d120 = ConstService(v_str='2/3*pi', tex_name='120^o', info='120 degrees')

        # --- positive sequence: magnitude and angle
        self.vp = Algeb(v_str='(va + vb + vc) / 3',
                        e_str='sqrt((vb * cos(ab + d120 - aa) + \
                                     vc * cos(ac - d120 - aa) + va) ** 2 + \
                                    (vb * sin(ab + d120 - aa) + \
                                     vc * sin(ac - d120 - aa)) ** 2) / 3 - vp',
                        info='positive sequence voltage magnitude',
                        tex_name='V_{p}',
                        )

        self.ap = Algeb(v_str='aa + ab + ac',
                        e_str='atan2((vb * sin(ab + d120 - aa) + \
                                      vc * sin(ac - d120 - aa)), \
                                     (vb * cos(ab + d120 - aa) + \
                                      vc * cos(ac - d120 - aa) + va)) + aa - ap',
                        info='positive sequence voltage phase',
                        tex_name=r'\theta_{p}',
                        )

        # --- negative sequence: d- and q-axis

        self.vnd = Algeb(v_str='0.0',
                         e_str='va * cos(aa) + vb * cos(ab - d120) + vc * cos(ac + d120) - vnd',
                         info='negative sequence voltage on d-axis (cos)',
                         tex_name='V_{nd}',
                         )

        self.vnq = Algeb(v_str='0.0',
                         e_str='va * sin(aa) + vb * sin(ab - d120) + vc * sin(ac + d120) - vnq',
                         info='negative sequence voltage on q-axis (sin)',
                         tex_name='V_{nq}',
                         )

        # --- zero sequence: d- and q-axis voltages
        self.vzd = Algeb(v_str='0.0',
                         e_str='va * cos(aa) + vb * cos(ab) + vc * cos(ac) - vzd',
                         info='zero sequence voltage on d-axis (cos)',
                         tex_name='V_{zd}',
                         )

        self.vzq = Algeb(v_str='0.0',
                         e_str='va * sin(aa) + vb * sin(ab) + vc * sin(ac) - vzq',
                         info='zero sequence voltage on q-axis (sin)',
                         tex_name='V_{zq}',
                         )

        # injection on the primary side

        self.a.e_str = 'u * (v ** 2 * (g + ghk) - v * vp * (ghk * cos(a - ap) + bhk * sin(a - ap)))'

        self.v.e_str = 'u * (-v ** 2 * (b + bhk) - v * vp * (ghk * sin(a - ap) - bhk * cos(a - ap)))'

        # injections on the secondary side

        p_phase = 'u/3*({v_phase}**2 *(g + ghk) - ' \
                  'v*{v_phase}*(ghk * cos(a-({a_phase})) - bhk * sin(a-({a_phase}))))'

        q_phase = 'u/3*(-{v_phase}**2 *(b + bhk) + ' \
                  'v *{v_phase}* (ghk * sin(a - ({a_phase})) + bhk * cos(a - ({a_phase}))))'

        self.aa.e_str = p_phase.format(v_phase='va', a_phase='aa')
        self.ab.e_str = p_phase.format(v_phase='vb', a_phase='ab + d120')
        self.ac.e_str = p_phase.format(v_phase='vc', a_phase='ac - d120')

        self.va.e_str = q_phase.format(v_phase='va', a_phase='aa')
        self.vb.e_str = q_phase.format(v_phase='vb', a_phase='ab + d120')
        self.vc.e_str = q_phase.format(v_phase='vc', a_phase='ac - d120')


class Fortescue(FortescueData, FortescueModel):
    """
    Fortescue's symmetric component interface.

    This model interfaces a positive-sequence, single-phase-equivalent bus with
    three buses representing three phases. It is effectively a transformer with
    one terminal on the primary side and three on the secondary. Only the
    positive sequence component on the secondary winding is used for simulation.

    The positive-sequence voltage magnitude and angle of the secondary winding
    are named ``vp`` and ``ap``.

    The negative and zero sequence variables given in the d- and q-axis due the
    angle being undefined when the voltage is zero. The negative sequence
    voltages are ``vnd`` and ``vnq`` for the d- anx q-axis, respectively.
    Likewise, the zero-sequence voltages are ``vzd`` and ``vzq``.

    """

    def __init__(self, system, config):
        FortescueData.__init__(self)
        FortescueModel.__init__(self, system, config)
