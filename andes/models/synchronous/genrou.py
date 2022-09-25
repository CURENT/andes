"""
Round-rotor generator model.
"""

import logging

from andes.core import NumParam, ConstService, Algeb, LessThan, State
from andes.core.service import InitChecker, FlagValue
from andes.models.exciter import ExcQuadSat
from andes.models.synchronous.genbase import GENBaseData, GENBase, Flux0

logger = logging.getLogger(__name__)


class GENROUData(GENBaseData):
    """
    GENROU data.
    """

    def __init__(self):
        super().__init__()
        self.xd = NumParam(default=1.9, info='d-axis synchronous reactance',
                           tex_name=r'x_d', z=True)
        self.xq = NumParam(default=1.7,
                           info="q-axis synchronous reactance",
                           tex_name='x_q',
                           z=True,
                           )
        self.xd2 = NumParam(default=0.3, info='d-axis sub-transient reactance',
                            tex_name=r"{x''_d}", z=True)

        self.xq1 = NumParam(default=0.5, info='q-axis transient reactance',
                            tex_name=r"{x'_q}", z=True)
        self.xq2 = NumParam(default=0.3, info='q-axis sub-transient reactance',
                            tex_name=r"{x''_q}", z=True)

        self.Td10 = NumParam(default=8.0, info='d-axis transient time constant',
                             tex_name=r"{T'_{d0}}")
        self.Td20 = NumParam(default=0.04, info='d-axis sub-transient time constant',
                             tex_name=r"{T''_{d0}}")
        self.Tq10 = NumParam(default=0.8, info='q-axis transient time constant',
                             tex_name=r"{T'_{q0}}")
        self.Tq20 = NumParam(default=0.02, info='q-axis sub-transient time constant',
                             tex_name=r"{T''_{q0}}")


class GENROUModel:
    def __init__(self):
        # parameter checking for `xl`
        self._xlc = InitChecker(u=self.xl,
                                info='(xl <= xd2)',
                                upper=self.xd2
                                )

        self.gd1 = ConstService(v_str='(xd2 - xl) / (xd1 - xl)',
                                tex_name=r"\gamma_{d1}")
        self.gq1 = ConstService(v_str='(xq2 - xl) / (xq1 - xl)',
                                tex_name=r"\gamma_{q1}")
        self.gd2 = ConstService(v_str='(xd1 - xd2) / (xd1 - xl) ** 2',
                                tex_name=r"\gamma_{d2}")
        self.gq2 = ConstService(v_str='(xq1 - xq2) / (xq1 - xl) ** 2',
                                tex_name=r"\gamma_{q2}")
        self.gqd = ConstService(v_str='(xq - xl) / (xd - xl)',
                                tex_name=r"\gamma_{qd}")

        # correct S12 to 1.0 if is zero
        self._fS12 = FlagValue(self.S12, value=0)
        self._S12 = ConstService(v_str='S12 + (1-_fS12)',
                                 info='Corrected S12',
                                 tex_name='S_{1.2}'
                                 )
        # Saturation services
        # Note:
        # To disable saturation, set `S10 = 0` and `S12 = 1`, so that SAT_B = 0.
        self.SAT = ExcQuadSat(1.0, self.S10, 1.2, self.S12, tex_name='S_{AT}')

        # Initialization reference: OpenIPSL at
        #   https://github.com/OpenIPSL/OpenIPSL/blob/master/OpenIPSL/Electrical/Machines/PSSE/GENROU.mo

        # internal voltage and rotor angle calculation
        self._V = ConstService(v_str='v * exp(1j * a)', tex_name='V_c', info='complex bus voltage',
                               vtype=complex)
        self._S = ConstService(v_str='p0 - 1j * q0', tex_name='S', info='complex terminal power',
                               vtype=complex)
        self._Zs = ConstService(v_str='ra + 1j * xd2', tex_name='Z_s', info='equivalent impedance',
                                vtype=complex)
        self._It = ConstService(v_str='_S / conj(_V)', tex_name='I_t', info='complex terminal current',
                                vtype=complex)
        self._Is = ConstService(tex_name='I_s', v_str='_It + _V / _Zs', info='equivalent current source',
                                vtype=complex)

        self.psi20 = ConstService(tex_name=r"{\psi''_0}", v_str='_Is * _Zs',
                                  info='sub-transient flux linkage in stator reference',
                                  vtype=complex)
        self.psi20_arg = ConstService(tex_name=r"\theta_{\psi''0}", v_str='arg(psi20)')
        self.psi20_abs = ConstService(tex_name=r"|{\psi''_0}|", v_str='abs(psi20)')
        self._It_arg = ConstService(tex_name=r"\theta_{It0}", v_str='arg(_It)')
        self._psi20_It_arg = ConstService(tex_name=r"\theta_{\psi a It}",
                                          v_str='psi20_arg - _It_arg')

        self.Se0 = ConstService(tex_name=r"S_{e0}",
                                v_str='Indicator(psi20_abs>=SAT_A) * (psi20_abs - SAT_A) ** 2 * SAT_B / psi20_abs')

        self._a = ConstService(tex_name=r"{a'}", v_str='psi20_abs * (1 + Se0*gqd)')
        self._b = ConstService(tex_name=r"{b'}", v_str='abs(_It) * (xq2 - xq)')  # xd2 == xq2

        self.delta0 = ConstService(tex_name=r'\delta_0',
                                   v_str='atan(_b * cos(_psi20_It_arg) / (_b * sin(_psi20_It_arg) - _a)) + '
                                         'psi20_arg')
        self._Tdq = ConstService(tex_name=r"T_{dq}",
                                 v_str='cos(delta0) - 1j * sin(delta0)',
                                 vtype=complex)
        self.psi20_dq = ConstService(tex_name=r"{\psi''_{0,dq}}",
                                     v_str='psi20 * _Tdq',
                                     vtype=complex)
        self.It_dq = ConstService(tex_name=r"I_{t,dq}",
                                  v_str='conj(_It * _Tdq)',
                                  vtype=complex)

        self.psi2d0 = ConstService(tex_name=r"\psi_{ad0}",
                                   v_str='re(psi20_dq)')
        self.psi2q0 = ConstService(tex_name=r"\psi_{aq0}",
                                   v_str='-im(psi20_dq)')

        self.Id0 = ConstService(v_str='im(It_dq)', tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='re(It_dq)', tex_name=r'I_{q0}')

        self.vd0 = ConstService(v_str='psi2q0 + xq2*Iq0 - ra * Id0', tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='psi2d0 - xd2*Id0 - ra*Iq0', tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')

        # `vf0` is also equal to `vq + xd*Id +ra*Iq + Se*psi2d` from phasor diagram
        self.vf0 = ConstService(tex_name=r'v_{f0}', v_str='(Se0 + 1)*psi2d0 + (xd - xd2) * Id0')
        self.psid0 = ConstService(tex_name=r"\psi_{d0}",
                                  v_str='u * (ra * Iq0) + vq0')
        self.psiq0 = ConstService(tex_name=r"\psi_{q0}",
                                  v_str='-u * (ra * Id0) - vd0')

        # initialization of internal voltage and delta
        self.e1q0 = ConstService(tex_name="{e'_{q0}}",
                                 v_str='Id0*(-xd + xd1) - Se0*psi2d0 + vf0')

        self.e1d0 = ConstService(tex_name="{e'_{d0}}",
                                 v_str='Iq0*(xq - xq1) - Se0*gqd*psi2q0')

        self.e2d0 = ConstService(tex_name="{e''_{d0}}",
                                 v_str='Id0*(xl - xd) - Se0*psi2d0 + vf0')
        self.e2q0 = ConstService(tex_name="{e''_{q0}}",
                                 v_str='-Iq0*(xl - xq) - Se0*gqd*psi2q0')

        # begin variables and equations
        self.psi2q = Algeb(tex_name=r"\psi_{aq}", info='q-axis air gap flux',
                           v_str='u * psi2q0',
                           e_str='gq1*e1d + (1-gq1)*e2q - psi2q',
                           )

        self.psi2d = Algeb(tex_name=r"\psi_{ad}", info='d-axis air gap flux',
                           v_str='u * psi2d0',
                           e_str='gd1*e1q + gd2*(xd1-xl)*e2d - psi2d')

        self.psi2 = Algeb(tex_name=r"\psi_a", info='air gap flux magnitude',
                          v_str='u * abs(psi20_dq)',
                          e_str='psi2d **2 + psi2q ** 2 - psi2 ** 2',
                          diag_eps=True,
                          )

        # `LT` is a reserved keyword for SymPy
        self.SL = LessThan(u=self.psi2, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"S_e(|\psi_{a}|)", info='saturation output',
                        v_str='u * Se0',
                        e_str='SL_z0 * (psi2 - SAT_A) ** 2 * SAT_B - psi2 * Se',
                        diag_eps=True,
                        )

        # separated `XadIfd` from `e1q` using \dot(e1q) = (vf - XadIfd) / Td10
        self.XadIfd.e_str = 'u * (e1q + (xd-xd1) * (gd1*Id - gd2*e2d + gd2*e1q) + Se*psi2d) - XadIfd'

        # `XadI1q` can also be given in `(xq-xq1)*gq2*(e1d-e2q+(xq1-xl)*Iq) + e1d - Iq*(xq-xq1) + Se*psi2q*gqd`
        self.XaqI1q =\
            Algeb(tex_name='X_{aq}I_{1q}',
                  info='q-axis reaction',
                  unit='p.u (kV)',
                  v_str='0',
                  e_str='e1d + (xq-xq1) * (gq2*e1d - gq2*e2q - gq1*Iq) + Se*psi2q*gqd - XaqI1q'
                  )

        self.e1q = State(info='q-axis transient voltage',
                         tex_name=r"{e'_q}",
                         v_str='u * e1q0',
                         e_str='(-XadIfd + vf)',
                         t_const=self.Td10,
                         )

        self.e1d = State(info='d-axis transient voltage',
                         tex_name=r"{e'_d}",
                         v_str='u * e1d0',
                         e_str='-XaqI1q',
                         t_const=self.Tq10,
                         )

        self.e2d = State(info='d-axis sub-transient voltage',
                         tex_name=r"{e''_d}",
                         v_str='u * e2d0',
                         e_str='(-e2d + e1q - (xd1 - xl) * Id)',
                         t_const=self.Td20,
                         )

        self.e2q = State(info='q-axis sub-transient voltage',
                         tex_name=r"{e''_q}",
                         v_str='u * e2q0',
                         e_str='(-e2q + e1d + (xq1 - xl) * Iq)',
                         t_const=self.Tq20,
                         )

        self.Iq.e_str += '+ xq2*Iq + psi2q'

        self.Id.e_str += '+ xd2*Id - psi2d'


class GENROU(GENROUData, GENBase, GENROUModel, Flux0):
    """
    Round rotor generator with quadratic saturation.

    Notes
    -----
    Parameters:

    - ``xd2`` and ``xq2`` must be equal to pass initialization.

    """

    def __init__(self, system, config):
        GENROUData.__init__(self)
        GENBase.__init__(self, system, config)
        Flux0.__init__(self)
        GENROUModel.__init__(self)
