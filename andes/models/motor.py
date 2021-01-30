"""
Induction machine models
"""

#  [ANDES] (C)2015-2021 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: induction.py
#  Last modified: 8/16/20, 7:25 PM

import logging
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, NumParam
from andes.core.var import Algeb, State, ExtAlgeb
from andes.core.service import ConstService

logger = logging.getLogger(__name__)


class MotorBaseData(ModelData):
    """Base parameters for induction machines
    """

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )

        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           tex_name='S_n',
                           )

        self.Vn = NumParam(default=110.0,
                           info="AC voltage rating",
                           tex_name='V_n',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )

        self.rs = NumParam(default=0.01,
                           info="rotor resistance",
                           z=True,
                           tex_name='r_s',
                           non_zero=True,
                           )

        self.xs = NumParam(default=0.15,
                           info="rotor reactance",
                           z=True,
                           tex_name='x_s',
                           non_zero=True,
                           )
        self.rr1 = NumParam(default=0.05,
                            info="1st cage rotor resistance",
                            z=True,
                            non_zero=True,
                            tex_name='r_{R1}',
                            )
        self.xr1 = NumParam(default=0.15,
                            info="1st cage rotor reactance",
                            z=True,
                            tex_name='x_{R1}',
                            non_zero=True,
                            )
        self.rr2 = NumParam(default=0.001,
                            info="2st cage rotor resistance",
                            z=True,
                            non_zero=True,
                            tex_name='r_{R2}',
                            )
        self.xr2 = NumParam(default=0.04,
                            info="2st cage rotor reactance",
                            z=True,
                            tex_name='x_{R2}',
                            non_zero=True,
                            )

        self.xm = NumParam(default=5.0,
                           info="magnetization reactance",
                           z=True,
                           tex_name='x_m',
                           non_zero=True,
                           )

        self.Hm = NumParam(default=3.0,
                           info='Inertia constant',
                           power=True,
                           tex_name='H_m',
                           unit='kWs/KVA',
                           )

        self.c1 = NumParam(default=0.1,
                           info='1st coeff. of Tm(w)',
                           tex_name='c_1',
                           )

        self.c2 = NumParam(default=0.02,
                           info='2nd coeff. of Tm(w)',
                           tex_name='c_2',
                           )

        self.c3 = NumParam(default=0.02,
                           info='3rd coeff. of Tm(w)',
                           tex_name='c_3',
                           )

        self.zb = NumParam(default=1.0,
                           info='Allow working as brake',
                           tex_name='z_b',
                           vrange=(0, 1),
                           )


class MotorBaseModel(Model):
    """Base model for motors
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.pflow = True
        self.flags.tds = True
        self.flags.tds_init = False
        self.group = 'Motor'

        # services
        self.wb = ConstService(v_str='2 * pi * fn',
                               tex_name=r'\omega_b',
                               )

        self.x0 = ConstService(v_str='xs + xm',
                               tex_name='x_0',
                               )

        self.x1 = ConstService(v_str='xs + xr1 * xm / (xr1 + xm)',
                               tex_name="x'",
                               )

        self.T10 = ConstService(v_str='(xr1 + xm) / (wb * rr1)',
                                tex_name="T'_0",
                                )

        self.M = ConstService(v_str='2 * Hm',
                              tex_name='M',
                              )

        self.aa = ConstService(v_str='c1 + c2 + c3',
                               tex_name=r'\alpha',
                               )

        self.bb = ConstService(v_str='-c2 - 2 * c3',
                               tex_name=r'\beta',
                               )

        # network algebraic variables
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='+p'
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='+q'
                          )

        self.vd = Algeb(info='d-axis voltage',
                        e_str='-u * v * sin(a) - vd',
                        tex_name=r'V_d',
                        )

        self.vq = Algeb(info='q-axis voltage',
                        e_str='u * v * cos(a) - vq',
                        tex_name=r'V_q',
                        )

        self.slip = State(tex_name=r"\sigma",
                          e_str='u * (tm - te)',
                          t_const=self.M,
                          diag_eps=True,
                          v_str='1.0 * u',
                          )

        self.p = Algeb(tex_name='P',
                       e_str='u * (vd * Id + vq * Iq) - p',
                       v_str='u * (vd * Id + vq * Iq)',
                       )

        self.q = Algeb(tex_name='Q',
                       e_str='u * (vq * Id - vd * Iq) - q',
                       v_str='u * (vq * Id - vd * Iq)',
                       )

        self.e1d = State(info='real part of 1st cage voltage',
                         tex_name="e'_d",
                         v_str='0.05 * u',
                         e_str='u * (wb*slip*e1q - (e1d + (x0 - x1) * Iq)/T10)',
                         diag_eps=True,
                         )

        self.e1q = State(info='imaginary part of 1st cage voltage',
                         tex_name="e'_q",
                         v_str='0.9 * u',
                         e_str='u * (-wb*slip*e1d - (e1q - (x0 - x1) * Id)/T10)',
                         diag_eps=True,
                         )

        self.Id = Algeb(tex_name='I_d',
                        diag_eps=True,
                        )

        self.Iq = Algeb(tex_name='I_q',
                        diag_eps=True,
                        )

        self.te = Algeb(tex_name=r'\tau_e',
                        )

        self.tm = Algeb(tex_name=r'\tau_m',
                        )

        self.tm.v_str = 'u * (aa + bb * slip + c2 * slip * slip)'
        self.tm.e_str = f'{self.tm.v_str} - tm'


class Motor5Model(MotorBaseModel):
    """
    Fifth-order Induction motor equations.
    """

    def __init__(self, system, config):
        MotorBaseModel.__init__(self, system, config)

        self.x2 = ConstService(v_str='xs + xr1*xr2*xm / (xr1*xr2 + xr1*xm + xr2*xm)',
                               tex_name="x''",
                               )
        self.T20 = ConstService(v_str='(xr2 + xr1*xm / (xr1 + xm) ) / (wb * rr2)',
                                tex_name="T''_0",
                                )

        self.e2d = State(info='real part of 2nd cage voltage',
                         e_str='u * '
                               '(-wb*slip*(e1q - e2q) + '
                               '(wb*slip*e1q - (e1d + (x0 - x1) * Iq)/T10) + '
                               '(e1d - e2d - (x1 - x2) * Iq)/T20)',
                         v_str='0.05 * u',
                         tex_name="e''_d",
                         diag_eps=True,
                         )

        self.e2q = State(info='imag part of 2nd cage voltage',
                         e_str='u * '
                               '(wb*slip*(e1d - e2d) + '
                               '(-wb*slip*e1d - (e1q - (x0 - x1) * Id)/T10) + '
                               '(e1q - e2q + (x1 - x2) * Id) / T20)',
                         v_str='0.9 * u',
                         tex_name="e''_q",
                         diag_eps=True,
                         )

        self.Id.e_str = 'u * (vd - e2d - rs * Id + x2 * Iq)'

        self.Id.v_str = '0.9 * u'

        self.Iq.e_str = 'u * (vq - e2q - rs * Iq - x2 * Id)'

        self.Iq.v_str = '0.1 * u'

        self.te.v_str = 'u * (e2d * Id + e2q * Iq)'

        self.te.e_str = f'{self.te.v_str} - te'


class Motor3Model(MotorBaseModel):
    """Third-order induction motor model
    """

    def __init__(self, system, config):
        MotorBaseModel.__init__(self, system, config)

        self.Id.e_str = 'u * (vd - e1d - rs * Id + x1 * Iq)'

        self.Id.v_str = '1'

        self.Iq.e_str = 'u * (vq - e1q - rs * Iq - x1 * Id)'

        self.te.v_str = 'u * (e1d * Id + e1q * Iq)'

        self.te.e_str = f'{self.te.v_str} - te'


class Motor3(MotorBaseData, Motor3Model):
    """
    Third-order induction motor model.

    See "Power System Modelling and Scripting" by F. Milano.

    To simulate motor startup, set the motor status ``u`` to ``0``
    and use a ``Toggler`` to control the model.
    """

    def __init__(self, system, config):
        MotorBaseData.__init__(self)
        Motor3Model.__init__(self, system, config)


class Motor5(MotorBaseData, Motor5Model):
    """
    Fifth-order induction motor model.

    See "Power System Modelling and Scripting" by F. Milano.

    To simulate motor startup, set the motor status ``u`` to ``0``
    and use a ``Toggler`` to control the model.
    """

    def __init__(self, system, config):
        MotorBaseData.__init__(self)
        Motor5Model.__init__(self, system, config)
