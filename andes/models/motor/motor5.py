"""
Fifth-order motor model.
"""

from andes.core import ConstService, State
from andes.models.motor.motorbase import MotorBaseModel, MotorBaseData


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


class Motor5(MotorBaseData, Motor5Model):
    """
    Fifth-order induction motor model.

    See "Power System Modelling and Scripting" by F. Milano.

    To simulate motor startup, set the motor status ``u`` to ``0``
    and use a ``Toggle`` to control the model.
    """

    def __init__(self, system, config):
        MotorBaseData.__init__(self)
        Motor5Model.__init__(self, system, config)
