"""
Third-order motor model.
"""

from andes.models.motor.motorbase import MotorBaseModel, MotorBaseData


class Motor3Model(MotorBaseModel):
    """
    Third-order induction motor model
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
    and use a ``Toggle`` to control the model.
    """

    def __init__(self, system, config):
        MotorBaseData.__init__(self)
        Motor3Model.__init__(self, system, config)
