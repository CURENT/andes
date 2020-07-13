from andes.core.model import Model, ModelData
from andes.core.param import NumParam


class REGCAU1Data(ModelData):
    """
    REGC_A model data.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.Tg = NumParam(default=0.1, tex_name='T_g',
                           info='converter time const.', unit='s',
                           )
        self.Rrpwr = NumParam(default=999, tex_name='R_{rpwr}',
                              info='Low voltage power logic (LVPL) ramp limit',
                              unit='p.u.',
                              )
        self.Brkpt = NumParam(default=1.0, tex_name='B_{rkpt}',
                              info='LVPL characteristic voltage 2',
                              unit='p.u.',
                              )
        self.Zerox = NumParam(default=0.5, tex_name='Z_{erox}',
                              info='LVPL characteristic voltage 1',
                              unit='p.u',
                              )
        self.Volim = NumParam(default=999, tex_name='V_{olim}',
                              info='Voltage lim for high volt. reactive current mgnt.',
                              unit='p.u.',
                              )
        self.Lvpnt1 = NumParam(default=0.8, tex_name='L_{vpnt1}',
                               info='High volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        self.Lvpnt0 = NumParam(default=0.2, tex_name='L_{vpnt0}',
                               info='Low volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        self.Iolim = NumParam(default=999.0, tex_name='I_{olim}',
                              info='current limit for high volt. reactive current mgnt.',
                              unit='p.u.',
                              )
        self.Tfltr = NumParam(default=0.1, tex_name='T_{fltr}',
                              info='Voltage filter T const for low volt. active current mgnt.',
                              unit='s',
                              )
        self.Khv = NumParam(default=1, tex_name='K_{hv}',
                            info='Overvolt. compensation gain in high volt. reactive current mgnt.',
                            )
        self.Iqrmax = NumParam(default=999, tex_name='I_{qrmax}',
                               info='Upper limit on the ROC for reactive current',
                               unit='p.u.',
                               )
        self.Iqrmin = NumParam(default=-999, tex_name='I_{qrmin}',
                               info='Lower limit on the ROC for reactive current',
                               unit='p.u.',
                               )
        self.Accel = NumParam(default=0.0, tex_name='A_{ccel}',
                              info='Acceleration factor',
                              vrange=(0, 1.0),
                              )


class REGCAU1Model(Model):
    """
    REGCAU1 implementation.
    """
    pass
    # TODO: pick up from here.
