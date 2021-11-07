"""
Bus frequency estimation based on high-pass filter.
"""

from andes.core import (ModelData, Model, IdxParam, NumParam,
                        ConstService, ExtService, Algeb, ExtAlgeb,
                        Lag, Washout)


class BusFreq(ModelData, Model):
    """
    Bus frequency measurement. Outputs frequency in per unit value.

    The bus frequency output variable is `f`.
    The frequency deviation variable is `WO_y`.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'FreqMeasurement'

        # Parameters
        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.Tf = NumParam(default=0.02,
                           info="input digital filter time const",
                           unit="sec",
                           tex_name='T_f',
                           )
        self.Tw = NumParam(default=0.02,
                           info="washout time const",
                           unit="sec",
                           tex_name='T_w',
                           )
        self.fn = NumParam(default=60.0,
                           info="nominal frequency",
                           unit='Hz',
                           tex_name='f_n',
                           )

        # Variables
        self.iwn = ConstService(v_str='u / (2 * pi * fn)', tex_name=r'1/\omega_n')
        self.a0 = ExtService(src='a',
                             model='Bus',
                             indexer=self.bus,
                             tex_name=r'\theta_0',
                             info='initial phase angle',
                             )
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          )
        self.L = Lag(u='(a-a0)',
                     T=self.Tf,
                     K=1,
                     info='digital filter',
                     )
        # the output `WO_y` is the frequency deviation in p.u.
        self.WO = Washout(u=self.L_y,
                          K=self.iwn,
                          T=self.Tw,
                          info='angle washout',
                          )
        self.WO_y.info = 'frequency deviation'
        self.WO_y.unit = 'p.u. (Hz)'

        self.f = Algeb(info='frequency output',
                       unit='p.u. (Hz)',
                       tex_name='f',
                       v_str='1',
                       e_str='1 + WO_y - f',
                       )
