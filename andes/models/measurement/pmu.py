"""
Simple PMU based on low-pass filter.
"""

from andes.core import ModelData, IdxParam, NumParam, Model, ExtAlgeb, State


class PMUData(ModelData):
    """
    Phasor measurement unit data.
    """

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.Ta = NumParam(default=0.1, tex_name='T_a', info='angle filter time constant')
        self.Tv = NumParam(default=0.1, tex_name='T_v', info='voltage filter time constant')


class PMU(PMUData, Model):
    """
    Simple phasor measurement unit model.

    This model tracks the bus voltage magnitude and phase angle, each using
    a low-pass filter.
    """

    def __init__(self, system, config):
        PMUData.__init__(self)
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'PhasorMeasurement'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          )

        self.am = State(tex_name=r'\theta_m', info='phase angle measurement',
                        unit='rad.', e_str='a - am', t_const=self.Ta, v_str='a',
                        )

        self.vm = State(tex_name='V_m', info='voltage magnitude measurement',
                        unit='p.u.(kV)', e_str='v - vm', t_const=self.Tv, v_str='v',
                        )
