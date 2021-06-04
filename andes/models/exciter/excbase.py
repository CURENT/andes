from andes.core.model import ModelData, Model
from andes.core.param import IdxParam, ExtParam
from andes.core.var import Algeb, ExtState, ExtAlgeb

from andes.core.service import ExtService, ConstService


class ExcBaseData(ModelData):
    """
    Common parameters for exciters.
    """

    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            )


class ExcBase(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Exciter'
        self.flags.tds = True

        # from synchronous generators, get u, Sn, Vn, bus; tm0; omega
        self.ug = ExtParam(src='u',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='u_g',
                           info='Generator online status',
                           unit='bool',
                           export=False,
                           )
        self.ue = ConstService(v_str='u * ug',
                               info="effective online status",
                               tex_name='u_e',
                               )
        self.Sn = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_m',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_m',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )
        self.vf0 = ExtService(src='vf',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name='v_{f0}',
                              info='Steady state excitation voltage')
        self.bus = ExtParam(src='bus',
                            model='SynGen',
                            indexer=self.syn,
                            tex_name='bus',
                            info='Bus idx of the generators',
                            export=False,
                            vtype=str,
                            )
        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              )
        self.vf = ExtAlgeb(src='vf',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'v_f',
                           e_str='ue * (vout - vf0)',
                           info='Excitation field voltage to generator',
                           )
        self.XadIfd = ExtAlgeb(src='XadIfd',
                               model='SynGen',
                               indexer=self.syn,
                               tex_name=r'X_{ad}I_{fd}',
                               info='Armature excitation current',
                               )
        # from bus, get a and v
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

        # output excitation voltage
        self.vout = Algeb(info='Exciter final output voltage',
                          tex_name='v_{out}',
                          v_str='vf0',
                          diag_eps=True,
                          )

        # Note:
        # Subclasses need to define `self.vref0` in the appropriate place.
        # Subclasses also need to define `self.vref`.
