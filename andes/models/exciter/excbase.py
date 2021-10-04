"""
Base module for exciters.

Exciters share generator and bus information.
"""

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
                           ename='vf',
                           tex_ename='v_f',
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


class ExcVsum():
    """
    Subclass for exciter model.
    """

    def __init__(self):
        self.UEL0 = ConstService('0')
        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='UEL0',
                         e_str='UEL0 - UEL'
                         )
        self.OEL0 = ConstService('0')
        self.OEL = Algeb(info='Interface var for over exc. limiter',
                         tex_name='O_{EL}',
                         v_str='OEL0',
                         e_str='OEL0 - OEL'
                         )
        self.Vs = Algeb(info='Voltage compensation from PSS',
                        tex_name='V_{s}',
                        v_str='0',
                        e_str='0 - Vs'
                        )
        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )
