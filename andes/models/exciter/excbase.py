"""
Base module for exciters.

Exciters share generator and bus information.
"""

from andes.core.block import Integrator
from andes.core.discrete import LessThan
from andes.core.model import Model, ModelData
from andes.core.param import ExtParam, IdxParam
from andes.core.service import BackRef, ConstService, ExtService
from andes.core.var import Algeb, ExtAlgeb, ExtState
from andes.models.exciter.saturation import ExcQuadSat


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
    """
    Base model for exciters.

    Notes
    -----
    As of v1.4.5, the input voltage Eterm (variable ``self.v``) is converted to
    type ``Algeb``. Since variables are evaluated after services,
    ``ConstService`` of exciters can no longer depend on ``v``.

    TODO: programmatically disallow ``ConstService`` use uninitialized
    variables.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Exciter'
        self.flags.tds = True

        # Voltage compensator idx-es
        self.VoltComp = BackRef()

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
                              is_input=True,
                              )
        self.vf = ExtAlgeb(src='vf',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'v_f',
                           e_str='ue * (vout - vf0)',
                           info='Excitation field voltage to generator',
                           ename='vf',
                           tex_ename='v_f',
                           is_input=False,
                           )
        self.XadIfd = ExtAlgeb(src='XadIfd',
                               model='SynGen',
                               indexer=self.syn,
                               tex_name=r'X_{ad}I_{fd}',
                               info='Armature excitation current',
                               is_input=True,
                               )
        # from bus, get a and v
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          is_input=True,
                          )
        self.vbus = ExtAlgeb(model='Bus',
                             src='v',
                             indexer=self.bus,
                             tex_name='V',
                             info='Bus voltage magnitude',
                             is_input=True
                             )

        # `self.v` is actually `ETERM` in other software
        # TODO:
        # Preferably, its name needs to be changed to `eterm`.
        # That requires updates in equations of all exciters.
        self.v = Algeb(info='Input to exciter (bus v or Eterm)',
                       tex_name='E_{term}',
                       v_str='vbus',
                       e_str='vbus - v',
                       v_str_add=True,
                       )

        # output excitation voltage
        self.vout = Algeb(info='Exciter final output voltage',
                          tex_name='v_{out}',
                          v_str='ue * vf0',
                          diag_eps=True,
                          is_output=True,
                          )

        # Note:
        # Subclasses need to define `self.vref0` in the appropriate place.
        # Subclasses also need to define `self.vref`.


class ExcVsum:
    """
    Subclass for the input section of exciters.

    It creates the placeholder variables for OEL, UEL, stabilizer, and ``vref``.
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
                          e_str='vref0 - vref'
                          # TODO: subclass to provide `vi.v_str`
                          )


class ExcACSat:
    """
    Subclass for the saturation commonly used in AC-type exciters.

    The inheriting class must define ``self.INTin`` as a string of
    equation for the input to ``self.INT``.
    """

    def __init__(self):

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        # Input (VR - VFE)
        self.INT = Integrator(u=self.INTin,
                              T=self.TE,
                              K=1,
                              y0=0,
                              info='V_E, integrator',
                              )
        self.INT.y.v_str = 0.1
        self.INT.y.v_iter = 'INT_y * FEX_y - vf0'

        self.SL = LessThan(u=self.INT_y, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"V_{out}*S_e(|V_{out}|)", info='saturation output',
                        v_str='Indicator(INT_y > SAT_A) * SAT_B * (INT_y - SAT_A) ** 2',
                        e_str='ue * (SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B - Se)',
                        diag_eps=True,
                        )

        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name='V_{FE}',
                         unit='p.u.',
                         v_str='INT_y * KE + Se + XadIfd * KD',
                         e_str='ue * (INT_y * KE + Se + XadIfd * KD - VFE)',
                         diag_eps=True,
                         )
