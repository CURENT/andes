from andes.core.model import Model, ModelData
from andes.core.param import ExtParam, IdxParam, NumParam
from andes.core.service import ExtService, VarService
from andes.core.var import Algeb, ExtAlgeb


class IEEEVCData(ModelData):
    """
    IEEEVC data.
    """

    def __init__(self):
        ModelData.__init__(self)
        self.avr = IdxParam(info='Exciter idx', mandatory=True, model='Exciter')

        self.rc = NumParam(default=0.0,
                           info="Active compensation degree.",
                           z=True,
                           tex_name='r_c'
                           )
        self.xc = NumParam(default=0.0,
                           info="Reactive compensation degree.",
                           z=True,
                           tex_name='x_c'
                           )


class IEEEVCModel(Model):
    """
    Implementation of the IEEEVC model.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'VoltComp'
        self.flags.tds = True

        # retrieve indices of connected generator, bus, and bus freq
        self.syn = ExtParam(model='Exciter', src='syn', indexer=self.avr, export=False,
                            info='Retrieved generator idx', vtype=str)

        self.vf0 = ExtService(src='vf',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'v_{f0}',
                              info='Steady state excitation voltage')
        # from Bus
        self.v = ExtAlgeb(model='SynGen', src='v', indexer=self.syn, tex_name=r'V',
                          info='Retrieved bus terminal voltage',
                          )
        # vd, vq, Id, Iq from SynGen
        self.vd = ExtAlgeb(src='vd',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_d',
                           info='d-axis machine voltage',
                           )
        self.vq = ExtAlgeb(src='vq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_q',
                           info='q-axis machine voltage',
                           )
        self.Id = ExtAlgeb(src='Id',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_d',
                           info='d-axis machine current',
                           )
        self.Iq = ExtAlgeb(src='Iq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_q',
                           info='q-axis machine current',
                           )

        self.vct = VarService(tex_name=r'V_{CT}',
                              v_str='u * Abs((vd + 1j*vq) + (rc + 1j * xc) * (Id + 1j*Iq))',
                              )

        # output voltage.
        # `vcomp` is the additional voltage to be added to bus terminal voltage
        self.vcomp = Algeb(info='Compensator output voltage to exciter',
                           tex_name=r'v_{comp}',
                           v_str='vct - u * v',
                           e_str='vct - u * v - vcomp',
                           )

        # do not need to interface to exciters here.
        # Let the exciters pick up `vcomp` through back referencing

        self.Eterm = ExtAlgeb(model='Exciter',
                              src='v',
                              indexer=self.avr,
                              v_str='vcomp',
                              e_str='vcomp',
                              )


class IEEEVC(IEEEVCData, IEEEVCModel):
    """
    Voltage compensator IEEEVC model.

    Reference:

    [1] PowerWorld, Voltage Compensator, IEEEVC, [Online],

    [2] NEPLAN, Exciters Models, [Online],

    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Voltage%20Compensator%20IEEEVC.htm?TocPath=%7C%7C%7CIEEEVC%7C_____0

    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_EXCITERS1.pdf
    """
    def __init__(self, system, config):
        IEEEVCData.__init__(self)
        IEEEVCModel.__init__(self, system, config)
