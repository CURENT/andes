"""
Distributed energy resource protection model base.
"""
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService, EventFlag, ExtService, VarService, ExtendedEvent
from andes.core.discrete import Limiter, Delay, Derivative
from andes.core.block import Integrator, PIController, Piecewise


class DGPRCTBaseData(ModelData):
    """
    DGPRCT Base model data.
    """

    def __init__(self):
        super(DGPRCTBaseData, self).__init__()

        self.dev = IdxParam(info='idx of the target device',
                            mandatory=True,
                            )

        self.busfreq = IdxParam(model='BusFreq',
                                info='Target device interface bus measurement device idx',
                                )

        # -- protection enable parameters
        self.fen = NumParam(default=1,
                            tex_name='fen',
                            vrange=[0, 1],
                            info='Frequency deviation protection enable. \
                                  1 for enable, 0 for disable.',
                            )

        # self.fflag = NumParam(default=1,
        #                       tex_name='fflag',
        #                       vrange=[0, 1],
        #                       info='Frequency lock option. \
        #                             1 to enable freq after protection, \
        #                             0 to disable freq after protection.',
        #                       )

        self.Ven = NumParam(default=0,
                            tex_name='Ven',
                            vrange=[0, 1],
                            info='Voltage deviation protection enable.\
                                  1 for enable, 0 for disable.',
                            )

        # -- protection parameters, frequency
        self.fl3 = NumParam(default=50,
                            tex_name='fl3',
                            info='Under frequency shadding point 3',
                            unit='Hz',
                            )

        self.fl2 = NumParam(default=57.5,
                            tex_name='fl2',
                            info='Over frequency shadding point 2',
                            unit='Hz',
                            )

        self.fl1 = NumParam(default=59.2,
                            tex_name='fl1',
                            info='Under frequency shadding point 1',
                            unit='Hz',
                            )

        self.fu1 = NumParam(default=60.5,
                            tex_name='fu1',
                            info='Over frequency shadding point 1',
                            unit='Hz',
                            )

        self.fu2 = NumParam(default=61.5,
                            tex_name='fu2',
                            info='Over frequency shadding point 2',
                            unit='Hz',
                            )

        self.fu3 = NumParam(default=70,
                            tex_name='fu3',
                            info='Over frequency shadding point 3',
                            unit='Hz',
                            )

        self.Tfl1 = NumParam(default=300,
                             tex_name=r'T_{fl1}',
                             info='Stand time for (fl2, fl1)',
                             non_negative=True,
                             )

        self.Tfl2 = NumParam(default=10,
                             tex_name=r'T_{fl2}',
                             info='Stand time for (fl3, fl2)',
                             non_negative=True,
                             )

        self.Tfu1 = NumParam(default=300,
                             tex_name=r'T_{fu1}',
                             info='Stand time for (fu1, fu2)',
                             non_negative=True,
                             )

        self.Tfu2 = NumParam(default=10,
                             tex_name=r'T_{fu2}',
                             info='Stand time for (fu2, fu3)',
                             non_negative=True,
                             )

        # -- protection parameters, voltage
        self.vl4 = NumParam(default=0.1,
                            tex_name='vl4',
                            info='Under voltage shadding point 4',
                            unit='p.u.',
                            )

        self.vl3 = NumParam(default=0.45,
                            tex_name='vl3',
                            info='Under voltage shadding point 3',
                            unit='p.u.',
                            )

        self.vl2 = NumParam(default=0.6,
                            tex_name='vl2',
                            info='Under voltage shadding point 2',
                            unit='p.u.',
                            )

        self.vl1 = NumParam(default=0.88,
                            tex_name='vl1',
                            info='Under voltage shadding point 1',
                            unit='p.u.',
                            )

        self.vu1 = NumParam(default=1.1,
                            tex_name='vu1',
                            info='Over voltage shadding point 1',
                            unit='p.u.',
                            )

        self.vu2 = NumParam(default=1.2,
                            tex_name='vu2',
                            info='Over voltage shadding point 2',
                            unit='p.u.',
                            )

        self.vu3 = NumParam(default=2,
                            tex_name='vu3',
                            info='Over voltage shadding point 3',
                            unit='p.u.',
                            )

        self.Tvl1 = NumParam(default=2,
                             tex_name=r'T_{vl1}',
                             info='Stand time for (vl2, vl1)',
                             non_negative=True,
                             )

        self.Tvl2 = NumParam(default=1,
                             tex_name=r'T_{vl2}',
                             info='Stand time for (vl3, vl2)',
                             non_negative=True,
                             )

        self.Tvl3 = NumParam(default=0.16,
                             tex_name=r'T_{vl3}',
                             info='Stand time for (vl4, vl3)',
                             non_negative=True,
                             )

        self.Tvu1 = NumParam(default=1,
                             tex_name=r'T_{vu1}',
                             info='Stand time for (vu1, vu2)',
                             non_negative=True,
                             )

        self.Tvu2 = NumParam(default=0.16,
                             tex_name=r'T_{vu2}',
                             info='Stand time for (vu2, vu3)',
                             non_negative=True,
                             )

        # -- debug
        self.Tr = NumParam(default=0.02,
                           info='Reset time constant',
                           )
        self.Tdb = NumParam(default=2,
                           info='Deadband time before reset',
                           )
        # --debug end

class DGPRCTBaseModel(Model):
    """
    Model implementation of DGPRCT Base.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DG'

        self.bus = ExtParam(model='DG', src='bus',
                            indexer=self.dev,
                            export=False)

        self.fn = ExtParam(model='DG', src='fn',
                           indexer=self.dev,
                           export=False)

        # -- Frequency protection
        # Convert frequency deviation range to p.u.
        self.f = ExtAlgeb(export=False,
                          info='DG frequency read value', unit='p.u.',
                          model='FreqMeasurement', src='f',
                          indexer=self.busfreq,
                          )

        self.fHz = Algeb(v_str='fn * f',
                         e_str='fn * f - fHz',
                         info='frequency in Hz',
                         tex_name=r'f_{Hz}',
                         )

        # -- Lock DG frequency signal and output power

        self.ltu = ConstService(v_str='0.8')
        self.ltl = ConstService(v_str='0.2')

        # `Ldsum_zu` is `ue`
        dsum = 'fen * (FLfl1_zu * Lfl1_zi + FLfl2_zu * Lfl2_zi + ' \
               'FLfu1_zu * Lfu1_zi + FLfu2_zu * Lfu2_zi) + ' \
               'Ven * (FLVl1_zu * LVl1_zi + FLVl2_zu * LVl2_zi + ' \
               'FLVl3_zu * LVl3_zi + ' \
               'FLVu1_zu * LVu1_zi + FLVu2_zu * LVu2_zi) - ' \
               'dsum'

        self.dsum = Algeb(v_str='0',
                          e_str=dsum,
                          info='lock signal summation',
                          tex_name=r'd_{tot}',
                          )

        self.Ldsum = Limiter(u=self.dsum,
                             lower=self.ltl,
                             upper=self.ltu,
                             info='lock signal comparer, zu is to act',
                             equal=False, no_warn=True,
                             )

        self.ue = Algeb(v_str='0',
                        e_str='Ldsum_zu - ue',
                        info='lock flag',
                        tex_name=r'ue',
                        )

        # lock DG frequency signal

        # fflag option 1: leave source signal online in protection
        self.fin = ExtAlgeb(model='DG', src='f',
                            indexer=self.dev,
                            info='original f from DG',
                            )

        self.fHzl = ExtAlgeb(model='DG', src='fHz',
                             indexer=self.dev,
                             export=False,
                             e_str='- ue * (fn * f)',
                             info='Frequency measure lock',
                             )

        # fflag option 2: lock source signal in protection
        # self.WOy = ExtAlgeb(model='FreqMeasurement', src='WO_y',
        #                     indexer=self.busfreq,
        #                     info='original Washout y from BusFreq',
        #                     )

        # self.fsrcl = ExtAlgeb(model='FreqMeasurement', src='f',
        #                       indexer=self.busfreq,
        #                       export=False,
        #                       e_str='- ue * (1 - fflag) * (1 + WOy)',
        #                       info='Frequency source lock',
        #                       )

        # lock output power of DG

        self.Pext = ExtAlgeb(model='DG', src='Pext',
                             indexer=self.dev,
                             info='original Pext from DG',
                             )

        self.Pref = ExtAlgeb(model='DG', src='Pref',
                             indexer=self.dev,
                             info='original Pref from DG',
                             )

        self.Pdrp = ExtAlgeb(model='DG', src='DB_y',
                             indexer=self.dev,
                             info='original Pdrp from DG',
                             )

        self.Psum = ExtAlgeb(model='DG', src='Psum',
                             indexer=self.dev,
                             export=False,
                             e_str='- ue * (Pext + Pref + Pdrp)',
                             info='Active power lock',
                             )

        self.Qdrp = ExtAlgeb(model='DG', src='Qdrp',
                             indexer=self.dev,
                             info='original Qdrp from DG',
                             )

        self.Qref = ExtAlgeb(model='DG', src='Qref',
                             indexer=self.dev,
                             info='original Qref from DG',
                             )

        self.Qsum = ExtAlgeb(model='DG', src='Qsum',
                             indexer=self.dev,
                             export=False,
                             e_str='- ue * (Qdrp + Qref)',
                             info='Reactive power lock',
                             )

        # TODO: apply ExtState here
        # TODO: clear State: BusFreq WO(Washout)
        # TODO: clear State: PVD1: Ipout, Iqout
        # TODO: develop integrator with reset function
        # TODO: Set EventFlag for ue


class fProtect:
    """
    Subclass for frequency protection logic.
    """

    def __init__(self):
        # Indicatior of frequency deviation
        self.Lfl1 = Limiter(u=self.fHz,
                            lower=self.fl3, upper=self.fl1,
                            info='Frequency comparer for (fl3, fl1)',
                            equal=False, no_warn=True,
                            )

        self.Lfl2 = Limiter(u=self.fHz,
                            lower=self.fl3, upper=self.fl2,
                            info='Frequency comparer for (fl3, fl2)',
                            equal=False, no_warn=True,
                            )

        self.Lfu1 = Limiter(u=self.fHz,
                            lower=self.fu1, upper=self.fu3,
                            info='Frequency comparer for (fu1, fu3)',
                            equal=False, no_warn=True,
                            )

        self.Lfu2 = Limiter(u=self.fHz,
                            lower=self.fu2, upper=self.fu3,
                            info='Frequency comparer for (fu2, fu3)',
                            equal=False, no_warn=True,
                            )

        # Frequency deviation time continuity check
        self.INTfl1 = Integrator(u='Lfl1_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (fl3, fl1)',
                                 )

        self.FLfl1 = Limiter(u=self.INTfl1_y,
                             lower=0, upper=self.Tfl1,
                             info='Flag comparer for (fl3, fl1)',
                             equal=False, no_warn=True,
                             )

        self.INTfl2 = Integrator(u='Lfl2_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (fl3, fl2)',
                                 )

        self.FLfl2 = Limiter(u=self.INTfl2_y,
                             lower=0, upper=self.Tfl2,
                             info='Flag comparer for (fl3, fl2)',
                             equal=False, no_warn=True,
                             )

        self.INTfu1 = Integrator(u='Lfu1_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (fu1, fu3)',
                                 )

        self.FLfu1 = Limiter(u=self.INTfu1_y,
                             lower=0, upper=self.Tfu1,
                             info='Flag comparer for (fu1, fu3)',
                             equal=False, no_warn=True,
                             )

        self.INTfu2 = Integrator(u='Lfu2_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (fu2, fu3)',
                                 )

        self.FLfu2 = Limiter(u=self.INTfu2_y,
                             lower=0, upper=self.Tfu2,
                             info='Flag comparer for (fu2, fu3)',
                             equal=False, no_warn=True,
                             )

        # -- debug

        self.uevs = VarService(v_str='1 - Ldsum_zu',
                                   info='Voltage dip flag; 1-dip, 0-normal',
                                   tex_name='z_{Vdip}',
                                   )

        self.ueee = ExtendedEvent(self.uevs, t_ext=self.Tdb, trig="rise", extend_only=True,)

        self.ob = Algeb(v_str='0',
                          e_str='ueee - ob',
                          info='observe ueee',
                          tex_name=r'ueee_{ob}',
                          )

        # self.L = Limiter(u=self.fHz,
        #                     lower=self.fl3, upper=self.fl1,
        #                     info='Frequency comparer for (fl3, fl1)',
        #                     equal=False, no_warn=True,
        #                     )

        # delay is inapproriate
        # self.dtime = Delay(u=self.data, mode='time', delay=self.time)

        # self.INT = Integrator(u='L_zi * INTL_zi + Tfl1 / Tr * (1 - ue) * (1 - L_zi)',
        #                          T=1.0, K=1.0,
        #                          y0='0',
        #                          info='Flag integerator for (fu2, fu3)',
        #                          )

        # how about derivative?
        # self.dv = Derivative(self.INT_y, tex_name='dV/dt', info='Finite difference of bus voltage')

        # self.INTL = Limiter(u=self.INT_y,
        #                      lower=0, upper=self.Tfl1,
        #                      info='Flag comparer for (fu2, fu3)',
        #                      equal=True, no_warn=True,
        #                      )

        # self.PI = PIController(u='L_zi * (1 - ue) * (1 - L_zu) + PI_y * re',
        #                          kp=0,
        #                          ki='1 * (1 - ue) * L_zi + Tfl1 / Tr * (1 - ue) * PIL_zi',
        #                          ref='Tfl1 * (1 - ue)',
        #                          x0='0',
        #                          )

        # self.PIL = Limiter(u='PI_y',
        #                      lower=0, upper=self.Tfl1,
        #                      info='Flag comparer for (fu2, fu3)',
        #                      equal=False, no_warn=True,
        #                      )

        # -- debug end


class VProtect:
    """
    Subclass for voltage protection logic
    """

    def __init__(self):
        # -- Voltage protection

        # Indicatior of voltage deviation
        self.LVl1 = Limiter(u=self.v,
                            lower=self.vl4, upper=self.vl1,
                            info='Voltage comparer for (vl4, vl1)',
                            equal=False, no_warn=True,
                            )

        self.LVl2 = Limiter(u=self.v,
                            lower=self.vl4, upper=self.vl2,
                            info='Voltage comparer for (vl4, vl2)',
                            equal=False, no_warn=True,
                            )

        self.LVl3 = Limiter(u=self.v,
                            lower=self.vl4, upper=self.vl3,
                            info='Voltage comparer for (vl4, vl3)',
                            equal=False, no_warn=True,
                            )

        self.LVu1 = Limiter(u=self.v,
                            lower=self.vu1, upper=self.vu3,
                            info='Voltage comparer for (vu1, vu3)',
                            equal=False, no_warn=True,
                            )

        self.LVu2 = Limiter(u=self.v,
                            lower=self.vu2, upper=self.vu3,
                            info='Voltage comparer for (vu2, vu3)',
                            equal=False, no_warn=True,
                            )

        # Voltage deviation time continuity check
        self.INTVl1 = Integrator(u='LVl1_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (vl4, vl1)',
                                 )

        self.FLVl1 = Limiter(u=self.INTVl1_y,
                             lower=0, upper=self.Tvl1,
                             info='Flag comparer for (vl4, vl1)',
                             equal=False, no_warn=True,
                             )

        self.INTVl2 = Integrator(u='LVl2_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (vl4, vl2)',
                                 )

        self.FLVl2 = Limiter(u=self.INTVl2_y,
                             lower=0, upper=self.Tvl2,
                             info='Flag comparer for (vl4, vl2)',
                             equal=False,
                             no_warn=True,
                             )

        self.INTVl3 = Integrator(u='LVl2_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (vl4, vl3)',
                                 )

        self.FLVl3 = Limiter(u=self.INTVl3_y,
                             lower=0, upper=self.Tvl3,
                             info='Flag comparer for (vl4, vl3)',
                             equal=False,
                             no_warn=True,
                             )

        self.INTVu1 = Integrator(u='LVu1_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (vu1, vu3)',
                                 )

        self.FLVu1 = Limiter(u=self.INTVu1_y,
                             lower=0, upper=self.Tvu1,
                             info='Flag comparer for (vu1, vu3)',
                             equal=False, no_warn=True,
                             )

        self.INTVu2 = Integrator(u='LVu2_zi',
                                 T=1.0, K=1.0,
                                 y0='0',
                                 info='Flag integerator for (vu2, vu3)',
                                 )

        self.FLVu2 = Limiter(u=self.INTVu2_y,
                             lower=0, upper=self.Tvu2,
                             info='Flag comparer for (vu2, vu3)',
                             equal=False, no_warn=True,
                             )


class DGPRCT1Model(DGPRCTBaseModel, fProtect, VProtect):
    """
    Model implementation of DGPRCT1.
    """

    def __init__(self, system, config):
        DGPRCTBaseModel.__init__(self, system, config)
        self.v = ExtAlgeb(model='Bus', src='v',
                          indexer=self.bus,
                          export=False,
                          info='Bus voltage',
                          unit='p.u.',
                          )
        fProtect.__init__(self)
        VProtect.__init__(self)


class DGPRCT1(DGPRCTBaseData, DGPRCT1Model):
    """
    DGPRCT1 model, follow IEEE-1547. DGPRCT stands for DG protection.

    Target device (limited to DG group) ``Psum`` and ``Qsum`` will decrease to zero
    immediately when frequency/voltage protection flag is raised. Once the lock is
    released, ``Psum`` and ``Qsum`` will return to normal immediately.

    ``fen`` and ``Ven`` are protection enabling parameters. 1 is on and 0 is off.

    ``ue`` is lock flag signal.

    It should be noted that, the lock only lock the ``fHz`` (frequency read value) of DG model.
    The source values (which come from ``BusFreq`` `f` remain unchanged.)

    DGPRCT1 can only be used once in a simulation.

    The model does not check the shedding points sequence.
    The input parameters are required to satisfy `fl3 < fl2 < fl1 < fu1 < fu2 < fu3`, and
    `ul4 < ul3 < ul2 < ul1 < uu1 < uu2 < uu3`.

    Default settings:\n
    Frequency (Hz):\n
    `(fl3, fl2), Tfl2` [(50.0, 57.5), 10s]\n
    `(fl2, fl1), Tfl1` [(57.5, 59.2), 300s]\n
    `(fu1, fu2), Tfu1` [(60.5, 61.5), 300s]\n
    `(fu2, fu3), Tfu2` [(61.5, 70.0), 10s]\n

    Voltage (p.u.):\n
    `(vl4, vl3), Tvl3` [(0.10, 0.45), 0.16s]\n
    `(vl3, vl2), Tvl2` [(0.45, 0.60), 1s]\n
    `(vl2, vl1), Tvl1` [(0.60, 0.88), 2s]\n
    `(vu1, vu2), Tvu1` [(1.10, 1.20), 1s]\n
    `(vu2, vu3), Tvu2` [(1.20, 2.00), 0.16s]\n
    """

    def __init__(self, system, config):
        DGPRCTBaseData.__init__(self)
        DGPRCT1Model.__init__(self, system, config)


class DGPRCTExtModel(Model):
    """
    Model implementation of DGPRCT1.
    """

    def __init__(self, system, config):
        DGPRCTBaseModel.__init__(self, system, config)

        self.v = ExtService(model='Bus', src='v',
                            indexer=self.bus,
                            info='retrived voltage',
                            tex_name='v',
                            )

        fProtect.__init__(self)
        VProtect.__init__(self)


class DGPRCTExt(DGPRCTBaseData, DGPRCTExtModel):
    """
    DGPRCT External model, follow IEEE-1547. DGPRCT stands for DG protection.

    Similar to DGPRCT, but the measured voltage is given from outside.

    Target device (limited to DG group) ``Psum`` and ``Qsum`` will decrease to zero
    immediately when frequency/voltage protection flag is raised. Once the lock is
    released, ``Psum`` and ``Qsum`` will return to normal immediately.

    ``fen`` and ``Ven`` are protection enabling parameters. 1 is on and 0 is off.

    ``ue`` is lock flag signal.

    It should be noted that, the lock only lock the ``fHz`` (frequency read value) of DG model.
    The source values (which come from ``BusFreq`` `f` remain unchanged.)

    DGPRCTExt can only be used once in a simulation.

    The model does not check the shedding points sequence.
    The input parameters are required to satisfy `fl3 < fl2 < fl1 < fu1 < fu2 < fu3`, and
    `ul4 < ul3 < ul2 < ul1 < uu1 < uu2 < uu3`.

    Default settings:\n
    Frequency (Hz):\n
    `(fl3, fl2), Tfl2` [(50.0, 57.5), 10s]\n
    `(fl2, fl1), Tfl1` [(57.5, 59.2), 300s]\n
    `(fu1, fu2), Tfu1` [(60.5, 61.5), 300s]\n
    `(fu2, fu3), Tfu2` [(61.5, 70.0), 10s]\n

    Voltage (p.u.):\n
    `(vl4, vl3), Tvl3` [(0.10, 0.45), 0.16s]\n
    `(vl3, vl2), Tvl2` [(0.45, 0.60), 1s]\n
    `(vl2, vl1), Tvl1` [(0.60, 0.88), 2s]\n
    `(vu1, vu2), Tvu1` [(1.10, 1.20), 1s]\n
    `(vu2, vu3), Tvu2` [(1.20, 2.00), 0.16s]\n
    """

    def __init__(self, system, config):
        DGPRCTBaseData.__init__(self)
        DGPRCTExtModel.__init__(self, system, config)
