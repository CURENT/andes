"""
Distributed energy resource protection model base.
"""
from andes.core.block import IntegratorAntiWindup
from andes.core.discrete import Limiter
from andes.core.model import Model, ModelData
from andes.core.param import ExtParam, IdxParam, NumParam
from andes.core.service import ConstService, ExtendedEvent, ExtService
from andes.core.var import Algeb, ExtAlgeb


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
                            vrange=(0, 1),
                            info='Frequency deviation protection enable. \
                                  1 for enable, 0 for disable.',
                            )

        # TODO: add `fflag` for choice that whether block source signal
        # self.fflag = NumParam(default=1,
        #                       tex_name='fflag',
        #                       vrange=[0, 1],
        #                       info='Frequency lock option. \
        #                             1 to enable freq after protection, \
        #                             0 to disable freq after protection.',
        #                       )

        self.Ven = NumParam(default=0,
                            tex_name='Ven',
                            vrange=(0, 1),
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

        self.Tres = NumParam(default=0.05,
                             info='Integrator reset time',
                             )


class DGPRCTBaseModel(Model):
    """
    Model implementation of DGPRCT Base.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DGProtection'

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
        dsum = 'fen * (IAWfl1_lim_zu * Lfl1_zi + IAWfl2_lim_zu * Lfl2_zi + ' \
               'IAWfu1_lim_zu * Lfu1_zi + IAWfu2_lim_zu * Lfu2_zi) + ' \
               'Ven * (IAWVl1_lim_zu * LVl1_zi + IAWVl2_lim_zu * LVl2_zi + ' \
               'IAWVl3_lim_zu * LVl3_zi + ' \
               'IAWVu1_lim_zu * LVu1_zi + IAWVu2_lim_zu * LVu2_zi) - ' \
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

        self.zero = ConstService('0')

        self.res = ExtendedEvent(self.ue, t_ext=self.Tres,
                                 trig="rise",
                                 extend_only=True)

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
                             ename='fHzl',
                             tex_ename='f_{Hzl}',
                             )
        # TODO: add fflag option 2: block the source signal in protection

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
                             ename='Pneg',
                             tex_ename='P_{neg}',
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
                             ename='Qneg',
                             tex_ename='Q_{neg}',
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

        self.IAWfl1 = IntegratorAntiWindup(u='Lfl1_zi * (1 - res) - Tfl1 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tfl1,
                                           info='condition check for (fl3, fl1)',
                                           no_warn=True,
                                           )

        self.IAWfl2 = IntegratorAntiWindup(u='Lfl2_zi * (1 - res) - Tfl2 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tfl2,
                                           info='condition check for (fl3, fl2)',
                                           no_warn=True,
                                           )

        self.IAWfu1 = IntegratorAntiWindup(u='Lfu1_zi * (1 - res) - Tfu1 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tfu1,
                                           info='condition check for (fu1, fu3)',
                                           no_warn=True,
                                           )

        self.IAWfu2 = IntegratorAntiWindup(u='Lfl2_zi * (1 - res) - Tfu2 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tfu2,
                                           info='condition check for (fu2, fu3)',
                                           no_warn=True,
                                           )


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
        self.IAWVl1 = IntegratorAntiWindup(u='LVl1_zi * (1 - res) - Tvl1 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tvl1,
                                           info='condition check for (Vl3, Vl1)',
                                           no_warn=True,
                                           )

        self.IAWVl2 = IntegratorAntiWindup(u='LVl2_zi * (1 - res) - Tvl2 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tvl2,
                                           info='condition check for (Vl3, Vl2)',
                                           no_warn=True,
                                           )

        self.IAWVl3 = IntegratorAntiWindup(u='LVl2_zi * (1 - res) - Tvl3 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tvl2,
                                           info='condition check for (Vl3, Vl2)',
                                           no_warn=True,
                                           )

        self.IAWVu1 = IntegratorAntiWindup(u='LVu1_zi * (1 - res) - Tvu1 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tvu1,
                                           info='condition check for (Vu1, Vu3)',
                                           no_warn=True,
                                           )

        self.IAWVu2 = IntegratorAntiWindup(u='LVu2_zi * (1 - res) - Tvu2 / Tres * res',
                                           T=1, K=1, y0='0',
                                           lower=self.zero, upper=self.Tvu2,
                                           info='condition check for (Vu2, Vu3)',
                                           no_warn=True,
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
    DGPRCT1 model, follow IEEE-1547-2018. DGPRCT stands for DG protection.

    A demo is provided: examples/demonstration/1.1 demo_DGPRCT1.ipynb

    Target device (limited to DG group) ``Psum`` and ``Qsum`` will decrease to zero
    immediately when frequency/voltage protection flag is raised. Once the lock is
    released, ``Psum`` and ``Qsum`` will return to normal immediately.

    DG group base model ``PVD1`` already has a degrading function which is used to
    degrade output under abnormal condition. it is recommended to turn it off by setting
    `recflag = 0`.

    ``fen`` and ``Ven`` are protection enabling parameters. 1/0 is on/off.

    ``ue`` is lock flag signal.

    It should be noted that, the lock only lock the ``fHz`` (frequency read value)
    of DG model. The source values (which come from ``BusFreq`` `f` remain unchanged.)

    Protection sensors (e.g., IAWfl1) are instances of ``IntergratorAntiWindup``. All
    the protection sensors will be reset after ``ue`` returns to 0.
    Resetting action takes `Tres` to finish.

    The model does not check the shedding points sequence.
    The input parameters are required to satisfy ``fl3 < fl2 < fl1 < fu1 < fu2 < fu3``,
    and ``ul4 < ul3 < ul2 < ul1 < uu1 < uu2 < uu3``.

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

    Reference:

    NERC. Bulk Power System Reliability Perspectives on the Adoption of IEEE 1547-2018.
    March 2020. Available:

    https://www.nerc.com/comm/PC_Reliability_Guidelines_DL/Guideline_IEEE_1547-2018_BPS_Perspectives.pdf
    """

    def __init__(self, system, config):
        DGPRCTBaseData.__init__(self)
        DGPRCT1Model.__init__(self, system, config)


class DGPRCTExtModel(DGPRCTBaseModel, fProtect, VProtect):
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
    DGPRCT External model, follow IEEE-1547-2018. DGPRCT stands for DG protection.

    Similar to DGPRCT1, but the measured voltage can be manipulated.

    A demo is provided: examples/demonstration/1.2 demo_DGPRCTExt.ipynb

    This model can be applied to co-simulation, where you can input the external votage signal
    into ANDES. If no extertal value is applied, the votalge will remain as the initialized value.

    Target device (limited to DG group) ``Psum`` and ``Qsum`` will decrease to zero
    immediately when frequency/voltage protection flag is raised. Once the lock is
    released, ``Psum`` and ``Qsum`` will return to normal immediately.

    DG group base model ``PVD1`` already has a degrading function which is used to
    degrade output under abnormal condition. it is recommended to turn it off by setting
    `recflag = 0`.

    ``fen`` and ``Ven`` are protection enabling parameters. 1/0 is on/off.

    ``ue`` is lock flag signal.

    It should be noted that, the lock only lock the ``fHz`` (frequency read value)
    of DG model. The source values (which come from ``BusFreq`` `f` remain unchanged.)

    Protection sensors (e.g., IAWfl1) are instances of ``IntergratorAntiWindup``. All
    the protection sensors will be reset after ``ue`` returns to 0.
    Resetting action takes `Tres` to finish.

    The model does not check the shedding points sequence.
    The input parameters are required to satisfy ``fl3 < fl2 < fl1 < fu1 < fu2 < fu3``,
    and ``ul4 < ul3 < ul2 < ul1 < uu1 < uu2 < uu3``.

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

    Reference:

    NERC. Bulk Power System Reliability Perspectives on the Adoption of IEEE 1547-2018.
    March 2020. Available:

    https://www.nerc.com/comm/PC_Reliability_Guidelines_DL/Guideline_IEEE_1547-2018_BPS_Perspectives.pdf
    """

    def __init__(self, system, config):
        DGPRCTBaseData.__init__(self)
        DGPRCTExtModel.__init__(self, system, config)
