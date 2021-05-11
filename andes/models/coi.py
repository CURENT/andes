"""
Classes for Center of Inertia calculation.
"""
import numpy as np

from andes.core.param import ExtParam
from andes.core.service import NumRepeat, IdxRepeat, BackRef, IdxJoin
from andes.core.service import NumReduce, RefFlatten, ExtService, ConstService
from andes.core.var import ExtState, Algeb, ExtAlgeb
from andes.core.model import ModelData, Model


class COIData(ModelData):
    """COI parameter data"""

    def __init__(self):
        ModelData.__init__(self)


class COIModel(Model):
    """
    Implementation of COI.

    To understand this model, please refer to
    :py:class:`andes.core.service.NumReduce`,
    :py:class:`andes.core.service.NumRepeat`,
    :py:class:`andes.core.service.IdxFlatten`, and
    :py:class:`andes.core.service.BackRef`.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Calculation'
        self.flags.update({'tds': True})

        self.SynGen = BackRef(info='Back reference to SynGen idx')

        self.SynGenIdx = RefFlatten(ref=self.SynGen)

        self.M = ExtParam(model='SynGen', src='M',
                          indexer=self.SynGenIdx, export=False,
                          info='Linearly stored SynGen.M',
                          )

        self.wgen = ExtState(model='SynGen',
                             src='omega',
                             indexer=self.SynGenIdx,
                             tex_name=r'\omega_{gen}',
                             info='Linearly stored SynGen.omega',
                             )
        self.agen = ExtState(model='SynGen',
                             src='delta',
                             indexer=self.SynGenIdx,
                             tex_name=r'\delta_{gen}',
                             info='Linearly stored SynGen.delta',
                             )
        self.d0 = ExtService(model='SynGen',
                             src='delta',
                             indexer=self.SynGenIdx,
                             tex_name=r'\delta_{gen,0}',
                             info='Linearly stored initial delta',
                             )

        self.a0 = ExtService(model='SynGen',
                             src='omega',
                             indexer=self.SynGenIdx,
                             tex_name=r'\omega_{gen,0}',
                             info='Linearly stored initial omega',
                             )

        self.Mt = NumReduce(u=self.M,
                            tex_name='M_t',
                            fun=np.sum,
                            ref=self.SynGen,
                            info='Summation of M by COI index',
                            )

        self.Mr = NumRepeat(u=self.Mt,
                            tex_name='M_{tr}',
                            ref=self.SynGen,
                            info='Repeated summation of M',
                            )

        self.Mw = ConstService(tex_name='M_w',
                               info='Inertia weights',
                               v_str='M/Mr')

        self.d0w = ConstService(tex_name=r'\delta_{gen,0,w}',
                                v_str='d0 * Mw',
                                info='Linearly stored weighted delta')

        self.a0w = ConstService(tex_name=r'\omega_{gen,0,w}',
                                v_str='a0 * Mw',
                                info='Linearly stored weighted omega')

        self.d0a = NumReduce(u=self.d0w,
                             tex_name=r'\delta_{gen,0,avg}',
                             fun=np.sum,
                             ref=self.SynGen,
                             info='Average initial delta',
                             cache=False,
                             )

        self.a0a = NumReduce(u=self.a0w,
                             tex_name=r'\omega_{gen,0,avg}',
                             fun=np.sum,
                             ref=self.SynGen,
                             info='Average initial omega',
                             cache=False,
                             )

        self.pidx = IdxRepeat(u=self.idx, ref=self.SynGen, info='Repeated COI.idx')

        # Note:
        # Even if d(omega) /d (omega) = 1, it is still stored as a lambda function.
        # When no SynGen is referencing any COI, j_update will not be called,
        # and Jacobian will become singular. `diag_eps = True` needs to be used.

        # Note:
        # Do not assign `v_str=1` for `omega`. Otherwise, COIs with no connected generators will
        # fail to initialize.
        self.omega = Algeb(tex_name=r'\omega_{coi}',
                           info='COI speed',
                           v_str='a0a',
                           v_setter=True,
                           e_str='-omega',
                           diag_eps=True,
                           )
        self.delta = Algeb(tex_name=r'\delta_{coi}',
                           info='COI rotor angle',
                           v_str='d0a',
                           v_setter=True,
                           e_str='-delta',
                           diag_eps=True,
                           )

        # Note:
        # `omega_sub` or `delta_sub` must not provide `v_str`.
        # Otherwise, values will be incorrectly summed for `omega` and `delta`.
        self.omega_sub = ExtAlgeb(model='COI',
                                  src='omega',
                                  e_str='Mw * wgen',
                                  indexer=self.pidx,
                                  info='COI frequency contribution of each generator'
                                  )
        self.delta_sub = ExtAlgeb(model='COI',
                                  src='delta',
                                  e_str='Mw * agen',
                                  indexer=self.pidx,
                                  info='COI angle contribution of each generator'
                                  )

    def set_in_use(self):
        """
        Set the ``Model.in_use`` flag based on ``len(self.SynGenIdx.v)``.
        """
        self.in_use = (len(self.SynGenIdx.v) > 0)


class COI(COIData, COIModel):
    """
    Center of inertia calculation class.
    """

    def __init__(self, system, config):
        COIData.__init__(self)
        COIModel.__init__(self, system, config)

class COI2Data(ModelData):
    """COI parameter data"""

    def __init__(self):
        ModelData.__init__(self)

class COI2Model(Model):
    """
    Implementation of COI.
    Added VSG int COI calculation.

    To understand this model, please refer to
    :py:class:`andes.core.service.NumReduce`,
    :py:class:`andes.core.service.NumRepeat`,
    :py:class:`andes.core.service.IdxFlatten`, and
    :py:class:`andes.core.service.BackRef`.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Calculation'
        self.flags.update({'tds': True})

        self.SynGen = BackRef(info='Back reference to SynGen idx')

        self.SynGenIdx = RefFlatten(ref=self.SynGen)

        self.REGCVSG = BackRef(info='Back reference to REGCVSG idx')

        self.REGCVSGIdx = RefFlatten(ref=self.REGCVSG)

        self.RefModel = IdxJoin(u2=self.SynGen,u1=self.REGCVSG)
        self.RefModelIdx = RefFlatten(ref=self.RefModel)

        self.MSynGen = ExtParam(model='SynGen', src='M',
                          indexer=self.SynGenIdx, export=False,
                          info='Linearly stored SynGen.M',
                          )
        self.MREGCVSG = ExtParam(model='REGCVSG', src='M',
                          indexer=self.REGCVSGIdx, export=False,
                          info='Linearly stored SynGen.M',
                          )

        self.wSynGen = ExtState(model='SynGen',
                                   src='omega',
                                   indexer=self.SynGenIdx,
                                   tex_name=r'\omega_{gen}',
                                   info='Linearly stored SynGen.omega',
                                   )
        self.wREGCVSG = ExtAlgeb(model='REGCVSG',
                                    src='omega',
                                    indexer=self.REGCVSGIdx,
                                    tex_name=r'\omega_{gen}',
                                    info='Linearly stored REGCVSG.omega',
                                    )

        self.aSynGen = ExtState(model='SynGen',
                                    src='delta',
                                    indexer=self.SynGenIdx,
                                    tex_name=r'\delta_{gen}',
                                    info='Linearly stored SynGen.delta',
                                    )
        self.aREGCVSG = ExtState(model='REGCVSG',
                                    src='delta',
                                    indexer=self.REGCVSGIdx,
                                    tex_name=r'\delta_{gen}',
                                    info='Linearly stored REGCVSG.delta',
                                    )

        self.d0SynGen = ExtService(model='SynGen',
                                    src='delta',
                                    indexer=self.SynGenIdx,
                                    tex_name=r'\delta_{gen,0}',
                                    info='Linearly stored initial delta',
                                    )
        self.d0REGCVSG = ExtService(model='REGCVSG',
                                    src='delta',
                                    indexer=self.REGCVSGIdx,
                                    tex_name=r'\delta_{gen,0}',
                                    info='Linearly stored initial delta',
                                    )

        self.a0SynGen = ExtService(model='SynGen',
                             src='omega',
                             indexer=self.SynGenIdx,
                             tex_name=r'\omega_{gen,0}',
                             info='Linearly stored initial omega',
                             )
        self.a0REGCVSG = ExtService(model='REGCVSG',
                             src='omega',
                             indexer=self.REGCVSGIdx,
                             tex_name=r'\omega_{gen,0}',
                             info='Linearly stored initial omega',
                             )

        self.MtSynGen = NumReduce(u=self.MSynGen,
                                    tex_name='M_t',
                                    fun=np.sum,
                                    ref=self.SynGen,
                                    info='Summation of M by COI index',
                                    )
        self.MtREGCVSG = NumReduce(u=self.MREGCVSG,
                                    tex_name='M_t',
                                    fun=np.sum,
                                    ref=self.REGCVSG,
                                    info='Summation of M by COI index',
                                    )

        self.Mt = ConstService(tex_name='M_t',
                                info='Inertia weights',
                                v_str='MtSynGen + MtREGCVSG')

        self.MrSynGen = NumRepeat(u=self.MtSynGen,
                                    tex_name='M_{tr}',
                                    ref=self.SynGen,
                                    info='Repeated summation of Mt',
                                    )
        self.MrREGCVSG = NumRepeat(u=self.MtREGCVSG,
                                    tex_name='M_{tr}',
                                    ref=self.REGCVSG,
                                    info='Repeated summation of Mt',
                                    )

        self.Mr = NumRepeat(u=self.Mt,
                                    tex_name='M_{tr}',
                                    ref=self.SynGen,
                                    info='Repeated summation of Mt',
                                    )

        # self.Mr = ConstService(tex_name='M_t',
        #                         info='Inertia weights',
        #                         v_str='MrSynGen + MrREGCVSG')

        self.MwSynGen = ConstService(tex_name='M_w',
                                    info='Inertia weights',
                                    v_str='MSynGen / MrSynGen')
        self.MwREGCVSG = ConstService(tex_name='M_w',
                                    info='Inertia weights',
                                    v_str='MREGCVSG / MrREGCVSG')

        self.MrSynGen2 = NumReduce(u=self.MwSynGen,
                                    tex_name='M_t',
                                    fun=np.mean,
                                    ref=self.SynGen,
                                    info='Summation of M by COI index',
                                    )
        self.MrREGCVSG2 = NumReduce(u=self.MwREGCVSG,
                                    tex_name='M_t',
                                    fun=np.mean,
                                    ref=self.REGCVSG,
                                    info='Summation of M by COI index',
                                    )

        self.d0wSynGen = ConstService(tex_name=r'\delta_{gen,0,w}',
                                        v_str='d0SynGen * MwSynGen',
                                        info='Linearly stored weighted delta')
        self.d0wREGCVSG = ConstService(tex_name=r'\delta_{gen,0,w}',
                                        v_str='d0REGCVSG * MwREGCVSG',
                                        info='Linearly stored weighted delta')

        self.a0wSynGen = ConstService(tex_name=r'\omega_{gen,0,w}',
                                        v_str='a0SynGen * MwSynGen',
                                        info='Linearly stored weighted omega')
        self.a0wREGCVSG = ConstService(tex_name=r'\omega_{gen,0,w}',
                                        v_str='a0REGCVSG * MwREGCVSG',
                                        info='Linearly stored weighted omega')

        self.d0aSynGen = NumReduce(u=self.d0wSynGen,
                                    tex_name=r'\delta_{gen,0,avg}',
                                    fun=np.sum,
                                    ref=self.SynGen,
                                    info='Average initial delta',
                                    cache=False,
                                    )
        self.d0aREGCVSG = NumReduce(u=self.d0wREGCVSG,
                                    tex_name=r'\delta_{gen,0,avg}',
                                    fun=np.sum,
                                    ref=self.REGCVSG,
                                    info='Average initial delta',
                                    cache=False,
                                    )

        self.a0aSynGen = NumReduce(u=self.a0wSynGen,
                                    tex_name=r'\omega_{gen,0,avg}',
                                    fun=np.sum,
                                    ref=self.SynGen,
                                    info='Average initial omega',
                                    cache=False,
                                    )
        self.a0aREGCVSG = NumReduce(u=self.a0wREGCVSG,
                                    tex_name=r'\omega_{gen,0,avg}',
                                    fun=np.sum,
                                    ref=self.REGCVSG,
                                    info='Average initial omega',
                                    cache=False,
                                    )

        self.d0a = ConstService(tex_name=r'\omega_{gen,0,w}',
                                v_str='d0aSynGen + d0aREGCVSG',
                                info='Linearly stored weighted omega')
        self.a0a = ConstService(tex_name=r'\omega_{gen,0,w}',
                                v_str='a0aSynGen + a0aREGCVSG',
                                info='Linearly stored weighted omega')

        # self.pidx = IdxRepeat(u=self.idx, ref=self.SynGen, info='Repeated COI.idx')

        # Note:
        # Even if d(omega) /d (omega) = 1, it is still stored as a lambda function.
        # When no SynGen is referencing any COI, j_update will not be called,
        # and Jacobian will become singular. `diag_eps = True` needs to be used.

        # Note:
        # Do not assign `v_str=1` for `omega`. Otherwise, COIs with no connected generators will
        # fail to initialize.
        self.omega = Algeb(tex_name=r'\omega_{coi}',
                           info='COI speed',
                           v_str='a0a',
                           v_setter=True,
                           e_str='-omega',
                           diag_eps=True,
                           )
        self.delta = Algeb(tex_name=r'\delta_{coi}',
                           info='COI rotor angle',
                           v_str='d0a',
                           v_setter=True,
                           e_str='-delta',
                           diag_eps=True,
                           )

        # Note:
        # `omega_sub` or `delta_sub` must not provide `v_str`.
        # # Otherwise, values will be incorrectly summed for `omega` and `delta`.
        # self.omega_sub = ExtAlgeb(model='COI',
        #                           src='omega',
        #                           e_str='Mw * wgen',
        #                           indexer=self.pidx,
        #                           info='COI frequency contribution of each generator'
        #                           )
        # self.delta_sub = ExtAlgeb(model='COI',
        #                           src='delta',
        #                           e_str='Mw * agen',
        #                           indexer=self.pidx,
        #                           info='COI angle contribution of each generator'
        #                           )

    def set_in_use(self):
        """
        Set the ``Model.in_use`` flag based on ``len(self.SynGenIdx.v)``.
        """
        self.in_use = (len(self.SynGenIdx.v) + len(self.REGCVSGIdx.v) > 0)

class COI2(COI2Data, COI2Model):
    """
    Center of inertia calculation class.
    """

    def __init__(self, system, config):
        COI2Data.__init__(self)
        COI2Model.__init__(self, system, config)
