"""
Classes for Center of Inertia calculation.
Added VSG into COI calculation.
"""
import numpy as np

from andes.core.param import ExtParam
from andes.core.service import NumRepeat, IdxRepeat, BackRef
from andes.core.service import NumReduce, RefFlatten, ExtService, ConstService
from andes.core.var import ExtState, Algeb, ExtAlgeb
from andes.core.model import ModelData, Model


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

        self.COIlist = ['SynGen', 'REGCVSG']
        
        for RefModel in self.COIlist:

            Gen = BackRef(info='Back reference to Gen idx')

            GenIdx = RefFlatten(ref=Gen)

            M = ExtParam(model=RefModel, src='M',
                         indexer=GenIdx, export=False,
                         info='Linearly stored Gen.M',
                         )

            wgen = ExtState(model=RefModel,
                            src='omega',
                            indexer=GenIdx,
                            info='Linearly stored Gen.omega',
                            )

            agen = ExtState(model=RefModel,
                            src='delta',
                            indexer=GenIdx,
                            info='Linearly stored Gen.delta',
                            )

            d0 = ExtService(model=RefModel,
                            src='delta',
                            indexer=GenIdx,
                            info='Linearly stored initial delta',
                            )

            a0 = ExtService(model='SynGen',
                            src='omega',
                            indexer=self.SynGenIdx,
                            info='Linearly stored initial omega',
                            )

            Mt = NumReduce(u=self.M,
                           fun=np.sum,
                           ref=self.SynGen,
                           info='Summation of M by COI index',
                           )

            Mr = NumRepeat(u=self.Mt,
                           ref=self.SynGen,
                           info='Repeated summation of M',
                           )

            Mw = ConstService(info='Inertia weights',
                              v_str='M/Mr')

            d0w = ConstService(v_str='d0 * Mw',
                               info='Linearly stored weighted delta')

            a0w = ConstService(v_str='a0 * Mw',
                               info='Linearly stored weighted omega')

            d0a = NumReduce(u=d0w,
                            fun=np.sum,
                            ref=self.SynGen,
                            info='Average initial delta',
                            cache=False,
                            )

            a0a = NumReduce(u=a0w,
                            fun=np.sum,
                            ref=self.SynGen,
                            info='Average initial omega',
                            cache=False,
                            )

        # self.pidx = IdxRepeat(u=self.idx, ref=self.SynGen, info='Repeated COI.idx')

        # # Note:
        # # Even if d(omega) /d (omega) = 1, it is still stored as a lambda function.
        # # When no SynGen is referencing any COI, j_update will not be called,
        # # and Jacobian will become singular. `diag_eps = True` needs to be used.

        # # Note:
        # # Do not assign `v_str=1` for `omega`. Otherwise, COIs with no connected generators will
        # # fail to initialize.
        # self.omega = Algeb(tex_name=r'\omega_{coi}',
        #                    info='COI speed',
        #                    v_str='a0a',
        #                    v_setter=True,
        #                    e_str='-omega',
        #                    diag_eps=True,
        #                    )
        # self.delta = Algeb(tex_name=r'\delta_{coi}',
        #                    info='COI rotor angle',
        #                    v_str='d0a',
        #                    v_setter=True,
        #                    e_str='-delta',
        #                    diag_eps=True,
        #                    )

        # # Note:
        # # `omega_sub` or `delta_sub` must not provide `v_str`.
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
        self.in_use = (len(self.SynGenIdx.v) > 0)


class COI2(COI2Data, COI2Model):
    """
    Center of inertia calculation class.
    """

    def __init__(self, system, config):
        COI2Data.__init__(self)
        COI2Model.__init__(self, system, config)
