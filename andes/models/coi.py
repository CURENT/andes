"""
Classes for Center of Inertia calculation.
"""
import numpy as np

from andes.core.param import RefParam, ExtParam
from andes.core.service import NumRepeater, IdxRepeater, ReducerService, FlattenService, ExtService
from andes.core.var import ExtState, Algeb, ExtAlgeb
from andes.core.model import ModelData, Model


class COIData(ModelData):
    """COI parameter data"""

    def __init__(self):
        ModelData.__init__(self)
        self.SynGen = RefParam(info='SynGen idx lists', export=False, )


class COIModel(Model):

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Calculation'
        self.flags.update({'tds': True})

        self.SynGenIdx = FlattenService(ref=self.SynGen)

        self.M = ExtParam(model='SynGen', src='M',
                          indexer=self.SynGenIdx, export=False,
                          info='Linearlly stored SynGen.M',
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
        self.ang0 = ExtService(model='SynGen',
                               src='delta',
                               indexer=self.SynGenIdx,
                               tex_name=r'\delta_{gen,0}',
                               info='Linearly stored initial delta',
                               )
        self.ang0avg = ReducerService(u=self.ang0,
                                      tex_name=r'\delta{gen,0,avg}',
                                      fun=np.average,
                                      ref=self.SynGen,
                                      info='Average initial rotor angle',
                                      )

        self.Mt = ReducerService(u=self.M,
                                 tex_name='M_t',
                                 fun=np.sum,
                                 ref=self.SynGen,
                                 info='Summation of M by COI index',
                                 )

        self.Mtr = NumRepeater(u=self.Mt,
                               tex_name='M_{tr}',
                               ref=self.SynGen,
                               info='Repeated summation of M',
                               )

        self.pidx = IdxRepeater(u=self.idx, ref=self.SynGen)

        # Note: even if d(omega) /d (omega) = 1, it is still stored as a lambda function.
        #       When no SynGen is referencing any COI, to avoid singular Jacobian,
        #       one needs to use `diag_eps`.

        # `wcoi` must have `v_setter = True`, otherwise, values from `wcoi_sub` will be summed.
        self.omega = Algeb(tex_name=r'\omega_{coi}',
                           info='COI speed',
                           v_str='1',
                           v_setter=True,
                           e_str='-omega',
                           diag_eps=1e-6,
                           )
        self.delta = Algeb(tex_name=r'\delta_{coi}',
                           info='COI rotor angle',
                           v_str='ang0avg',
                           v_setter=True,
                           e_str='-delta',
                           diag_eps=1e-6,
                           )

        self.omega_sub = ExtAlgeb(model='COI',
                                  src='omega',
                                  e_str='M * wgen / Mtr',
                                  indexer=self.pidx,
                                  info='COI frequency contribution of each generator'
                                  )
        self.delta_sub = ExtAlgeb(model='COI',
                                  src='delta',
                                  e_str='M * agen / Mtr',
                                  indexer=self.pidx,
                                  info='COI angle contribution of each generator'
                                  )

    @property
    def in_use(self):
        return len(self.SynGenIdx.v) > 0


class COI(COIData, COIModel):
    """
    Center of inertia calculation class.
    """

    def __init__(self, system, config):
        COIData.__init__(self)
        COIModel.__init__(self, system, config)
