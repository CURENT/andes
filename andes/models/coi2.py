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

        COIModelName = ["SynGen", "REGCVSG"]

        for RefModelName in COIModelName:

            RefModel = BackRef(info='Back reference to RefModel idx')
            setattr(self, "{}".format(RefModelName), RefModel)

            Idx = RefFlatten(ref=getattr(self, RefModelName))
            setattr(self, "Idx{}".format(RefModelName), Idx)

            M = ExtParam(model=RefModelName, src='M',
                         indexer=getattr(self, 'Idx'+RefModelName), export=False,
                         info='Linearly stored '+'RefModelName'+'.M',
                         )
            setattr(self, "M{}".format(RefModelName), M)

            wgen = ExtState(model=RefModelName,
                            src='omega',
                            indexer=getattr(self, 'Idx'+RefModelName),
                            tex_name=r'\omega_{gen}',
                            info='Linearly stored RefModel.omega',
                            )
            setattr(self, "wgen{}".format(RefModelName), wgen)

            agen = ExtState(model=RefModelName,
                            src='delta',
                            indexer=getattr(self, 'Idx'+RefModelName),
                            tex_name=r'\delta_{gen}',
                            info='Linearly stored RefModel.delta',
                            )
            setattr(self, "agen{}".format(RefModelName), agen)

            d0 = ExtService(model=RefModelName,
                             src='delta',
                             indexer=getattr(self, 'Idx'+RefModelName),
                             tex_name=r'\delta_{gen,0}',
                             info='Linearly stored initial delta',
                             )
            setattr(self, "d0{}".format(RefModelName), d0)

            a0 = ExtService(model=RefModelName,
                             src='omega',
                             indexer=getattr(self, 'Idx'+RefModelName),
                             tex_name=r'\omega_{gen,0}',
                             info='Linearly stored initial omega',
                             )
            setattr(self, "a0{}".format(RefModelName), a0)



        """
        Past initial state
        """
        self.Mt = ConstService(tex_name='M_w',
                               info='Summation of M by COI index for all RefModel',
                               v_str='',
                               )
        for RefModelName in COIModelName:

            self.Mt.v_str += '+ M'+RefModelName

        for RefModelName in COIModelName:

            Mr = NumRepeat(u=self.Mt,
                            tex_name='M_{tr}',
                            ref=getattr(self, RefModelName),
                            info='Repeated summation of M',
                            )
            setattr(self, "Mr{}".format(RefModelName), Mr)

            Mw = ConstService(tex_name='M_w',
                              info='Inertia weights',
                              v_str='M'+RefModelName+' / '+'Mr'+RefModelName,
                              )
            setattr(self, "Mw{}".format(RefModelName), Mw)

            d0w = ConstService(tex_name=r'\delta_{gen,0,w}',
                                info='Linearly stored weighted delta',
                                v_str='d0'+RefModelName+' * '+'Mw'+RefModelName,
                                )
            setattr(self, "d0w{}".format(RefModelName), d0w)

            a0w = ConstService(tex_name=r'\omega_{gen,0,w}',
                                info='Linearly stored weighted omega',
                                v_str='a0'+RefModelName+' * '+'Mw'+RefModelName,
                                )
            setattr(self, "a0w{}".format(RefModelName), a0w)

            d0a = NumReduce(u=getattr(self, 'd0w'+RefModelName),
                             tex_name=r'\delta_{gen,0,avg}',
                             fun=np.sum,
                             ref=getattr(self, RefModelName),
                             info='Average initial delta',
                             cache=False,
                             )
            setattr(self, "d0a{}".format(RefModelName), d0a)

            a0a = NumReduce(u=getattr(self, 'a0w'+RefModelName),
                             tex_name=r'\omega_{gen,0,avg}',
                             fun=np.sum,
                             ref=getattr(self, RefModelName),
                             info='Average initial omega',
                             cache=False,
                             )
            setattr(self, "a0a{}".format(RefModelName), a0a)

            pidx = IdxRepeat(u=getattr(self, 'Idx'+RefModelName), 
                             ref=getattr(self, RefModelName),
                             info='Repeated COI.idx')
            setattr(self, "pidx{}".format(RefModelName), pidx)

        # Note:
        # Even if d(omega) /d (omega) = 1, it is still stored as a lambda function.
        # When no SynGen is referencing any COI, j_update will not be called,
        # and Jacobian will become singular. `diag_eps = True` needs to be used.

        # Note:
        # Do not assign `v_str=1` for `omega`. Otherwise, COIs with no connected generators will
        # fail to initialize.
        self.omega = Algeb(tex_name=r'\omega_{coi}',
                        info='COI speed',
                        v_str='',
                        v_setter=True,
                        e_str='',
                        diag_eps=True,
                        )
        for RefModelName in COIModelName:                        
            self.omega.v_str = ' + a0a'+RefModelName
            self.omega.e_str = '- omega'+RefModelName           

        self.delta = Algeb(tex_name=r'\delta_{coi}',
                        info='COI rotor angle',
                        v_str='d0a'+RefModelName,
                        v_setter=True,
                        e_str='-delta'+RefModelName,
                        diag_eps=True,
                        )
        for RefModelName in COIModelName:                        
            self.delta.v_str = ' + d0a'+RefModelName
            self.delta.e_str = '- delta'+RefModelName    

            # Note:
            # `omega_sub` or `delta_sub` must not provide `v_str`.
            # Otherwise, values will be incorrectly summed for `omega` and `delta`.
            # self.omega_sub = ExtAlgeb(model='COI',
            #                         src='omega',
            #                         e_str='Mw * wgen',
            #                         indexer=self.pidx,
            #                         info='COI frequency contribution of each generator'
            #                         )
            # setattr(self, "delta{}".format(RefModelName), delta)

            # self.delta_sub = ExtAlgeb(model='COI',
            #                         src='delta',
            #                         e_str='Mw * agen',
            #                         indexer=self.pidx,
            #                         info='COI angle contribution of each generator'
            #                         )
            # setattr(self, "delta{}".format(RefModelName), delta)     

    def set_in_use(self):
        """
        Set the ``Model.in_use`` flag based on ``len(self.SynGenIdx.v)``.
        """
        lentotal = 0
        for RefModelName in COIModelName:
            len_total += len(self.getattr(self, RefModelName+'.v'))
        self.in_use = (len_total > 0)

class COI2(COI2Data, COI2Model):
    """
    Center of inertia calculation class.
    """

    def __init__(self, system, config):
        COI2Data.__init__(self)
        COI2Model.__init__(self, system, config)
