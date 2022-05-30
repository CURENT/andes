"""
Base class for Turbine Governor models.
"""

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        ExtState, IdxParam, Model, ModelData, NumParam,)
from andes.core.service import NumSelect


class TGBaseData(ModelData):
    """
    Base data for turbine governors.
    """

    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            unique=True,
                            )
        self.Tn = NumParam(info='Turbine power rating. Equal to `Sn` if not provided.',
                           tex_name='T_n',
                           unit='MVA',
                           default=0.0,
                           )
        self.wref0 = NumParam(info='Base speed reference',
                              tex_name=r'\omega_{ref0}',
                              default=1.0,
                              unit='p.u.',
                              )


class TGBase(Model):
    """
    Base Turbine Governor model.

    Parameters
    ----------
    add_sn : bool
        True to add ``NumSelect`` Sn; False to add later in custom models.
        This is useful when the governor connects to two generators.
    add_tm0 : bool
        True to add ``ExtService`` ``tm0``.

    """

    def __init__(self, system, config, add_sn=True, add_tm0=True):
        Model.__init__(self, system, config)
        self.group = 'TurbineGov'
        self.flags.update({'tds': True})
        self.Sg = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_n',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.ug = ExtParam(src='u',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='u_g',
                           info='Generator connection status',
                           unit='bool',
                           export=False,
                           )
        self.ue = ConstService(v_str='u * ug',
                               info="effective connection status considering generator's",
                               tex_name='u_{e}',
                               )

        if add_sn is True:
            self.Sn = NumSelect(self.Tn,
                                fallback=self.Sg,
                                tex_name='S_n',
                                info='Turbine or Gen rating',
                                )

        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_n',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )

        # Note: changing the values of `tm0` is not allowed at any time!!
        if add_tm0 is True:
            self.tm0 = ExtService(src='tm',
                                  model='SynGen',
                                  indexer=self.syn,
                                  tex_name=r'\tau_{m0}',
                                  info='Initial mechanical input')
            self.pref0 = ConstService(v_str='tm0', info='initial pref',
                                      tex_name='P_{ref0}',
                                      )

        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              unit='p.u.'
                              )

        # Note: changing `paux0` is allowed.
        # It is a way how one can input from external programs such as reinforcement learning.
        self.paux0 = ConstService(v_str='0',
                                  tex_name='P_{aux0}',
                                  info='const. auxiliary input')

        self.tm = ExtAlgeb(src='tm',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'\tau_m',
                           e_str='ue * (pout - tm0)',
                           info='Mechanical power interface to SynGen',
                           ename='tm',
                           tex_ename=r'\tau_{m}',
                           )
        # `paux` must be zero upon initialization
        self.paux = Algeb(info='Auxiliary power input',
                          tex_name='P_{aux}',
                          v_str='paux0',
                          e_str='paux0 - paux',
                          )
        self.pout = Algeb(info='Turbine final output power',
                          tex_name='P_{out}',
                          v_str='ue * tm0',
                          )
        self.wref = Algeb(info='Speed reference variable',
                          tex_name=r'\omega_{ref}',
                          v_str='wref0',
                          e_str='wref0 - wref',
                          )
