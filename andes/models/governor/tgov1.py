from andes.core import (Algeb, AntiWindup, ConstService, LagAntiWindup,
                        LeadLag, NumParam, State,)
from andes.core.block import DeadBand1
from andes.models.governor.tgbase import TGBase, TGBaseData


class TGOV1Data(TGBaseData):
    def __init__(self):
        super().__init__()
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.VMAX = NumParam(info='Maximum valve position',
                             tex_name='V_{max}',
                             unit='p.u.',
                             default=1.2,
                             power=True,
                             )
        self.VMIN = NumParam(info='Minimum valve position',
                             tex_name='V_{min}',
                             unit='p.u.',
                             default=0.0,
                             power=True,
                             )

        self.T1 = NumParam(info='Valve time constant',
                           default=0.1,
                           tex_name='T_1')
        self.T2 = NumParam(info='Lead-lag lead time constant',
                           default=0.2,
                           tex_name='T_2')
        self.T3 = NumParam(info='Lead-lag lag time constant',
                           default=10.0,
                           tex_name='T_3')
        self.Dt = NumParam(info='Turbine damping coefficient',
                           default=0.0,
                           tex_name='D_t',
                           power=True,
                           )


class TGOV1DBData(TGOV1Data):
    def __init__(self):
        TGOV1Data.__init__(self)
        self.dbL = NumParam(info='Lower bound of deadband',
                            tex_name='db_L',
                            default=0.0,
                            unit='p.u.',
                            )
        self.dbU = NumParam(info='Upper bound of deadband',
                            tex_name='db_U',
                            default=0.0,
                            unit='p.u.',
                            )


class TGOV1Model(TGBase):
    """
    Implement TGOV1 model.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.gain = ConstService(v_str='ue/R',
                                 tex_name='G',
                                 )

        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='pref0 * R - pref',
                          )

        self.wd = Algeb(info='Generator speed deviation',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (omega - wref) - wd',
                        )
        self.pd = Algeb(info='Pref plus speed deviation times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='ue * tm0',
                        e_str='ue*(- wd + pref + paux) * gain - pd')

        self.LAG = LagAntiWindup(u=self.pd,
                                 K=1,
                                 T=self.T1,
                                 lower=self.VMIN,
                                 upper=self.VMAX,
                                 )
        self.LL = LeadLag(u=self.LAG_y,
                          T1=self.T2,
                          T2=self.T3,
                          )
        self.pout.e_str = 'ue * (LL_y - Dt * wd) - pout'


class TGOV1NModel(TGOV1Model):
    """
    New TGOV1 model with `pref` and `paux` summed after the gain.
    """

    def __init__(self, system, config):
        TGOV1Model.__init__(self, system, config)
        self.pref.v_str = 'tm0'
        self.pref.e_str = 'pref0 - pref'

        self.pd.e_str = 'ue*(-wd * gain + pref + paux) - pd'


class TGOV1DBModel(TGOV1Model):
    """
    Model TGOV1 with deadband.
    """

    def __init__(self, system, config):
        TGOV1Model.__init__(self, system, config)
        self.DB = DeadBand1(u=self.wd, center=0.0, lower=self.dbL,
                            upper=self.dbU, tex_name='DB',
                            info='deadband for speed deviation',
                            )
        self.pd.e_str = 'ue * (-DB_y + pref + paux) * gain - pd'
        self.pout.e_str = '(LL_y - Dt * DB_y) - pout'


class TGOV1NDBModel(TGOV1DBModel):
    """
    Implementation of TGOV1NDB
    """

    def __init__(self, system, config):
        TGOV1DBModel.__init__(self, system, config)
        self.pref.v_str = 'tm0'
        self.pref.e_str = 'pref0 - pref'

        self.pd.e_str = 'ue*(DB_y * gain + pref + paux) - pd'


class TGOV1ModelAlt(TGBase):
    """
    An alternative implementation of TGOV1 from equations
    (without using Blocks).
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='pref0 * R - pref',
                          )
        self.wd = Algeb(info='Generator speed deviation',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (omega - wref) - wd',
                        )
        self.pd = Algeb(info='Pref plus speed deviation times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='tm0',
                        e_str='(- wd + pref + paux) * gain - pd')

        self.LAG_y = State(info='State in lag transfer function',
                           tex_name=r"x'_{LAG}",
                           e_str='LAG_lim_zi * (1 * pd - LAG_y)',
                           t_const=self.T1,
                           v_str='pd',
                           )
        self.LAG_lim = AntiWindup(u=self.LAG_y,
                                  lower=self.VMIN,
                                  upper=self.VMAX,
                                  tex_name='lim_{lag}',
                                  )
        self.LL_x = State(info='State in lead-lag transfer function',
                          tex_name="x'_{LL}",
                          v_str='LAG_y',
                          e_str='(LAG_y - LL_x)',
                          t_const=self.T3
                          )
        self.LL_y = Algeb(info='Lead-lag Output',
                          tex_name='y_{LL}',
                          v_str='LAG_y',
                          e_str='T2 / T3 * (LAG_y - LL_x) + LL_x - LL_y',
                          )

        self.pout.e_str = 'ue * (LL_y - Dt * wd) - pout'


class TGOV1(TGOV1Data, TGOV1Model):
    """
    TGOV1 turbine governor model.

    Implements the PSS/E TGOV1 model without deadband.
    """

    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1Model.__init__(self, system, config)


class TGOV1N(TGOV1Data, TGOV1NModel):
    """
    New TGOV1 (TGOV1N) turbine governor model.

    The TGOV1N model that sums ``pref`` and ``paux`` signals after the droop.
    This model is useful for incorporating AGC and scheduling signals, which
    will not be multiplied by ``1/R`` like in the original TGOV1 model.

    Scheduling changes should write to ``pref0.v`` in place. AGC signal should
    write to ``paux0.v`` in place.

    Modifying ``tm0`` is not allowed.

    Examples
    --------
    To update all ``paux0`` values to ``paux_new``, which contains the new
    values, do

    .. code:: python

        ss.TGOV1N.paux0.v[:] = paux_new  # in-place update of the `paux0.v` array

    instead of

    .. code:: python

        ss.TGOV1N.paux0.v = paux_new  # error; changes the reference of `paux0.v`

    """

    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1NModel.__init__(self, system, config)


class TGOV1DB(TGOV1DBData, TGOV1DBModel):
    """
    TGOV1 turbine governor model with speed input deadband.
    """

    def __init__(self, system, config):
        TGOV1DBData.__init__(self)
        TGOV1DBModel.__init__(self, system, config)


class TGOV1NDB(TGOV1DBData, TGOV1NDBModel):
    """
    TGOV1N turbine governor model with speed input deadband.
    """

    def __init__(self, system, config):
        TGOV1DBData.__init__(self)
        TGOV1NDBModel.__init__(self, system, config)
