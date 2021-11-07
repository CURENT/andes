"""
Switched shunt model.
"""

import ast
from collections import OrderedDict

import numpy as np

from andes.core import NumParam, ConstService
from andes.core.discrete import ShuntAdjust
from andes.core.service import SwBlock
from andes.models.shunt.shunt import ShuntData, ShuntModel


class ShuntSwData(ShuntData):
    """
    Data for switched shunts.
    """

    def __init__(self):
        ShuntData.__init__(self)
        self.gs = NumParam(info='a list literal of switched conductances blocks',
                           default=0.0,
                           unit='p.u.',
                           vtype=object,
                           iconvert=list_iconv,
                           oconvert=list_oconv,
                           y=True,
                           )

        self.bs = NumParam(info='a list literal of switched susceptances blocks',
                           default=0.0,
                           unit='p.u.',
                           vtype=object,
                           iconvert=list_iconv,
                           oconvert=list_oconv,
                           y=True,
                           )

        self.ns = NumParam(info='a list literal of the element numbers in each switched block',
                           default=[0],
                           vtype=object,
                           iconvert=list_iconv,
                           oconvert=list_oconv,
                           )

        self.vref = NumParam(info='voltage reference',
                             default=1.0,
                             unit='p.u.',
                             non_zero=True,
                             non_negative=True,
                             )

        self.dv = NumParam(info='voltage error deadband',
                           default=0.05,
                           unit='p.u.',
                           non_zero=True,
                           non_negative=True,
                           )

        self.dt = NumParam(info='delay before two consecutive switching',
                           default=30.,
                           unit='seconds',
                           non_negative=True,
                           )


def list_iconv(x):
    """
    Helper function to convert a list literal into a numpy array.
    """
    if isinstance(x, str):
        x = ast.literal_eval(x)
    if isinstance(x, (int, float)):
        if not np.isnan(x):
            x = [x]
        else:
            return None
    if isinstance(x, list):
        x = np.array(x)
    return x


def list_oconv(x):
    """
    Convert list into a list literal.
    """
    return np.array2string(x, separator=', ')


class ShuntSwModel(ShuntModel):
    """
    Switched shunt model.
    """

    def __init__(self, system, config):
        ShuntModel.__init__(self, system, config)

        self.config.add(OrderedDict((('min_iter', 2),
                                     ('err_tol', 0.01),
                                     )))
        self.config.add_extra("_help",
                              min_iter="iteration number starting from which to enable switching",
                              err_tol="iteration error below which to enable switching",
                              )
        self.config.add_extra("_alt",
                              min_iter='int',
                              err_tol='float',
                              )
        self.config.add_extra("_tex",
                              min_iter="sw_{iter}",
                              err_tol=r"\epsilon_{tol}",
                              )

        self.beff = SwBlock(init=self.b, ns=self.ns, blocks=self.bs)
        self.geff = SwBlock(init=self.g, ns=self.ns, blocks=self.gs,
                            ext_sel=self.beff)

        self.vlo = ConstService(v_str='vref - dv', tex_name='v_{lo}')
        self.vup = ConstService(v_str='vref + dv', tex_name='v_{up}')

        self.adj = ShuntAdjust(v=self.v, lower=self.vlo, upper=self.vup,
                               bsw=self.beff, gsw=self.geff, dt=self.dt,
                               u=self.u,
                               min_iter=self.config.min_iter,
                               err_tol=self.config.err_tol,
                               info='shunt adjuster')

        self.a.e_str = 'u * v**2 * geff'
        self.v.e_str = '-u * v**2 * beff'


class ShuntSw(ShuntSwData, ShuntSwModel):
    """
    Switched Shunt Model.

    Parameters `gs`, `bs` and `bs` must be entered in string literals,
    comma-separated. They need to have the same length.

    For example, in the excel file, one can put ::

        gs = [0, 0]
        bs = [0.2, 0.2]
        ns = [2, 4]

    To use individual shunts as fixed shunts, set the corresponding
    `ns = 0` or `ns = [0]`.

    The effective shunt susceptances and conductances are stored in
    services `beff` and `geff`.
    """

    def __init__(self, system=None, config=None):
        ShuntSwData.__init__(self)
        ShuntSwModel.__init__(self, system, config)
