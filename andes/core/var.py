from typing import Optional, Union, List

import numpy as np
from andes.core.param import DataParam
from numpy import ndarray


class VarBase(object):
    """
    Variable class
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 descr: Optional[str] = None,
                 unit: Optional[str] = None,
                 **kwargs
                 ):

        self.name = name
        self.descr = descr
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        self.owner = None

        self.n = 0
        self.a: Optional[Union[ndarray, List]] = None
        self.v: Optional[ndarray] = None
        self.e: Optional[ndarray] = None

        self.e_symbolic = None
        self.e_lambdify = None
        self.e_numeric = None

        # metadata
        self.property = {"intf": False,
                         "ext": False,
                         "limit": False,
                         "windup": False,
                         "anti_windup": False,
                         "deadband": False}

    def set_address(self, addr):
        self.a = addr
        self.n = len(self.a)
        self.v = np.zeros(self.n)
        self.e = np.zeros(self.n)


class Algeb(VarBase):
    """
    Algebraic variable
    """
    pass


class State(VarBase):
    pass


class Calc(VarBase):
    pass


class ExtVar(VarBase):
    def __init__(self,
                 model: str,
                 src: str,
                 indexer: Optional[Union[List, ndarray, DataParam]] = None,
                 *args,
                 **kwargs):
        super(ExtVar, self).__init__(*args, **kwargs)
        self.initialized = False
        self.model = model
        self.src = src
        self.indexer = indexer

        self.parent_model = None
        self.parent_instance = None
        self.uid = None

    def set_external(self, ext_model):
        self.parent_model = ext_model
        self.parent_instance = ext_model.__dict__[self.src]
        self.uid = ext_model.idx2uid(self.indexer.v)

        n_indexer = len(self.indexer.v)
        if n_indexer == 0:
            return

        # pull in values
        self.a = self.parent_instance.a[self.uid]
        self.n = len(self.a)

        # set initial v and e values to zero
        self.v = np.zeros(self.n)
        self.e = np.zeros(self.n)
