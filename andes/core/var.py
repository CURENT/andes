#  [ANDES] (C)2015-2020 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: var.py
#  Last modified: 8/16/20, 7:27 PM

from typing import Optional, Union, List
from andes.core.param import BaseParam
from andes.core.common import DummyValue
from andes.core.discrete import Discrete
from andes.core.service import BaseService
from andes.models.group import GroupBase
from andes.shared import np, ndarray


class BaseVar(object):
    """
    Base variable class.

    Derived classes `State` and `Algeb` should be used to build
    model variables.

    Parameters
    ----------
    name : str, optional
        Variable name
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable name. If is None, use `name`
        instead.
    discrete : Discrete
        Associated discrete component. Will call `check_var` on
        the discrete component.

    Attributes
    ----------
    a : array-like
        variable address
    v : array-like
        local-storage of the variable value
    e : array-like
        local-storage of the corresponding equation value
    e_str : str
        the string/symbolic representation of the equation
    """
    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[Union[str, float]] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 discrete: Optional[Discrete] = None,
                 v_setter: Optional[bool] = False,
                 e_setter: Optional[bool] = False,
                 addressable: Optional[bool] = True,
                 export: Optional[bool] = True,
                 diag_eps: Optional[float] = 0.0,
                 ):

        self.name = name
        self.info = info
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        self.owner = None  # instance of the owner Model
        self.id = None     # variable internal index inside a model (assigned in run time)

        self.v_str = v_str  # equation string (v = v_str) for variable initialization
        self.v_iter = v_iter  # the implicit equation (0 = v_iter) for iterative initialization
        self.e_str = e_str  # string for symbolic equation

        self.discrete = discrete
        self.v_setter = v_setter  # True if this variable sets the variable value
        self.e_setter = e_setter  # True if this var sets the equation value
        self.addressable = addressable  # True if this var needs to be assigned an address FIXME: not in use
        self.export = export            # True if this var's value needs to exported
        self.diag_eps = diag_eps        # small diagonal value to be added to `dae.gy`

        # attributes assigned by `set_address`
        self.n = 0
        self.a: ndarray = np.array([], dtype=int)       # address array
        self.v: ndarray = np.array([], dtype=np.float)  # variable value array
        self.e: ndarray = np.array([], dtype=np.float)  # equation value array

        self.av: ndarray = np.array([], dtype=int)      # FIXME: future var. address array
        self.ae: ndarray = np.array([], dtype=int)      # FIXME: future equation address array

        # internal flags
        # NOTE:
        # contiguous is only True for internal variables of models with flag ``collate`` equal to ``False``.
        self._contiguous = False  # True if if address is contiguous to allow slicing into arrays without copy.

        self.e_inplace = False    # True if `self.e` is in-place access to `System.dae.__dict__[self.e_code]`
        self.v_inplace = False    # True if `self.v` is in-place access to `System.dae.__dict__[self.v_code]`
        self.allow_none = False   # True to allow None in address (NOT IN USE)

    def reset(self):
        """
        Reset the internal numpy arrays and flags.
        """
        self.n = 0
        self.a[:] = 0
        self.v[:] = 0
        self.e[:] = 0
        self.av[:] = 0
        self.ae[:] = 0

        self._contiguous = False
        self.e_inplace = False
        self.v_inplace = False

    def __repr__(self):
        if self.n == 0:
            span = []

        elif 1 <= self.n <= 20:
            span = f'a={self.a}, v={self.v}, e={self.e}'

        else:
            span = []
            if not isinstance(self, ExtVar):
                span.append(self.a[0])
                span.append(self.a[-1])
                span.append(self.a[1] - self.a[0])
                span = ':'.join([str(i) for i in span])
                span = 'a=[' + span + ']'

        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}, {span}'

    def set_address(self, addr: ndarray, contiguous=False):
        """
        Set the address of internal variables.

        Parameters
        ----------
        addr : ndarray
            The assigned address for this variable
        contiguous : bool, optional
            If the addresses are contiguous
        """
        self.a = addr
        self.n = len(self.a)

        # NOT IN USE
        self.ae = np.array(self.a)
        self.av = np.array(self.a)
        # -----------

        self._contiguous = contiguous

        if self._contiguous:
            if self.e_setter is False:
                self.e_inplace = True

            if self.v_setter is False:
                self.v_inplace = True

    def set_arrays(self, dae):
        """
        Set the equation and values arrays.

        It slicing into DAE (when contiguous) or allocating new memory (when not contiguous).

        Parameters
        ----------
        dae : DAE
            Reference to System.dae
        """
        slice_idx = slice(self.a[0], self.a[-1] + 1)
        if self.v_inplace:
            self.v = dae.__dict__[self.v_code][slice_idx]
        else:
            self.v = np.zeros(self.n)

        if self.e_inplace:
            self.e = dae.__dict__[self.e_code][slice_idx]
        else:
            self.e = np.zeros(self.n)

    def get_names(self):
        return [self.name]

    @property
    def class_name(self):
        return self.__class__.__name__


class Algeb(BaseVar):
    """
    Algebraic variable class, an alias of the `BaseVar`.

    Attributes
    ----------
    e_code : str
        Equation code string, equals string literal ``g``
    v_code : str
        Variable code string, equals string literal ``y``
    """
    e_code = 'g'
    v_code = 'y'


class State(BaseVar):
    """
    Differential variable class, an alias of the `BaseVar`.

    Parameters
    ----------
    t_const : BaseParam, DummyValue
        Left-hand time constant for the differential equation.
        Time constants will not be evaluated as part of the differential equation.
        They will be collected to array `dae.Tf` to multiply to the right-hand side `dae.f`.

    Attributes
    ----------
    e_code : str
        Equation code string, equals string literal ``f``
    v_code : str
        Variable code string, equals string literal ``x``
    """
    e_code = 'f'
    v_code = 'x'

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[Union[str, float]] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 discrete: Optional[Discrete] = None,
                 t_const: Optional[Union[BaseParam, DummyValue, BaseService]] = None,
                 v_setter: Optional[bool] = False,
                 e_setter: Optional[bool] = False,
                 addressable: Optional[bool] = True,
                 export: Optional[bool] = True,
                 diag_eps: Optional[float] = 0.0,
                 ):
        BaseVar.__init__(self, name=name,
                         tex_name=tex_name,
                         info=info,
                         unit=unit,
                         v_str=v_str,
                         v_iter=v_iter,
                         e_str=e_str,
                         discrete=discrete,
                         v_setter=v_setter,
                         e_setter=e_setter,
                         addressable=addressable,
                         export=export,
                         diag_eps=diag_eps,
                         )
        self.t_const = t_const


class ExtVar(BaseVar):
    """
    Externally defined algebraic variable

    This class is used to retrieve the addresses of externally-
    defined variable. The `e` value of the `ExtVar` will be added
    to the corresponding address in the DAE equation.

    Parameters
    ----------
    model : str
        Name of the source model
    src : str
        Source variable name
    indexer : BaseParam, BaseService
        A parameter of the hosting model, used as indices into
        the source model and variable. If is None, the source
        variable address will be fully copied.
    allow_none : bool
        True to allow None in indexer

    Attributes
    ----------
    parent_model : Model
        The parent model providing the original parameter.
    uid : array-like
        An array containing the absolute indices into the
        parent_instance values.
    e_code : str
        Equation code string; copied from the parent instance.
    v_code : str
        Variable code string; copied from the parent instance.
    """
    def __init__(self,
                 model: str,
                 src: str,
                 indexer: Optional[Union[List, ndarray, BaseParam, BaseService]] = None,
                 allow_none: Optional[bool] = False,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[Union[str, float]] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 v_setter: Optional[bool] = False,
                 e_setter: Optional[bool] = False,
                 addressable: Optional[bool] = True,
                 export: Optional[bool] = True,
                 diag_eps: Optional[float] = 0.0,
                 ):
        super().__init__(name=name,
                         tex_name=tex_name,
                         info=info,
                         unit=unit,
                         v_str=v_str,
                         v_iter=v_iter,
                         e_str=e_str,
                         v_setter=v_setter,
                         e_setter=e_setter,
                         addressable=addressable,
                         export=export,
                         diag_eps=diag_eps,
                         )
        self.model = model
        self.src = src
        self.indexer = indexer
        self.allow_none = allow_none
        self.parent = None
        self._idx = None
        self._n = []
        self._n_count = 0

    @property
    def _v(self):
        out = []
        idx = 0
        for n in self._n:
            out.append(self.v[idx:idx+n])
            idx += n
        return out

    @property
    def _a(self):
        out = []
        idx = 0
        for n in self._n:
            out.append(self.a[idx:idx+n])
            idx += n
        return out

    def set_address(self, addr, contiguous=False):
        """
        Empty function.
        """
        pass

    def set_arrays(self, dae):
        """Empty function."""
        pass

    def link_external(self, ext_model):
        """
        Update variable addresses provided by external models

        This method sets attributes including `parent_model`,
        `parent_instance`, `uid`, `a`, `n`, `e_code` and
        `v_code`. It initializes the `e` and `v` to zero.

        Returns
        -------
        None

        Parameters
        ----------
        ext_model : Model
            Instance of the parent model

        Warnings
        --------
        `link_external` does not check if the ExtVar type is the same
        as the original variable to reduce performance overhead.
        It will be a silent error (a dimension too small error from `dae.build_pattern`)
        if a model uses `ExtAlgeb` to access a `State`, or vice versa.

        """

        self.parent = ext_model

        if isinstance(ext_model, GroupBase):
            if self.indexer.n > 0 and isinstance(self.indexer.v[0], (list, np.ndarray)):
                self._n = [len(i) for i in self.indexer.v]  # number of elements in each sublist
                self._idx = np.concatenate([np.array(i) for i in self.indexer.v])
            else:
                self._n = [len(self.indexer.v)]
                self._idx = self.indexer.v

            # use `0` for non-existent addresses (corr. to None in indexer)
            self.a = ext_model.get(src=self.src,
                                   idx=self._idx,
                                   attr='a',
                                   allow_none=self.allow_none,
                                   default=0,
                                   ).astype(int)
            self.n = len(self.a)

        else:
            original_var = ext_model.__dict__[self.src]

            if self.allow_none:
                raise NotImplementedError(f"{self.name}: allow_none not implemented for Model")

            if self.indexer is not None:
                uid = ext_model.idx2uid(self.indexer.v)
            else:
                uid = np.arange(ext_model.n, dtype=int)

            self._n = [len(uid)]
            if len(uid) > 0:
                self.a = original_var.a[uid]
            else:
                self.a = np.array([], dtype=int)

            # set initial v and e values to zero
            self.n = len(self.a)

        self.v = np.zeros(self.n)
        self.e = np.zeros(self.n)


class ExtState(ExtVar):
    """
    External state variable type.

    Warnings
    --------
    ``ExtState`` is not allowed to set ``t_const``, as it will conflict with the
    source ``State`` variable. In fact, one should not set ``e_str`` for ``ExtState``.
    """
    e_code = 'f'
    v_code = 'x'
    t_const = None


class ExtAlgeb(ExtVar):
    """
    External algebraic variable type.
    """
    e_code = 'g'
    v_code = 'y'


class AliasAlgeb(ExtAlgeb):
    """
    Alias algebraic variable. Essentially ``ExtAlgeb`` that links to a a model's
    own variable.

    ``AliasAlgeb`` is useful when the final output of a model is from a block, but
    the model must provide the final output in a pre-defined name.
    Using ``AliasAlgeb``, A model can avoid adding an additional variable with a dummy equations.

    Like ``ExtVar``, labels of ``AliasAlgeb`` will not be saved in the final output.
    When plotting from file, one need to look up the original variable name.
    """
    def __init__(self, var, **kwargs):
        ExtAlgeb.__init__(self,
                          model=var.owner.class_name,
                          src=var.name,
                          indexer=var.owner.idx,
                          info=f'Alias of {var.name}',
                          **kwargs,
                          )


class AliasState(ExtState):
    """
    Alias state variable.

    Refer to the docs of ``AliasAlgeb``.
    """
    def __init__(self, var, **kwargs):
        ExtState.__init__(self,
                          model=var.owner.class_name,
                          src=var.name,
                          indexer=var.owner.idx,
                          info=f'Alias of {var.name}',
                          **kwargs,
                          )
