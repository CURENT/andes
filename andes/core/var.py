from typing import Optional, Union, List

from andes.core.param import BaseParam
from andes.models.group import GroupBase
from andes.shared import np, ndarray


class BaseVar(object):
    """
    Base variable class

    This class can be used to instantiate a variable as
    an attribute of a model class.

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
                 v_str: Optional[str] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
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

        self.n = 0
        self.a: Optional[Union[ndarray, List]] = np.array([], dtype=int)  # address array
        self.v: Optional[Union[ndarray, float]] = np.array([], dtype=np.float)  # variable value array
        self.e: Optional[Union[ndarray, float]] = np.array([], dtype=np.float)   # equation value array

        self.v_str = v_str  # equation string (v = v_str) for variable initialization
        self.v_iter = v_iter  # the implicit equation (0 = v_iter) for iterative initialization
        self.e_str = e_str  # string for symbolic equation

        self.v_setter = v_setter  # True if this variable sets the variable value
        self.e_setter = e_setter  # True if this var sets the equation value
        self.addressable = addressable  # True if this var needs to be assigned an address FIXME: not in use
        self.export = export  # True if this var's value needs to exported
        self.diag_eps = diag_eps  # small value to be added to the jacobian matrix

    def reset(self):
        self.a = np.array([], dtype=int)
        self.v = np.array([], dtype=np.float)
        self.e = np.array([], dtype=np.float)

    def __repr__(self):
        span = []
        if 1 <= self.n <= 20:
            span = self.a.tolist()
            span = ','.join([str(i) for i in span])
        elif self.n > 20:
            if not isinstance(self, ExtVar):
                span.append(self.a[0])
                span.append(self.a[-1])
                span.append(self.a[1] - self.a[0])
                span = ':'.join([str(i) for i in span])

        if span:
            span = ' [' + span + ']'

        return f'{self.__class__.__name__}, {self.owner.__class__.__name__}.{self.name}{span}'

    def set_address(self, addr):
        """
        Set the address of this variables

        Parameters
        ----------
        addr : array-like
            The assigned address for this variable
        """
        self.a = addr
        self.n = len(self.a)
        self.v = np.zeros(self.n)
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
                 v_str: Optional[str] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 t_const: Optional[BaseParam] = None,
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
    indexer : BaseParam
        A parameter of the hosting model, used as indices into
        the source model and variable. If is None, the source
        variable address will be fully copied.

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
                 indexer: Optional[Union[List, ndarray, BaseParam]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[str] = None,
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
        """
        self.parent = ext_model

        if isinstance(ext_model, GroupBase):
            if self.indexer.n > 0 and isinstance(self.indexer.v[0], (list, np.ndarray)):
                self._n = [len(i) for i in self.indexer.v]  # number of elements in each sublist
                self._idx = np.concatenate([np.array(i) for i in self.indexer.v])
            else:
                self._n = [len(self.indexer.v)]
                self._idx = self.indexer.v

            self.a = ext_model.get(src=self.src, idx=self._idx, attr='a').astype(int)
            self.n = len(self.a)
            self.v = np.zeros(self.n)
            self.e = np.zeros(self.n)

        else:
            original_var = ext_model.__dict__[self.src]

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
    e_code = 'f'
    v_code = 'x'

    def __init__(self,
                 model: str,
                 src: str,
                 indexer: Optional[Union[List, ndarray, BaseParam]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[str] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 t_const: Optional[BaseParam] = None,
                 v_setter: Optional[bool] = False,
                 e_setter: Optional[bool] = False,
                 addressable: Optional[bool] = True,
                 export: Optional[bool] = True,
                 diag_eps: Optional[float] = 0.0,
                 ):
        super().__init__(model=model,
                         src=src,
                         indexer=indexer,
                         name=name,
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
        self.t_const = t_const


class ExtAlgeb(ExtVar):
    e_code = 'g'
    v_code = 'y'

    def __init__(self,
                 model: str,
                 src: str,
                 indexer: Optional[Union[List, ndarray, BaseParam]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v_str: Optional[str] = None,
                 v_iter: Optional[str] = None,
                 e_str: Optional[str] = None,
                 v_setter: Optional[bool] = False,
                 e_setter: Optional[bool] = False,
                 addressable: Optional[bool] = True,
                 export: Optional[bool] = True,
                 diag_eps: Optional[float] = 0.0,
                 ):
        super().__init__(model=model,
                         src=src,
                         indexer=indexer,
                         name=name,
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
