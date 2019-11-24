import numpy as np
from typing import Optional, Union, Callable
from andes.core.param import RefParam
from andes.common.operation import list_flatten


class ServiceBase(object):
    def __init__(self, name=None):
        """
        Base class for service variables

        Parameters
        ----------
        name
        """
        self.name = name
        self.owner = None

    def get_name(self):
        """
        Return `name` in a list

        Returns
        -------
        list
            A list only containing the name of the service variable
        """
        return [self.name]

    @property
    def n(self):
        """
        Return the count of the service variable

        Returns
        -------
        int
            The count of elements in this variable
        """
        if isinstance(self.v, np.ndarray):
            return len(self.v)
        else:
            return 1


class ServiceConst(ServiceBase):
    """
    Service variables that remains constants

    Service variables are constants calculated from
    parameters. They are only evaluated once in the
    initialization phase.

    Parameters
    ----------
    name : str
        Name of the service variable

    Attributes
    ----------
    owner : Model
        The hosting/owner model instance
    e_symbolic : str
        A string with the equation to calculate the service
        variable.
    e_lambdify : Callable
        SymPy-generated lambda function for updating the
        value; Not to be provided or modified by the user
    v : array-like
        Evaluated service variable value
    """
    def __init__(self,
                 v_str: Optional[str] = None,
                 name: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(name)
        self.v_str = v_str
        self.v: Union[float, int, np.ndarray] = 0.


class ExtService(ServiceBase):
    """
    Service variable from an attribute of an external model or group.

    Examples
    --------
    A synchronous generator needs to retrieve the p and q values from static generators
    for initialization. It will be stored in an `ExtService` instance.
    """
    def __init__(self,
                 src: str,
                 model: str,
                 indexer,
                 name: Optional[str] = None,
                 **kwargs):
        super().__init__(name)
        self.src = src
        self.model = model
        self.indexer = indexer  # `indexer` cannot be None for now
        self.v = 0

    def link_external(self, ext_model):
        # set initial v values to zero
        self.v = np.zeros(self.n)
        if self.n == 0:
            return

        # the same `get` api for Group and Model
        self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr='v')


class ServiceOperation(ServiceBase):

    def __init__(self,
                 origin,
                 ref: RefParam,
                 name=None):
        self._v = None
        super().__init__(name)
        self.origin = origin
        self.ref = ref
        self.v_str = None

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


class ServiceReduce(ServiceOperation):
    def __init__(self,
                 fun: Callable,
                 **kwargs):
        super().__init__(**kwargs)
        self.fun = fun

    @property
    def v(self):
        if self._v is None:
            self._v = np.zeros(len(self.ref.v))
            idx = 0
            for i, v in enumerate(self.ref.v):
                self._v[i] = self.fun(self.origin.v[idx:idx + len(v)])
                idx += len(v)
            return self._v
        else:
            return self._v


class ServiceRepeat(ServiceOperation):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    @property
    def v(self):
        if self._v is None:
            self._v = np.zeros(len(list_flatten(self.ref.v)))
            idx = 0
            for i, v in enumerate(self.ref.v):
                self._v[idx:idx + len(v)] = self.origin.v[i]
                idx += len(v)
            return self._v
        else:
            return self._v


class ServiceRandom(ServiceConst):
    """
    A service variable for generating random numbers

    Parameters
    ----------
    name : str
        Name
    func : Callable
        A callable for generating the random variable.
    """
    def __init__(self, name=None, func=np.random.rand):
        super(ServiceRandom, self).__init__(name)
        self.func = func

    @property
    def v(self):
        """
        This class has `v` wrapped by a property descriptor.

        Returns
        -------
        array-like
            Randomly generated service variables
        """
        return np.random.rand(self.n)
