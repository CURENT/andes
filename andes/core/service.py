import numpy as np
from typing import Optional, Union
from andes.devices.group import GroupBase


class ServiceBase(object):
    def __init__(self, name=None):
        """
        Base class for service variables

        Parameters
        ----------
        name
        """
        self.v: Union[float, int, np.ndarray] = 0.
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
        if isinstance(self.v, (int, float)):
            return 1
        else:
            return len(self.v)


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
        self.v = None


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
        super().__init__()
        self.name = name
        self.src = src
        self.model = model
        self.indexer = indexer  # `indexer` cannot be None for now

    def link_external(self, ext_model):
        self.v = np.zeros(self.n)
        if self.n == 0:
            return

        if isinstance(ext_model, GroupBase):
            self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr='v')
        else:
            uid = ext_model.idx2uid(self.indexer.v)
            # set initial v and e values to zero
            self.v = ext_model.__dict__[self.src].v[uid]


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
        delattr(self, 'v')

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
