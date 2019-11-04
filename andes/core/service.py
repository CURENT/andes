import numpy as np


class Service(object):
    """
    Base class for service variables

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
    e_numeric : Callable
        A user-defined callback for calculating the service
        variable.
    e_lambdify : Callable
        SymPy-generated lambda function for updating the
        value; Not to be provided or modified by the user
    v : array-like
        Evaluated service variable value
    """
    def __init__(self, name=None, *args, **kwargs):
        self.name = name
        self.owner = None
        self.e_symbolic = None
        self.e_numeric = None
        self.v = None

        self.e_lambdify = None

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
        return self.owner.n if self.owner is not None else 0


class ServiceRandom(Service):
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
