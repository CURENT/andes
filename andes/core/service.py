import numpy as np


class Service(object):
    """
    Base class for service variables
    """
    def __init__(self, name=None, *args, **kwargs):
        self.name = name
        self.owner = None
        self.e_symbolic = None
        self.e_numeric = None
        self.v = None

        self.e_lambdify = None

    def get_name(self):
        return [self.name]

    @property
    def n(self):
        return self.owner.n if self.owner is not None else 0


class ServiceRandom(Service):
    """
    a service variable for generating random numbers
    """
    def __init__(self, name=None, func=np.random.rand, seed=None):
        super(ServiceRandom, self).__init__(name)
        self.func = func
        self.seed = seed
        delattr(self, 'v')

    @property
    def v(self):
        return np.random.rand(self.n)
