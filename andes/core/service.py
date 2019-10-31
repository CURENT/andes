class Service(object):
    """
    Base class for service variables
    """
    def __init__(self, name=None):
        self.name = name
        self.owner = None
        self.e_symbolic = None
        self.e_numeric = None
        self.v = None

        self.e_lambdify = None

    def get_name(self):
        return [self.name]
