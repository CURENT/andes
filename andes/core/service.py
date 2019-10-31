class ServiceBase(object):
    """
    Base class for service variables
    """
    def __init__(self, name=None):
        self.name = name
        self.equation = None
        self.efunction = None
        self.v = None

    def eval(self):
        pass  # update self.v here


class ServiceConstant(ServiceBase):
    pass


class ServiceVariable(ServiceBase):
    pass
