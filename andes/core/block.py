from andes.core.var import Algeb, State


class Block(object):
    def __init__(self, *args, **kwargs):
        self.name = None
        self.owner = None

    def get_syms(self):
        pass

    def export_vars(self):
        pass


class SampleAndHolder(Block):
    pass


class PIController(Block):
    """
    Proportional Integral Controller
    """
    def __init__(self, var, kp, ki, **kwargs):
        super(PIController, self).__init__(var, kp, ki, **kwargs)

        self.var = var
        self.kp = kp
        self.ki = ki

        self.xi = State(descr="integration value of PI controller")
        self.y = Algeb(descr="integration value of PI controller")

    def export_vars(self):
        print(self.__dict__)
        return {'xi': self.xi,
                'y': self.y}
