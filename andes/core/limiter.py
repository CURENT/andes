import numpy as np


class Limiter(object):
    def __init__(self, var, lower, upper, **kwargs):
        self.name = None
        self.owner = None

        self.var = var
        self.lower = lower
        self.upper = upper
        self.zu = None
        self.zl = None
        self.zi = None

    def eval(self):
        """
        Evaluate `self.zu` and `self.zl`

        Returns
        -------

        """
        self.zu = np.greater_equal(self.var.v, self.upper.v).astype(np.float64)
        self.zl = np.less_equal(self.var.v, self.lower.v).astype(np.float64)
        self.zi = np.logical_not(
            np.logical_or(
                self.zu.astype(np.bool),
                self.zl.astype(np.bool))).astype(np.float64)

    def set_limit(self):
        """
        Set the value to the limit

        Returns
        -------

        """
        pass

    def get_syms(self):
        """
        Available symbols from thhis class

        Returns
        -------

        """
        return [self.name + '_zl', self.name + '_zi', self.name + '_zu']


class Comparer(Limiter):
    pass


class HardLimiter(Limiter):
    def __init__(self, var, lower, upper, **kwargs):
        super(HardLimiter, self).__init__(var, lower, upper, **kwargs)

    def set_limit(self):
        pass  # TODO: set to the limits


class WindupLimiter(Limiter):
    def __init__(self, var, lower, upper, **kwargs):
        super(WindupLimiter, self).__init__(var, lower, upper, **kwargs)

    def set_limit(self):
        pass  # TODO: set to the limits


class AntiWindupLimiter(WindupLimiter):
    def __init__(self, var, lower, upper, state=None, **kwargs):
        super(AntiWindupLimiter, self).__init__(var, lower, upper, **kwargs)
        self.state = state

    def eval(self):
        super(AntiWindupLimiter, self).eval()
        self.zu = np.logical_and(self.zu, np.greater_equal(self.state.e, 0)).astype(np.float64)
        self.zl = np.logical_and(self.zl, np.less_equal(self.state.e, 0)).astype(np.float64)
        self.zi = np.logical_not(
            np.logical_or(self.zu.astype(np.bool),
                          self.zl.astype(np.bool))).astype(np.float64)

    def set_limit(self):
        pass  # TODO: set to the limits


class DeadBand(Limiter):
    def __init__(self, var, lower, upper, **kwargs):
        super(DeadBand, self).__init__(var, lower, upper, **kwargs)

    def set_limit(self):
        pass  # TODO: set the value based on deadband
