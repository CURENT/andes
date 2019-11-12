import numpy as np
from typing import Optional  # NOQA


class NonLinear(object):

    def __init__(self):
        pass

    def eval(self):
        pass

    def set(self):
        pass

    def get_name(self):
        pass

    def get_value(self):
        pass


class Limiter(object):
    """
    Base limiter class

    This class compares values and sets limit values

    Parameters
    ----------
    var : VarBase
        Variable instance
    lower : ParamBase
        Parameter instance for the lower limit
    upper : ParamBase
        Parameter instance for the upper limit

    Attributes
    ----------
    zl : array-like
        Flags of elements violating the lower limit;
        A array of zeros and/or ones.
    zi : array-like
        Flags for within the limits
    zu : array-like
        Flags for violating the upper limit
    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, var, lower, upper, **kwargs):
        self.name = None
        self.owner = None

        self.var = var
        self.lower = lower
        self.upper = upper
        self.zu = 0
        self.zl = 0
        self.zi = 1

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

    def set_value(self):
        """
        Set the value to the limit

        Returns
        -------

        """
        self.var.v = self.var.v * (1 - self.zu) * (1 - self.zl) + \
            self.lower.v * self.zl + \
            self.upper.v * self.zu

    def get_name(self):
        """
        Available symbols from this class

        Returns
        -------

        """
        return [self.name + '_zl', self.name + '_zi', self.name + '_zu']

    def get_value(self):
        return [self.zl, self.zi, self.zu]


class Comparer(Limiter):
    """
    A value comparer

    Same comparison algorithm as the Limiter. But the comparer
    does not set the limit value.
    """

    def set_value(self):
        # empty set_value function
        pass


class SortedLimiter(Limiter):
    """
    A comparer with the top value selection

    """

    def __init__(self, var, lower, upper,
                 n_select: Optional[int] = None,
                 **kwargs):

        super().__init__(var, lower, upper, **kwargs)
        self.n_select = n_select

    def eval(self):
        super().eval()

        if self.n_select is not None and self.n_select > 0:
            asc = np.argsort(self.var.v)
            desc = np.flip(asc)

            lowest_n = asc[:self.n_select]
            highest_n = desc[:self.n_select]

            reset_in = np.ones(self.var.v.shape)
            reset_in[lowest_n] = 0
            reset_in[highest_n] = 0
            reset_out = 1 - reset_in

            self.zi = np.logical_or(reset_in, self.zi).astype(np.float64)
            self.zl = np.logical_and(reset_out, self.zl).astype(np.float64)
            self.zu = np.logical_and(reset_out, self.zu).astype(np.float64)


class HardLimiter(Limiter):
    def __init__(self, var, lower, upper, **kwargs):
        super(HardLimiter, self).__init__(var, lower, upper, **kwargs)


class WindupLimiter(Limiter):
    def __init__(self, var, lower, upper, **kwargs):
        super(WindupLimiter, self).__init__(var, lower, upper, **kwargs)


class AntiWindupLimiter(WindupLimiter):
    """
    Anti-windup limiter

    This class takes one more optional parameter for specifying the
    equation.

    Parameters
    ----------
    State: VarBase
        A VarBase instance whose equation value will be checked and reset
        by the anti-windup-limiter.
    """

    def __init__(self, var, lower, upper, state=None, **kwargs):
        super(AntiWindupLimiter, self).__init__(var, lower, upper, **kwargs)
        self.state = state if state else var

    def eval(self):
        super(AntiWindupLimiter, self).eval()
        self.zu = np.logical_and(self.zu, np.greater_equal(self.state.e, 0)).astype(np.float64)
        self.zl = np.logical_and(self.zl, np.less_equal(self.state.e, 0)).astype(np.float64)
        self.zi = np.logical_not(
            np.logical_or(self.zu.astype(np.bool),
                          self.zl.astype(np.bool))).astype(np.float64)

    def set_value(self):
        super(AntiWindupLimiter, self).set_value()
        self.state.e = self.state.e * (1 - self.zu) * (1 - self.zl)


class DeadBand(Limiter):
    """
    Deadband class with a range threshold

    Warnings
    --------
    Not implemented yet.

    """

    def __init__(self, var, lower, upper, **kwargs):
        super(DeadBand, self).__init__(var, lower, upper, **kwargs)

    def set_value(self):
        pass  # TODO: set the value based on deadband
