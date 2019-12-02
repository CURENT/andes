import numpy as np
from typing import Optional  # NOQA


class Discrete(object):

    def __init__(self):
        self.name = None
        self.tex_name = None
        self.owner = None

    def check_var(self):
        pass

    def check_eq(self):
        pass

    def set_var(self):
        pass

    def set_eq(self):
        pass

    def get_names(self):
        pass

    def get_values(self):
        pass


class Limiter(Discrete):
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

    def __init__(self, var, lower, upper, origin=None, enable=True, **kwargs):
        super().__init__()
        self.var = var
        self.lower = lower
        self.upper = upper
        self.origin = self.var if origin is None else origin
        self.enable = enable
        self.zu = 0
        self.zl = 0
        self.zi = 1

    def check_var(self):
        """
        Evaluate `self.zu` and `self.zl`

        Returns
        -------

        """
        if not self.enable:
            return
        self.zu = np.greater_equal(self.origin.v, self.upper.v)
        self.zl = np.less_equal(self.origin.v, self.lower.v)
        self.zi = np.logical_not(np.logical_or(self.zu, self.zl))

        self.zu = self.zu.astype(np.float64)
        self.zl = self.zl.astype(np.float64)
        self.zi = self.zi.astype(np.float64)

    def set_var(self):
        """
        Set the value to the limit

        Returns
        -------

        """
        if not self.enable:
            return
        self.var.v = self.var.v * self.zi + \
            self.lower.v * self.zl + \
            self.upper.v * self.zu

    def get_names(self):
        """
        Available symbols from this class

        Returns
        -------

        """
        return [self.name + '_zl', self.name + '_zi', self.name + '_zu']

    def get_values(self):
        return [self.zl, self.zi, self.zu]


class Comparer(Limiter):
    """
    A value comparer

    Same comparison algorithm as the Limiter. But the comparer
    does not set the limit value.
    """

    def set_var(self):
        # empty set_value function
        pass


class SortedLimiter(Limiter):
    """
    A comparer with the top value selection

    """

    def __init__(self, var, lower, upper, enable=True,
                 n_select: Optional[int] = None,
                 **kwargs):

        super().__init__(var, lower, upper, enable=enable, **kwargs)
        self.n_select = int(n_select) if n_select else 0

    def check_var(self):
        if not self.enable:
            return
        super().check_var()

        if self.n_select is not None and self.n_select > 0:
            asc = np.argsort(self.var.v - self.lower.v)   # ascending order
            desc = np.argsort(self.upper.v - self.var.v)

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
    """
    Hard limit on an algebraic variable
    """
    def __init__(self, var, origin, lower, upper, enable=True, **kwargs):
        super().__init__(var, lower, upper, origin=origin, enable=enable, **kwargs)

    def set_var(self):
        pass

    def set_eq(self):
        pass


class WindupLimiter(Limiter):
    def __init__(self, var, lower, upper, enable=True, **kwargs):
        super().__init__(var, lower, upper, enable=enable, **kwargs)


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

    def __init__(self, var, lower, upper, enable=True, state=None, **kwargs):
        super().__init__(var, lower, upper, enable=enable, **kwargs)
        self.state = state if state else var

    def check_var(self):
        super().check_var()

    def set_var(self):
        pass

    def check_eq(self):
        self.zu = np.logical_and(self.zu, np.greater_equal(self.state.e, 0)).astype(np.float64)
        self.zl = np.logical_and(self.zl, np.less_equal(self.state.e, 0)).astype(np.float64)
        self.zi = np.logical_not(
            np.logical_or(self.zu.astype(np.bool),
                          self.zl.astype(np.bool))).astype(np.float64)

    def set_eq(self):
        super().set_var()
        self.state.e = self.state.e * self.zi


class Selector(Discrete):
    """
    Selection of variables using the provided function.

    The function should take the given number of arguments. Example function is `np.maximum.reduce`.

    Names are in `s0`, `s1`, ... and `sn`
    """
    def __init__(self, *args, fun):
        super().__init__()
        self.input_vars = args
        self.fun = fun
        self.n = len(args)
        self._s = [0] * self.n
        self._inputs = None
        self._outputs = None

    def get_names(self):
        return [f'{self.name}_s' + str(i) for i in range(len(self.input_vars))]

    def get_values(self):
        return self._s

    def check_var(self):
        # set the correct flags to 1 if it is the maximum
        self._inputs = [self.input_vars[i].v for i in range(self.n)]
        self._outputs = self.fun(self._inputs)
        for i in range(self.n):
            self._s[i] = np.equal(self._inputs[i], self._outputs).astype(int)


class DeadBand(Limiter):
    """
    Deadband class with a range threshold

    Warnings
    --------
    Not implemented yet.

    """
    def __init__(self, var, origin, center, lower, upper, enable=True):
        super().__init__(var, lower, upper, origin=origin, enable=enable)
        self.center = center
        # default state if enable is False
        self.zi = 0
        self.zl = 1
        self.zu = 0

    def set_var(self):
        pass


class NonLinearGain(Discrete):
    """
    Non-linear gain function
    """
    pass
