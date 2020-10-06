#  [ANDES] (C)2015-2020 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: discrete.py
#  Last modified: 8/16/20, 7:28 PM

import logging
from typing import Optional, Union, Tuple, List

from andes.core.common import dummify  # NOQA
from andes.shared import np
from andes.utils.tab import Tab
from andes.utils.func import interp_n2

logger = logging.getLogger(__name__)


class Discrete(object):
    """
    Base discrete class.

    Discrete classes export flag arrays (usually boolean) .
    """

    def __init__(self, name=None, tex_name=None, info=None, no_warn=False):
        self.name = name
        self.tex_name = tex_name
        self.info = info
        self.owner = None
        self.export_flags = []
        self.export_flags_tex = []
        self.x_set = list()
        self.y_set = list()   # NOT being used
        self.warn_flags = []  # warn if flags in `warn_flags` not initialized to zero
        self.no_warn = no_warn

        self.has_check_var = False  # if subclass implements `check_var()`
        self.has_check_eq = False   # if subclass implements `check_eq()`

    def check_var(self, *args, **kwargs):
        """
        This function is called in ``l_update_var`` before evaluating equations.

        It should update internal flags only.
        """
        pass

    def check_eq(self):
        """
        This function is called in ``l_check_eq`` after updating equations.

        It updates internal flags, set differential equations, and record pegged variables.
        """
        pass

    def get_names(self):
        """
        Available symbols from this class

        Returns
        -------

        """
        return [f'{self.name}_{flag}' for flag in self.export_flags]

    def get_tex_names(self):
        """
        Return tex_names of exported flags.

        TODO: Fix the bug described in the warning below.

        Warnings
        --------
        If underscore `_` appears in both flag tex_name and `self.tex_name` (for example, when this discrete is
        within a block), the exported tex_name will become invalid for SymPy.
        Variable name substitution will fail.

        Returns
        -------
        list
            A list of tex_names for all exported flags.
        """

        return [rf'{flag_tex}^{self.tex_name}' for flag_tex in self.export_flags_tex]

    def get_values(self):
        return [self.__dict__[flag] for flag in self.export_flags]

    @property
    def class_name(self):
        return self.__class__.__name__

    def list2array(self, n):
        for flag in self.export_flags:
            self.__dict__[flag] = self.__dict__[flag] * np.ones(n, dtype=float)

    def warn_init_limit(self):
        """
        Warn if initialized at limits.
        """
        if self.no_warn:
            return

        for f, limit in self.warn_flags:
            if f not in self.export_flags:
                logger.error(f'warn_flags contain unknown flag {f}')
                continue

            pos = np.argwhere(np.not_equal(self.__dict__[f], 0)).ravel()
            if not len(pos):
                continue
            err_msg = f'{self.owner.class_name}.{self.name} {self.__dict__[limit].name} at limits'
            if isinstance(self.__dict__[limit].v, np.ndarray):
                lim_value = self.__dict__[limit].v[pos]
            else:
                lim_value = self.__dict__[limit].v

            err_data = {'idx': [self.owner.idx.v[i] for i in pos],
                        'Flag': [f] * len(pos),
                        'Input Value': self.u.v[pos],
                        'Limit': lim_value * np.ones_like(pos),
                        }

            tab = Tab(title=err_msg,
                      header=err_data.keys(),
                      data=list(map(list, zip(*err_data.values()))))

            logger.warning(tab.draw())


class LessThan(Discrete):
    """
    Less than (<) comparison function.

    Exports two flags: z1 and z0.
    For elements satisfying the less-than condition, the corresponding z1 = 1.
    z0 is the element-wise negation of z1.

    Notes
    -----
    The default z0 and z1, if not enabled, can be set through the constructor.
    """
    def __init__(self, u, bound, equal=False, enable=True, name=None, tex_name=None,
                 info=None, cache=False, z0=0, z1=1):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = u
        self.bound = dummify(bound)
        self.equal: bool = equal
        self.enable: bool = enable
        self.cache: bool = cache
        self._eval: bool = False  # if has been eval'ed and cached

        self.z0 = np.array([z0])  # negation of `self.z1`
        self.z1 = np.array([z1])  # if the less-than condition (u < bound) is True
        self.export_flags = ['z0', 'z1']
        self.export_flags_tex = ['z_0', 'z_1']

        self.has_check_var = True

    def check_var(self, *args, **kwargs):
        """
        If enabled, set flags based on inputs. Use cached values if enabled.
        """
        if not self.enable:
            return

        if self.cache and self._eval:
            return

        if not self.equal:
            self.z1[:] = np.less(self.u.v, self.bound.v)
        else:
            self.z1[:] = np.less_equal(self.u.v, self.bound.v)

        self.z0[:] = np.logical_not(self.z1)

        self._eval = True


class Limiter(Discrete):
    """
    Base limiter class.

    This class compares values and sets limit values. Exported flags are `zi`, `zl` and `zu`.

    Notes
    -----
    If not enabled, the default flags are ``zu = zl = 0``, ``zi = 1``.

    Parameters
    ----------
    u : BaseVar
        Input Variable instance
    lower : BaseParam
        Parameter instance for the lower limit
    upper : BaseParam
        Parameter instance for the upper limit
    no_lower : bool
        True to only use the upper limit
    no_upper : bool
        True to only use the lower limit
    equal: bool
        True to include equal signs in comparison (>= or <=).
    zu : 0 or 1
        Default value for `zu` if not enabled
    zl : 0 or 1
        Default value for `zl` if not enabled
    zi : 0 or 1
        Default value for `zi` if not enabled

    Attributes
    ----------
    zl : array-like
        Flags of elements violating the lower limit;
        A array of zeros and/or ones.
    zi : array-like
        Flags for within the limits
    zu : array-like
        Flags for violating the upper limit
    """

    def __init__(self, u, lower, upper, enable=True, name=None, tex_name=None, info=None,
                 no_upper=False, no_lower=False, equal=True, zu=0.0, zl=0.0, zi=1.0):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)
        self.u = u
        self.lower = dummify(lower)
        self.upper = dummify(upper)
        self.enable = enable
        self.no_upper = no_upper
        self.no_lower = no_lower
        self.equal = equal

        self.zu = np.array([zu])
        self.zl = np.array([zl])
        self.zi = np.array([zi])

        self.has_check_var = True

        self.export_flags = ['zi']
        self.export_flags_tex = ['z_i']

        if not self.no_lower:
            self.export_flags.append('zl')
            self.export_flags_tex.append('z_l')
            self.warn_flags.append(('zl', 'lower'))
        if not self.no_upper:
            self.export_flags.append('zu')
            self.export_flags_tex.append('z_u')
            self.warn_flags.append(('zu', 'upper'))

    def check_var(self, *args, **kwargs):
        """
        Evaluate the flags.
        """
        if not self.enable:
            return

        if not self.no_upper:
            if self.equal:
                self.zu[:] = np.greater_equal(self.u.v, self.upper.v)
            else:
                self.zu[:] = np.greater(self.u.v, self.upper.v)

        if not self.no_lower:
            if self.equal:
                self.zl[:] = np.less_equal(self.u.v, self.lower.v)
            else:
                self.zl[:] = np.less(self.u.v, self.lower.v)

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))


class SortedLimiter(Limiter):
    """
    A comparer with the top value selection.

    """

    def __init__(self, u, lower, upper, enable=True,
                 n_select: Optional[int] = None, name=None, tex_name=None):

        super().__init__(u, lower, upper, enable=enable, name=name, tex_name=tex_name)
        self.n_select = int(n_select) if n_select else 0

    def check_var(self, *args, **kwargs):
        if not self.enable:
            return
        super().check_var()

        if self.n_select is not None and self.n_select > 0:
            asc = np.argsort(self.u.v - self.lower.v)   # ascending order
            desc = np.argsort(self.upper.v - self.u.v)

            lowest_n = asc[:self.n_select]
            highest_n = desc[:self.n_select]

            reset_in = np.ones(self.u.v.shape)
            reset_in[lowest_n] = 0
            reset_in[highest_n] = 0
            reset_out = 1 - reset_in

            self.zi[:] = np.logical_or(reset_in, self.zi)
            self.zl[:] = np.logical_and(reset_out, self.zl)
            self.zu[:] = np.logical_and(reset_out, self.zu)


class HardLimiter(Limiter):
    """
    Hard limiter for algebraic or differential variable. This class is an alias of `Limiter`.
    """
    pass


class AntiWindup(Limiter):
    """
    Anti-windup limiter.

    Anti-windup limiter prevents the wind-up effect of a differential variable.
    The derivative of the differential variable is reset if it continues to increase in the same direction
    after exceeding the limits.
    During the derivative return, the limiter will be inactive ::

        if x > xmax and x dot > 0: x = xmax and x dot = 0
        if x < xmin and x dot < 0: x = xmin and x dot = 0

    This class takes one more optional parameter for specifying the equation.

    Parameters
    ----------
    state : State, ExtState
        A State (or ExtState) whose equation value will be checked and, when condition satisfies, will be reset
        by the anti-windup-limiter.
    """

    def __init__(self, u, lower, upper, enable=True,
                 no_lower=False, no_upper=False, name=None, tex_name=None, info=None, state=None):
        super().__init__(u, lower, upper, enable=enable,
                         no_lower=no_lower, no_upper=no_upper,
                         name=name, tex_name=tex_name, info=info)
        self.state = state if state else u

        self.has_check_var = False
        self.has_check_eq = True

    def check_var(self, *args, **kwargs):
        """
        This function is empty. Defers `check_var` to `check_eq`.
        """
        pass

    def check_eq(self):
        """
        Check the variables and equations and set the limiter flags.
        Reset differential equation values based on limiter flags.

        Notes
        -----
        The current implementation reallocates memory for `self.x_set` in each call.
        Consider improving for speed. (TODO)
        """
        if not self.no_upper:
            self.zu[:] = np.logical_and(np.greater_equal(self.u.v, self.upper.v),
                                        np.greater_equal(self.state.e, 0))

        if not self.no_lower:
            self.zl[:] = np.logical_and(np.less_equal(self.u.v, self.lower.v),
                                        np.less_equal(self.state.e, 0))

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))

        # must flush the `x_set` list at the beginning
        self.x_set = list()

        if not np.all(self.zi):
            idx = np.where(self.zi == 0)
            self.state.e[:] = self.state.e * self.zi
            self.state.v[:] = self.state.v * self.zi + self.upper.v * self.zu + self.lower.v * self.zl
            self.x_set.append((self.state.a[idx], self.state.v[idx], 0))  # (address, var. values, eqn. values)

            # logger.debug(f'AntiWindup for states {self.state.a[idx]}')

        # Very important note:
        # `System.fg_to_dae` is called after `System.l_update_eq`, which calls this function.
        # Equation values set in `self.state.e` is collected by `System._e_to_dae`, while
        # variable values are collected by the separate loop in `System.fg_to_dae`.
        # Also, equation values are processed in `TDS` for resetting the `q`.


class RateLimiter(Discrete):
    """
    Rate limiter for a differential variable.

    RateLimiter does not export any variable. It directly modifies the differential equation value.

    Notes
    -----
    RateLimiter inherits from Discrete to avoid internal naming conflicts with `Limiter`.

    Warnings
    --------
    RateLimiter cannot be applied to a state variable that already undergoes an AntiWindup limiter.
    Use `AntiWindupRate` for a rate-limited anti-windup limiter.
    """
    def __init__(self, u, lower, upper, enable=True,
                 no_lower=False, no_upper=False, lower_cond=None, upper_cond=None,
                 name=None, tex_name=None, info=None):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)
        self.u = u
        self.rate_lower = dummify(lower)
        self.rate_upper = dummify(upper)

        # `lower_cond` and `upper_cond` are arrays of 0/1 indicating whether
        # the corresponding rate limit should be *enabled*.
        # 0 - disabled, 1 - enabled.
        # If is `None`, all rate limiters will be enabled.

        self.rate_lower_cond = dummify(lower_cond)
        self.rate_upper_cond = dummify(upper_cond)

        self.rate_no_lower = no_lower
        self.rate_no_upper = no_upper
        self.enable = enable

        self.zur = np.array([0])
        self.zlr = np.array([0])

        self.has_check_eq = True

        # Note: save ops by not calculating `zir`
        # self.zir = np.array([1])
        # self.export_flags = ['zir']
        # self.export_flags_tex = ['z_{ir}']

        if not self.rate_no_lower:
            self.export_flags.append('zlr')
            self.export_flags_tex.append('z_{lr}')
            self.warn_flags.append(('zlr', 'lower'))
        if not self.rate_no_upper:
            self.export_flags.append('zur')
            self.export_flags_tex.append('z_{ur}')
            self.warn_flags.append(('zur', 'upper'))

    def check_eq(self):
        if not self.enable:
            return

        if not self.rate_no_lower:
            self.zlr[:] = np.less(self.u.e, self.rate_lower.v)  # 1 if at the lower rate limit

            if self.rate_lower_cond is not None:
                self.zlr[:] = self.zlr * self.rate_lower_cond.v  # 1 if both at the lower rate limit and enabled

            # for where `zlr == 1`, set the equation value to the lower limit
            self.u.e[np.where(self.zlr)] = self.rate_lower.v

        if not self.rate_no_upper:
            self.zur[:] = np.greater(self.u.e, self.rate_upper.v)

            if self.rate_upper_cond is not None:
                self.zur[:] = self.zur * self.rate_upper_cond.v

            self.u.e[np.where(self.zur)] = self.rate_upper.v


class AntiWindupRate(AntiWindup, RateLimiter):
    """
    Anti-windup limiter with rate limits
    """
    def __init__(self, u, lower, upper, rate_lower, rate_upper,
                 no_lower=False, no_upper=False, rate_no_lower=False, rate_no_upper=False,
                 rate_lower_cond=None, rate_upper_cond=None,
                 enable=True, name=None, tex_name=None, info=None):
        RateLimiter.__init__(self, u, lower=rate_lower, upper=rate_upper, enable=enable,
                             no_lower=rate_no_lower, no_upper=rate_no_upper,
                             lower_cond=rate_lower_cond, upper_cond=rate_upper_cond,
                             )

        AntiWindup.__init__(self, u, lower=lower, upper=upper, enable=enable,
                            no_lower=no_lower, no_upper=no_upper,
                            name=name, tex_name=tex_name, info=info,
                            )

    def check_eq(self):
        RateLimiter.check_eq(self)
        AntiWindup.check_eq(self)


class Selector(Discrete):
    """
    Selection between two variables using the provided reduce function.

    The reduce function should take the given number of arguments. An example function is `np.maximum.reduce`
    which can be used to select the maximum.

    Names are in `s0`, `s1`.

    Warnings
    --------
    A potential bug when more than two inputs are provided, and values in different inputs are equal.
    Only two inputs are allowed.

    Examples
    --------
    Example 1: select the largest value between `v0` and `v1` and put it into vmax.

    After the definitions of `v0` and `v1`, define the algebraic variable `vmax` for the largest value,
    and a selector `vs` ::

        self.vmax = Algeb(v_str='maximum(v0, v1)',
                          tex_name='v_{max}',
                          e_str='vs_s0 * v0 + vs_s1 * v1 - vmax')

        self.vs = Selector(self.v0, self.v1, fun=np.maximum.reduce)

    The initial value of `vmax` is calculated by ``maximum(v0, v1)``, which is the element-wise maximum in SymPy
    and will be generated into ``np.maximum(v0, v1)``. The equation of `vmax` is to select the values based on
    `vs_s0` and `vs_s1`.

    Notes
    -----
    A common pitfall is the 0-based indexing in the Selector flags. Note that exported flags start from 0. Namely,
    `s0` corresponds to the first variable provided for the Selector constructor.

    See Also
    --------
    numpy.ufunc.reduce : NumPy reduce function

    andes.core.block.HVGate

    andes.core.block.LVGate
    """
    def __init__(self, *args, fun, tex_name=None, info=None):
        super().__init__(tex_name=tex_name, info=info)
        # TODO: only allow two inputs
        self.input_vars = args
        self.fun = fun
        self.n = len(args)
        self._inputs = None
        self._outputs = None

        # TODO: allow custom initial value
        for i in range(len(self.input_vars)):
            self.__dict__[f's{i}'] = np.array([0])

        self.export_flags = [f's{i}' for i in range(len(self.input_vars))]
        self.export_flags_tex = [f's_{i}' for i in range(len(self.input_vars))]

        self.has_check_var = True

    def check_var(self, *args, **kwargs):
        """
        Set the i-th variable's flags to 1 if the return of the reduce function equals the i-th input.
        """
        if self._inputs is None:
            # input is only evaluated at the first time due to memory stability
            self._inputs = [self.input_vars[i].v for i in range(self.n)]

        if self._outputs is None:
            self._outputs = self.fun(self._inputs)
        else:
            self._outputs[:] = self.fun(self._inputs)

        for i in range(self.n):
            self.__dict__[f's{i}'][:] = np.equal(self._inputs[i], self._outputs)


class Switcher(Discrete):
    """
    Switcher based on an input parameter.

    The switch class takes one v-provider, compares the input with each value in the option list, and exports
    one flag array for each option. The flags are 0-indexed.

    Exported flags are named with `_s0`, `_s1`, ..., with a total number of `len(options)`. See the examples
    section.

    Notes
    -----
    Switches needs to be distinguished from Selector.

    Switcher is for generating flags indicating option selection based on an input parameter. Selector is
    for generating flags at run time based on variable values and a selection function.

    Examples
    --------
    The IEEEST model takes an input for selecting the signal. Options are 1 through 6.
    One can construct ::

        self.IC = NumParam(info='input code 1-6')  # input code
        self.SW = Switcher(u=self.IC, options=[1, 2, 3, 4, 5, 6])

    If the IC values from the data file ends up being ::

        self.IC.v = np.array([1, 2, 2, 4, 6])

    Then, the exported flag arrays will be ::

        {'IC_s0': np.array([1, 0, 0, 0, 0]),
         'IC_s1': np.array([0, 1, 1, 0, 0]),
         'IC_s2': np.array([0, 0, 0, 0, 0]),
         'IC_s3': np.array([0, 0, 0, 1, 0]),
         'IC_s4': np.array([0, 0, 0, 0, 0]),
         'IC_s5': np.array([0, 0, 0, 0, 1])
        }
    """

    def __init__(self, u, options: Union[list, Tuple], name: str = None, tex_name: str = None, cache=True):
        super().__init__(name=name, tex_name=tex_name)
        self.u = u
        self.options: Union[List, Tuple] = options
        self.cache: bool = cache
        self._eval: bool = False  # if the flags has been evaluated

        for i in range(len(options)):
            self.__dict__[f's{i}'] = 0

        self.export_flags = [f's{i}' for i in range(len(options))]
        self.export_flags_tex = [f's_{i}' for i in range(len(options))]

        self.has_check_var = True

    def check_var(self, *args, **kwargs):
        """
        Set the switcher flags based on inputs. Uses cached flags if cache is set to True.
        """
        if self.cache and self._eval:
            return

        for v in self.u.v:
            if v not in self.options and not np.isnan(v):
                raise ValueError(f'option {v} is invalid for {self.owner.class_name}.{self.u.name}. '
                                 f'Options are {self.options}.')
        for i in range(len(self.options)):
            self.__dict__[f's{i}'][:] = np.equal(self.u.v, self.options[i])

        self._eval = True

    def list2array(self, n):
        """
        This forces to evaluate Switcher upon System setup
        """
        super().list2array(n)
        self.check_var()


class DeadBand(Limiter):
    r"""
    The basic deadband type.

    Parameters
    ----------
    u : NumParam
        The pre-deadband input variable
    center : NumParam
        Neutral value of the output
    lower : NumParam
        Lower bound
    upper : NumParam
        Upper bound
    enable : bool
        Enabled if True; Disabled and works as a pass-through if False.

    Notes
    -----

    Input changes within a deadband will incur no output changes. This component computes and exports three flags.

    Three flags computed from the current input:
     - zl: True if the input is below the lower threshold
     - zi: True if the input is within the deadband
     - zu: True if is above the lower threshold

    Initial condition:

    All three flags are initialized to zero. All flags are updated during `check_var` when enabled. If the
    deadband component is not enabled, all of them will remain zero.

    Examples
    --------

    Exported deadband flags need to be used in the algebraic equation corresponding to the post-deadband variable.
    Assume the pre-deadband input variable is `var_in` and the post-deadband variable is `var_out`. First, define a
    deadband instance `db` in the model using ::

        self.db = DeadBand(u=self.var_in, center=self.dbc,
                           lower=self.dbl, upper=self.dbu)

    To implement a no-memory deadband whose output returns to center when the input is within the band,
    the equation for `var` can be written as ::

        var_out.e_str = 'var_in * (1 - db_zi) + \
                         (dbc * db_zi) - var_out'

    """
    def __init__(self, u, center, lower, upper, enable=True, equal=True, zu=0.0, zl=0.0, zi=0.0,
                 name=None, tex_name=None, info=None):
        Limiter.__init__(self, u, lower, upper, enable=enable, equal=equal, zi=zi, zl=zl, zu=zu,
                         name=name, tex_name=tex_name, info=info)
        self.center = dummify(center)  # CURRENTLY NOT IN USE

    def check_var(self, *args, **kwargs):
        """
        Notes
        -----

        Updates three flags: zi, zu, zl based on the following rules:

        zu:
          1 if u > upper; 0 otherwise.

        zl:
          1 if u < lower; 0 otherwise.

        zi:
          not(zu or zl);
        """
        Limiter.check_var(self, *args, **kwargs)


class DeadBandRT(DeadBand):
    r"""
    Deadband with flags for directions of return.

    Parameters
    ----------
    u : NumParam
        The pre-deadband input variable
    center : NumParam
        Neutral value of the output
    lower : NumParam
        Lower bound
    upper : NumParam
        Upper bound
    enable : bool
        Enabled if True; Disabled and works as a pass-through if False.

    Notes
    -----
    Input changes within a deadband will incur no output changes. This component computes and exports five flags.
    The additional two flags on top of `DeadBand` indicate the direction of return:

     - zur: True if the input is/has been within the deadband and was returned from the upper threshold
     - zlr: True if the input is/has been within the deadband and was returned from the lower threshold

    Initial condition:

    All five flags are initialized to zero. All flags are updated during `check_var` when enabled. If the
    deadband component is not enabled, all of them will remain zero.

    Examples
    --------

    To implement a deadband whose output is pegged at the nearest deadband bounds, the equation for `var` can be
    provided as ::

        var_out.e_str = 'var_in * (1 - db_zi) + \
                         dbl * db_zlr + \
                         dbu * db_zur - var_out'

    """
    def __init__(self, u, center, lower, upper, enable=True):
        """

        """
        DeadBand.__init__(self, u, center=center, lower=lower, upper=upper, enable=enable)

        # default state if not enabled
        self.zur = np.array([0.])
        self.zlr = np.array([0.])

        self.export_flags.extend(['zur', 'zlr'])
        self.export_flags_tex.extend(['z_ur', 'z_lr'])

        self.has_check_var = True

    def check_var(self, *args, **kwargs):
        """
        Notes
        -----

        Updates five flags: zi, zu, zl; zur, and zlr based on the following rules:

        zu:
          1 if u > upper; 0 otherwise.

        zl:
          1 if u < lower; 0 otherwise.

        zi:
          not(zu or zl);

        zur:
         - set to 1 when (previous zu + present zi == 2)
         - hold when (previous zi == zi)
         - clear otherwise

        zlr:
         - set to 1 when (previous zl + present zi == 2)
         - hold when (previous zi == zi)
         - clear otherwise
        """
        DeadBand.check_var(self, *args, **kwargs)

        if not self.enable:
            return

        # square return dead band
        self.zur[:] = np.equal(self.zu + self.zi, 2) + self.zur * np.equal(self.zi, self.zi)
        self.zlr[:] = np.equal(self.zl + self.zi, 2) + self.zlr * np.equal(self.zi, self.zi)


class Delay(Discrete):
    """
    Delay class to memorize past variable values.

    Delay allows to impose a predefined "delay" (in either steps or seconds)
    for an input variable. The amount of delay is a scalar and has to be fixed
    at model definition for now.

    """

    def __init__(self, u, mode='step', delay=0, name=None, tex_name=None, info=None):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)

        if mode not in ('step', 'time'):
            raise ValueError(f'mode {mode} is invalid. Must be in "step" or "time"')

        self.u = u
        self.mode = mode
        self.delay = delay
        self.export_flags = ['v']
        self.export_flags_tex = ['v']
        self.has_check_var = True

        self.t = np.array([0])
        self.v = np.array([0])
        self._v_mem = np.zeros((0, 1))
        self.rewind = False

    def list2array(self, n):
        """
        Allocate memory for storage arrays.
        """
        super().list2array(n)
        if self.mode == 'step':
            self._v_mem = np.zeros((n, self.delay + 1))
            self.t = np.zeros(self.delay + 1)
        else:
            self._v_mem = np.zeros((n, 1))

    def check_var(self, dae_t, *args, **kwargs):

        # Storage:
        # Output values is in the first col.
        # Latest values are stored in /appended to the last column
        self.rewind = False

        if dae_t == 0:
            self._v_mem[:] = self.u.v[:, None]

        elif dae_t < self.t[-1]:
            self.rewind = True
            self.t[-1] = dae_t
            self._v_mem[:, -1] = self.u.v

        elif dae_t == self.t[-1]:
            self._v_mem[:, -1] = self.u.v

        elif dae_t > self.t[-1]:
            if self.mode == 'step':
                self.t[:-1] = self.t[1:]
                self.t[-1] = dae_t

                self._v_mem[:, :-1] = self._v_mem[:, 1:]
                self._v_mem[:, -1] = self.u.v
            else:
                self.t = np.append(self.t, dae_t)
                self._v_mem = np.hstack((self._v_mem, self.u.v[:, None]))

                if dae_t - self.t[0] > self.delay:
                    t_interp = dae_t - self.delay
                    idx = np.argmax(self.t >= t_interp) - 1
                    v_interp = interp_n2(t_interp,
                                         self.t[idx:idx+2],
                                         self._v_mem[:, idx:idx + 2])

                    self.t[idx] = t_interp
                    self._v_mem[:, idx] = v_interp

                    self.t = np.delete(self.t, np.arange(0, idx))
                    self._v_mem = np.delete(self._v_mem, np.arange(0, idx), axis=1)

        self.v[:] = self._v_mem[:, 0]

    def __repr__(self):
        out = ''
        out += f'v:\n {self.v}\n'
        out += f't:\n {self.t}\n'
        out += f'_v_men: \n {self._v_mem}\n'
        return out


class Average(Delay):
    """
    Compute the average of a BaseVar over a period of time or a number of samples.
    """
    def check_var(self, dae_t, *args, **kwargs):
        Delay.check_var(self, dae_t, *args, **kwargs)

        if dae_t == 0:
            self.v[:] = self._v_mem[:, -1]
            self._v_mem[:, :-1] = 0
            return
        else:
            nt = len(self.t)
            self.v[:] = 0.5 * np.sum((self._v_mem[:, 1-nt:] + self._v_mem[:, -nt:-1]) *
                                     (self.t[1:] - self.t[:-1]), axis=1) / (self.t[-1] - self.t[0])


class Derivative(Delay):
    """
    Compute the derivative of an algebraic variable using numerical differentiation.
    """
    def __init__(self, u, name=None, tex_name=None, info=None):
        Delay.__init__(self, u=u, mode='step', delay=1,
                       name=name, tex_name=tex_name, info=info)

    def check_var(self, dae_t, *args, **kwargs):
        Delay.check_var(self, dae_t, *args, **kwargs)

        # Note:
        #    Very small derivatives (< 1e-8) could cause numerical problems (chattering).
        #    Need to reset the output to zero following a rewind.
        if (dae_t == 0) or (self.rewind is True):
            self.v[:] = 0
        else:
            self.v[:] = (self._v_mem[:, 1] - self._v_mem[:, 0]) / (self.t[1] - self.t[0])
            self.v[np.where(self.v < 1e-8)] = 0


class Sampling(Discrete):
    """
    Sample an input variable repeatedly at a given time interval.
    """
    def __init__(self, u, interval=1.0, offset=0.0, name=None, tex_name=None, info=None):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)

        self.u = u
        self.interval = interval
        self.offset = offset

        self.export_flags = ['v']
        self.export_flags_tex = ['v']
        self.has_check_var = True

        self.v = np.array([0])
        self._last_t = np.array([0])
        self._last_v = np.array([0])
        self.indices = np.array([0])

        self.rewind = False

    def list2array(self, n):
        super().list2array(n)
        self._last_v = np.zeros(n)

    def check_var(self, dae_t, *args, **kwargs):
        """
        Check and update the output.

        Notes
        -----
        Present output stored in `v`. Output of the last step is stored in `_last_v`.
        Time for the last output is stored in `_last_t`.

        Initially, store `v` and `_last_v`.

        If time progresses and `dae_t` is a multiple of `period`, update `_last_v` and then `v`.
        Record `_last_t`.

        If time does not progress, update `v`.

        If time rewinds, restore `_last_v` to `v`.

        """
        self.rewind = False

        if dae_t == 0:
            # initial step
            self._last_v[:] = self.u.v[:]
            self.v[:] = self.u.v[:]

        elif dae_t > self._last_t:
            do_sample = (dae_t - self.offset - self._last_t) > self.interval

            if do_sample:
                self._last_v[:] = self.v
                self.v[:] = self.u.v
                self._last_t[0] = dae_t

        elif dae_t == self._last_t:
            if len(self.indices) > 0:
                self.v[:] = self.u.v

        else:
            # if dae_t < self._last_t
            self.rewind = True

            if self._last_t[0] > dae_t:
                self.v[:] = self._last_v
                self._last_t[0] = dae_t
