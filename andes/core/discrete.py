#  [ANDES] (C)2015-2022 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: discrete.py

import logging
from typing import List, Tuple, Union

import numpy as np

from andes.core.common import dummify
from andes.utils.func import interp_n2
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)


class Discrete:
    """
    Base discrete class.

    Discrete classes export flag arrays (usually boolean) .
    """

    def __init__(self, name=None, tex_name=None, info=None, no_warn=False,
                 min_iter=2, err_tol=1e-2,
                 ):
        self.name = name
        self.tex_name = tex_name
        self.info = info
        self.owner = None
        if not hasattr(self, 'export_flags'):
            self.export_flags = []
        if not hasattr(self, 'export_flags_tex'):
            self.export_flags_tex = []

        self.input_list = []   # references to input variables
        self.param_list = []  # references to parameters

        self.x_set = list()
        self.y_set = list()   # NOT being used

        self.warn_flags = []  # warn if flags in `warn_flags` not initialized to zero
        self.no_warn = no_warn

        # default minimum iteration number and error tolerance to allow checking
        # To enable `min_iter` and `err_tol`, a `Discrete` subclass needs to call
        # `check_iter_err()` manually in `check_var()` and/or `check_eq()`.

        self.min_iter = min_iter
        self.err_tol = err_tol

        self.has_check_var = False  # if subclass implements `check_var()`
        self.has_check_eq = False   # if subclass implements `check_eq()`

    def check_var(self, *args, **kwargs):
        """
        This function is called in ``l_update_var`` before evaluating equations.

        It should update internal flags only.

        Parameters
        ----------
        adjust_upper : bool
            True to adjust the upper limit to the value of the variable.
            Supported by limiters.
        adjust_lower : bool
            True to adjust the lower limit to the value of the variable.
            Supported by limiters.

        """
        pass

    def check_eq(self, **kwargs):
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
        """
        Allocate memory for the discrete flags specified in `self.export_flags`.

        Parameters
        ----------
        n : int
            Number of elements in the array. Provided by the calling function.
        """

        for flag in self.export_flags:
            self.__dict__[flag] = self.__dict__[flag] * np.ones(n, dtype=float)

    def warn_init_limit(self):
        """
        Warn if associated variables are initialized at limits.
        """

        if self.no_warn:
            return

        for f, limit in self.warn_flags:
            if f not in self.export_flags:
                logger.error('warn_flags contain unknown flag %s', f)
                continue

            mask = np.ones(self.owner.n, dtype=bool)
            if limit == 'upper':
                mask = self.mask_upper
            elif limit == 'lower':
                mask = self.mask_lower
            else:
                logger.debug('Unknown limit name <%s>', limit)

            # process online devices only
            flag_vals = np.logical_and(self.__dict__[f], self.owner.u.v)

            # ignore limits that has been adjusted
            flag_vals = np.logical_and(flag_vals, np.logical_not(mask))

            pos = np.argwhere(np.not_equal(flag_vals, 0)).ravel()

            if len(pos) == 0:
                continue

            # convert limie values to arrays
            if isinstance(self.__dict__[limit].v, np.ndarray):
                lim_value = self.__dict__[limit].v
            else:
                lim_value = self.__dict__[limit].v * np.ones(self.owner.n)

            at_limit_pos = list()
            out_limit_pos = list()

            for item in pos:
                if np.isclose(lim_value[item], self.u.v[item]):
                    at_limit_pos.append(item)
                else:
                    out_limit_pos.append(item)

            if len(out_limit_pos) > 0:
                # warn out of limits
                err_msg = f'{self.owner.class_name}.{self.name} out of limits <{self.__dict__[limit].name}>'
                err_data = {'idx': [self.owner.idx.v[i] for i in out_limit_pos],
                            'Flag': [f] * len(out_limit_pos),
                            'Input Value': self.u.v[out_limit_pos],
                            'Limit': lim_value[out_limit_pos]
                            }

                tab = Tab(title=err_msg,
                          header=err_data.keys(),
                          data=list(map(list, zip(*err_data.values()))))

                logger.warning(tab.draw())

            if len(at_limit_pos) > 0:
                # warn at limits
                err_msg = f'{self.owner.class_name}.{self.name} at limits <{self.__dict__[limit].name}>'
                err_data = {'idx': [self.owner.idx.v[i] for i in at_limit_pos],
                            'Flag': [f] * len(at_limit_pos),
                            'Input Value': self.u.v[at_limit_pos],
                            'Limit': lim_value[at_limit_pos]
                            }

                tab = Tab(title=err_msg,
                          header=err_data.keys(),
                          data=list(map(list, zip(*err_data.values()))))

                logger.debug(tab.draw())

    def check_iter_err(self, niter=None, err=None):
        """
        Check if the minimum iteration or maximum error is reached
        so that this discrete block should be enabled.

        Only when both `niter` and `err` are given,  (niter < min_iter)
        , and (err > err_tol) it will return False.

        This logic will start checking the discrete states if called
        from an external solver that does not feed `niter` or `err`
        at each step.

        Returns
        -------
        bool
            True if it should be enabled, False otherwise
        """
        if (niter is not None) and (niter < self.min_iter) and \
                (err is not None) and (err > self.err_tol):
            return False

        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}'


class LessThan(Discrete):
    """
    Less than (<) comparison function that tests if ``u < bound``.

    Exports two flags: z1 and z0.
    For elements satisfying the less-than condition, the corresponding z1 = 1.
    z0 is the element-wise negation of z1.

    Notes
    -----
    The default z0 and z1, if not enabled, can be set through the constructor.
    By default, the model will not adjust the limit.
    """

    def __init__(self, u, bound, equal=False, enable=True, name=None, tex_name=None,
                 info: str = None, cache: bool = False,
                 z0=0, z1=1):
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

        self.input_list.extend([self.u])
        self.param_list.extend([self.bound])

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


class IsEqual(Discrete):
    """
    Is equal (==) comparison function to test if ``u == bound``.

    Exports one flag: z1.
    For elements satisfying the equality condition, the corresponding z1 = 1.

    Notes
    -----
    The default z1 when not enabled can be set through the constructor.
    By default, the model will not adjust the limit.
    """

    def __init__(self, u, bound, enable=True, name=None, tex_name=None,
                 info: str = None, cache: bool = False, z1=1):
        super().__init__(name=name, tex_name=tex_name, info=info)

        self.u = u
        self.bound = dummify(bound)
        self.enable: bool = enable
        self.cache: bool = cache
        self._eval: bool = False

        self.z1 = np.array([z1])
        self.export_flags = ['z1']
        self.export_flags_tex = ['z_1']

        self.input_list.extend([self.u])
        self.param_list.extend([self.bound])

        self.has_check_var = True

    def check_var(self, *args, **kwargs):
        """
        If enabled, set flags based on inputs. Use cached values if enabled.
        """

        if not self.enable:
            return

        if self.cache and self._eval:
            return

        self.z1[:] = np.equal(self.u.v, self.bound.v)

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
    sign_lower: 1 or -1
        Sign to be multiplied to the lower limit
    sign_upper: bool
        Sign to be multiplied to the upper limit
    equal : bool
        True to include equal signs in comparison (>= or <=).
    no_warn : bool
        Disable initial limit warnings
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

    def __init__(self, u, lower, upper, enable=True,
                 name: str = None, tex_name: str = None, info: str = None,
                 min_iter: int = 2, err_tol: float = 0.01,
                 allow_adjust: bool = True,
                 no_lower=False, no_upper=False, sign_lower=1, sign_upper=1,
                 equal=True, no_warn=False,
                 zu=0.0, zl=0.0, zi=1.0):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info,
                          min_iter=min_iter, err_tol=err_tol)
        self.u = u
        self.lower = dummify(lower)
        self.upper = dummify(upper)
        self.enable = enable
        self.no_lower = no_lower
        self.no_upper = no_upper

        self.allow_adjust = allow_adjust

        if sign_lower not in (1, -1):
            raise ValueError("sign_lower must be 1 or -1, got %s" % sign_lower)
        if sign_upper not in (1, -1):
            raise ValueError("sign_upper must be 1 or -1, got %s" % sign_upper)

        self.sign_lower = dummify(sign_lower)
        self.sign_upper = dummify(sign_upper)
        self.equal = equal
        self.no_warn = no_warn

        self.zu = np.array([zu])
        self.zl = np.array([zl])
        self.zi = np.array([zi])

        self.mask_upper = None
        self.mask_lower = None

        self.has_check_var = True

        self.export_flags.append('zi')
        self.export_flags_tex.append('z_i')

        self.input_list.extend([self.u])
        self.param_list.extend([self.lower, self.upper])

        if not self.no_lower:
            self.export_flags.append('zl')
            self.export_flags_tex.append('z_l')
            self.warn_flags.append(('zl', 'lower'))
        if not self.no_upper:
            self.export_flags.append('zu')
            self.export_flags_tex.append('z_u')
            self.warn_flags.append(('zu', 'upper'))

    def check_var(self,
                  allow_adjust=True,  # allow flag from model
                  adjust_lower=False,
                  adjust_upper=False,
                  is_init: bool = False,
                  *args, **kwargs):
        """
        Check the input variable and set flags.
        """

        if not self.enable:
            return

        if not self.no_upper:
            upper_v = -self.upper.v if self.sign_upper.v == -1 else self.upper.v

            # FIXME: adjust will not be successful when sign is -1
            if self.allow_adjust and is_init:
                self.do_adjust_upper(self.u.v, upper_v, allow_adjust, adjust_upper)

            if self.equal:
                self.zu[:] = np.greater_equal(self.u.v, upper_v)
            else:
                self.zu[:] = np.greater(self.u.v, upper_v)

        if not self.no_lower:
            lower_v = -self.lower.v if self.sign_lower.v == -1 else self.lower.v

            # FIXME: adjust will not be successful when sign is -1
            if self.allow_adjust and is_init:
                self.do_adjust_lower(self.u.v, lower_v, allow_adjust, adjust_lower)

            if self.equal:
                self.zl[:] = np.less_equal(self.u.v, lower_v)
            else:
                self.zl[:] = np.less(self.u.v, lower_v)

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))

    def do_adjust_lower(self, val, lower, allow_adjust=True, adjust_lower=False):
        """
        Adjust the lower limit.

        Notes
        -----
        This method is only executed if `allow_adjust` is True
        and `adjust_lower` is True.
        """

        if allow_adjust:
            mask = (val < lower)

            if sum(mask) == 0:
                return

            if adjust_lower:
                self._show_adjust(val, lower, mask, self.lower.name, adjusted=True)
                lower[mask] = val[mask]
                self.mask_lower = mask  # store after adjusting
            else:
                self._show_adjust(val, lower, mask, self.lower.name, adjusted=False)

    def _show_adjust(self, val, old_limit, mask, limit_name, adjusted=True):
        """
        Helper function to show a table of the adjusted limits.
        """

        idxes = np.array(self.owner.idx.v)[mask]
        if isinstance(val, (int, float)):
            val = np.array([val])
        if isinstance(old_limit, (int, float)):
            old_limit = np.array([old_limit])

        adjust_or_not = 'adjusted' if adjusted else '*not adjusted*'

        tab = Tab(title=f"{self.owner.class_name}.{self.name}: {adjust_or_not} limit <{limit_name}>",
                  header=['Idx', 'Input', 'Old Limit'],
                  data=[*zip(idxes, val[mask], old_limit[mask])],
                  )

        if adjusted:
            logger.info(tab.draw())
        else:
            logger.warning(tab.draw())

    def do_adjust_upper(self, val, upper, allow_adjust=True, adjust_upper=False):
        """
        Adjust the upper limit.

        Notes
        -----
        This method is only executed if `allow_adjust` is True
        and `adjust_upper` is True.
        """

        if allow_adjust:
            mask = (val > upper)
            if sum(mask) == 0:
                return

            if adjust_upper:
                self._show_adjust(val, upper, mask, self.upper.name, adjusted=True)
                upper[mask] = val[mask]
                self.mask_upper = mask
            else:
                self._show_adjust(val, upper, mask, self.lower.name, adjusted=False)


class SortedLimiter(Limiter):
    """
    A limiter that sorts inputs based on the absolute or
    relative amount of limit violations.

    Parameters
    ----------
    n_select : int
        the number of violations to be flagged,
        for each of over-limit and under-limit cases.
        If `n_select` == 1, at most one over-limit
        and one under-limit inputs will be flagged.
        If `n_select` is zero, heuristics will be used.

    abs_violation : bool
        True to use the absolute violation.
        False if the relative violation
        abs(violation/limit) is used for sorting.
        Since most variables are in per unit,
        absolute violation is recommended.

    """

    def __init__(self, u, lower, upper, n_select: int = 5,
                 name=None, tex_name=None, enable=True, abs_violation=True,
                 min_iter: int = 2, err_tol: float = 0.01,
                 allow_adjust: bool = True,
                 zu=0.0, zl=0.0, zi=1.0, ql=0.0, qu=0.0,
                 ):

        super().__init__(u, lower, upper,
                         enable=enable, name=name, tex_name=tex_name,
                         min_iter=min_iter, err_tol=err_tol,
                         allow_adjust=allow_adjust,
                         zu=zu, zl=zl, zi=zi,
                         )

        self.n_select = int(n_select)
        self.auto = True if self.n_select == 0 else False
        self.abs_violation = abs_violation

        self.ql = np.array([ql])
        self.qu = np.array([qu])

        # count of ones in `ql` and `qu`
        self.nql = 0
        self.nqu = 0

        # smallest and largest `n_select`
        self.min_sel = 2
        self.max_sel = 50

        # store the lower and upper limit values with zeros converted to a small number
        self.lower_denom = None
        self.upper_denom = None

        self.export_flags.extend(['ql', 'qu'])
        self.export_flags_tex.extend(['q_l', 'q_u'])

    def list2array(self, n):
        """
        Initialize maximum and minimum `n_select` based on input size.
        """

        super().list2array(n)
        if self.auto:
            self.min_sel = max(2, int(n / 10))
            self.max_sel = max(2, int(n / 2))

    def check_var(self, *args, niter=None, err=None, **kwargs):
        """
        Check for the largest and smallest `n_select` elements.
        """

        if not self.enable:
            return

        if not self.check_iter_err(niter=niter, err=err):
            return

        super().check_var()

        # first run - calculate the denominators if using relative violation
        if not self.abs_violation:
            if self.lower_denom is None:
                self.lower_denom = np.array(self.lower.v)
                self.lower_denom[self.lower_denom == 0] = 1e-6
            if self.upper_denom is None:
                self.upper_denom = np.array(self.upper.v)
                self.upper_denom[self.upper_denom == 0] = 1e-6

        # calculate violations - abs or relative
        if self.abs_violation:
            lower_vio = self.u.v - self.lower.v
            upper_vio = self.upper.v - self.u.v
        else:
            lower_vio = np.abs((self.u.v - self.lower.v) / self.lower_denom)
            upper_vio = np.abs((self.upper.v - self.u.v) / self.upper_denom)

        # count the number of inputs flagged
        if self.auto:
            self.calc_select()

        # sort in both ascending and descending orders
        asc = np.argsort(lower_vio)
        desc = np.argsort(upper_vio)
        top_n = asc[:self.n_select]
        bottom_n = desc[:self.n_select]

        # `reset_out` is used to flag the
        reset_out = np.zeros_like(self.u.v)
        reset_out[top_n] = 1
        reset_out[bottom_n] = 1

        # set new flags
        self.zl[:] = np.logical_or(np.logical_and(reset_out, self.zl),
                                   self.ql)
        self.zu[:] = np.logical_or(np.logical_and(reset_out, self.zu),
                                   self.qu)
        self.zi[:] = 1 - np.logical_or(self.zl, self.zu)
        self.ql[:] = self.zl
        self.qu[:] = self.zu

        # compute the number of updated flags
        ql1 = np.count_nonzero(self.ql)
        qu1 = np.count_nonzero(self.qu)
        dqu = qu1 - self.nqu
        dql = ql1 - self.nql

        if dqu > 0 or dql > 0:
            logger.debug("SortedLimiter: flagged %s upper and %s lower limit violations",
                         dqu, dql)
            self.nqu = qu1
            self.nql = ql1

    def calc_select(self):
        """
        Set `n_select` automatically.
        """
        ret = int((np.count_nonzero(self.zl) + np.count_nonzero(self.zu)) / 2) + 1

        if ret > self.max_sel:
            ret = self.max_sel
        elif ret < self.min_sel:
            ret = self.min_sel

        self.n_select = ret


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

    def __init__(self, u, lower, upper, enable=True, no_warn=False,
                 no_lower=False, no_upper=False, sign_lower=1, sign_upper=1,
                 name=None, tex_name=None, info=None, state=None,
                 allow_adjust: bool = True,):
        super().__init__(u, lower, upper, enable=enable, no_warn=no_warn,
                         no_lower=no_lower, no_upper=no_upper,
                         sign_lower=sign_lower, sign_upper=sign_upper,
                         name=name, tex_name=tex_name, info=info,
                         allow_adjust=allow_adjust,)
        self.state = state if state else u

        self.has_check_var = False
        self.has_check_eq = True
        self.no_warn = no_warn

        self.export_flags.extend(['zu0', 'zl0'])
        self.export_flags_tex.extend(['z_{u0}', 'z_{l0}'])

        self.zu0 = np.array(self.zu)
        self.zl0 = np.array(self.zl)
        self.niter_lock = 4  # lock limiter after `niter_lock` iterations to stop chattering

    def check_var(self, *args, **kwargs):
        """
        This function is empty. Defers `check_var` to `check_eq`.
        """
        pass

    def check_eq(self,
                 allow_adjust=True,
                 adjust_lower=False,
                 adjust_upper=False,
                 is_init: bool = False,
                 niter: int = 0,
                 **kwargs):
        """
        Check the variables and equations and set the limiter flags.
        Reset differential equation values based on limiter flags.

        Notes
        -----
        The current implementation reallocates memory for `self.x_set` in each call.
        Consider improving for speed. (TODO)
        """
        if not self.no_upper:
            upper_v = -self.upper.v if self.sign_upper.v == -1 else self.upper.v

            if self.allow_adjust and is_init:
                self.do_adjust_upper(self.u.v, upper_v,
                                     allow_adjust=allow_adjust,
                                     adjust_upper=adjust_upper)

            self.zu0[:] = self.zu
            self.zu[:] = np.logical_and(np.greater_equal(self.u.v, upper_v),
                                        np.greater_equal(self.state.e, 0))

            if niter > self.niter_lock:
                self.zu[:] = np.logical_or(self.zu0, self.zu)

        if not self.no_lower:
            lower_v = -self.lower.v if self.sign_lower.v == -1 else self.lower.v

            if self.allow_adjust and is_init:
                self.do_adjust_lower(self.u.v, lower_v,
                                     allow_adjust=allow_adjust,
                                     adjust_lower=adjust_lower)

            self.zl0[:] = self.zl
            self.zl[:] = np.logical_and(np.less_equal(self.u.v, lower_v),
                                        np.less_equal(self.state.e, 0))
            if niter > self.niter_lock:
                self.zl[:] = np.logical_or(self.zl0, self.zl)

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))

        # must flush the `x_set` list at the beginning
        self.x_set = list()

        if not np.all(self.zi):
            idx = np.where(self.zi == 0)
            self.state.e[:] = self.state.e * self.zi
            self.state.v[:] = self.state.v * self.zi
            if not self.no_upper:
                self.state.v[:] += upper_v * self.zu
            if not self.no_lower:
                self.state.v[:] += lower_v * self.zl
            self.x_set.append((self.state.a[idx], self.state.v[idx], 0))  # (address, var. values, eqn. values)

            # logger.debug(f'AntiWindup for states {self.state.a[idx]}')

        # Very important note:
        # The set equation values and variable values are collected by `System.fg_to_dae`:
        # - Equation values is collected by `System._e_to_dae`,
        # - Variable values are collected at the end of `System.fg_to_dae`.
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
                 no_lower=False, no_upper=False, lower_cond=1, upper_cond=1,
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

        self.mask_lower = None
        self.mask_upper = None

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

        self.param_list.extend([self.rate_lower, self.rate_upper,
                                self.rate_lower_cond, self.rate_upper_cond])

    def check_eq(self, **kwargs):
        if not self.enable:
            return

        if not self.rate_no_lower:
            self.zlr[:] = np.less(self.u.e, self.rate_lower.v)  # 1 if at the lower rate limit

            if self.rate_lower_cond is not None:
                self.zlr[:] = self.zlr * self.rate_lower_cond.v  # 1 if both at the lower rate limit and enabled

            # for where `zlr == 1`, set the equation value to the lower limit
            self.u.e[np.where(self.zlr)] = self.rate_lower.v[np.where(self.zlr)]

        if not self.rate_no_upper:
            self.zur[:] = np.greater(self.u.e, self.rate_upper.v)

            if self.rate_upper_cond is not None:
                self.zur[:] = self.zur * self.rate_upper_cond.v

            self.u.e[np.where(self.zur)] = self.rate_upper.v[np.where(self.zur)]


class AntiWindupRate(AntiWindup, RateLimiter):
    """
    Anti-windup limiter with rate limits
    """

    def __init__(self, u, lower, upper, rate_lower, rate_upper,
                 no_lower=False, no_upper=False, rate_no_lower=False, rate_no_upper=False,
                 rate_lower_cond=None, rate_upper_cond=None,
                 enable=True, name=None, tex_name=None, info=None,
                 allow_adjust: bool = True,):

        RateLimiter.__init__(self, u, lower=rate_lower, upper=rate_upper, enable=enable,
                             no_lower=rate_no_lower, no_upper=rate_no_upper,
                             lower_cond=rate_lower_cond, upper_cond=rate_upper_cond,
                             )

        AntiWindup.__init__(self, u, lower=lower, upper=upper, enable=enable,
                            no_lower=no_lower, no_upper=no_upper,
                            name=name, tex_name=tex_name, info=info,
                            allow_adjust=allow_adjust,
                            )

    def check_eq(self, **kwargs):
        RateLimiter.check_eq(self, **kwargs)
        AntiWindup.check_eq(self, **kwargs)


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

    .. deprecated:: 1.5.9

        Use of this class for comparison-based output is discouraged.
        Instead, use `LessThan` and `Limiter` to construct piesewise equations.

        See the new implementation of ``HVGate`` and ``LVGate``.

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

        self.input_list = args

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
        self.SW = Switcher(u=self.IC, options=[0, 1, 2, 3, 4, 5, 6])

    If the IC values from the data file ends up being ::

        self.IC.v = np.array([1, 2, 2, 4, 6])

    Then, the exported flag arrays will be ::

        {'IC_s0': np.array([0, 0, 0, 0, 0]),
         'IC_s1': np.array([1, 0, 0, 0, 0]),
         'IC_s2': np.array([0, 1, 1, 0, 0]),
         'IC_s3': np.array([0, 0, 0, 0, 0]),
         'IC_s4': np.array([0, 0, 0, 1, 0]),
         'IC_s5': np.array([0, 0, 0, 0, 0]),
         'IC_s6': np.array([0, 0, 0, 0, 1])
        }

    where `IC_s0` is used for padding so that following flags align with the options.
    """

    def __init__(self, u, options: Union[list, Tuple], info: str = None,
                 name: str = None, tex_name: str = None, cache=True,):
        super().__init__(name=name, tex_name=tex_name, info=info,)
        self.u = u
        self.options: Union[List, Tuple] = options
        self.cache: bool = cache
        self._eval: bool = False  # if the flags has been evaluated

        for i in range(len(options)):
            self.__dict__[f's{i}'] = 0

        self.export_flags = [f's{i}' for i in range(len(options))]
        self.export_flags_tex = [f's_{i}' for i in range(len(options))]
        self.input_list.extend([self.u])

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
        if len(self.u.v) > 0:
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

    def __init__(self, u, center, lower, upper,
                 enable=True, equal=False,
                 zu=0.0, zl=0.0, zi=0.0,
                 name=None, tex_name=None, info=None,
                 ):
        Limiter.__init__(self, u, lower, upper,
                         enable=enable, equal=equal, zi=zi, zl=zl, zu=zu,
                         name=name, tex_name=tex_name, info=info,
                         allow_adjust=False,)

        self.center = dummify(center)  # CURRENTLY NOT IN USE

        self.param_list.extend([self.center])

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
    The delay class.

    Delay allows to impose a predefined and fixed "delay" (in either steps or
    seconds) for an input variable. The amount of delay must be a scalar and has
    to be given when instantiating the Delay class when defining the model.

    Delay implements an internal memorize to store past variable values.

    The default delay mode is `step` but can be set to `time`.  In the `time`
    mode, the value at the ``current time - delay`` will be interpolated based
    on the two nearest times and values.

    Delay can be applied to a state or an algebraic variable. The exported
    variable is named ``<INSTANCE_NAME>_v``, where `<INSTANCE_NAME>` is the name
    of the Delay instance.
    """

    def __init__(self, u, mode='step', delay=0,
                 name=None, tex_name=None, info=None):

        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)

        if mode not in ('step', 'time'):
            raise ValueError(f'mode {mode} is invalid. Must be in "step" or "time"')

        self.u = u
        self.mode = mode
        self.delay = delay
        self.export_flags = ['v']
        self.export_flags_tex = ['v']
        self.input_list.extend([u])

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
    Compute the average value of a BaseVar over a period of time or a number of
    simulation steps.

    Average is based on the memory implemented in the Delay class. The same
    modes as in Delay are supported.

    The output of the Average class is named ``<INSTANCE_NAME>_v``, where
    `<INSTANCE_NAME>` is the instance name of Average.
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
    Compute the derivative of a variable using numerical differentiation.

    Derivative is based on the storage implemented in the Delay class. The delay
    is set to `1` step so that the current and the previous step are used.

    A simple first order derivative is computed using ``u(t) - u(t-1) / tstep``,
    where ``tstep`` is the current step size.

    Derivative is intended to be used for algebraic variables because of
    discontinuity. It can be applied to a state variable, but one should instead
    implement the right-hand side equation of the state variable in an algebraic
    equation to obtain the accurate derivative.

    Alternatively, the washout filter (:py:class:`andes.core.block.Washout`) can
    be used to implement a numerically stable derivative.

    The output of the Derivative class is named ``<INSTANCE_NAME>_v`` just like
    :py:class:`Delay`.

    """

    def __init__(self, u, name=None, tex_name=None, info=None):
        Delay.__init__(self, u=u, mode='step', delay=1,
                       name=name, tex_name=tex_name, info=info)

    def check_var(self, dae_t, *args, **kwargs):
        """
        Calculate the numerical differentiation.

        .. note::

            Very small derivatives (< 1e-8) could cause numerical problems
            (chattering).
        """

        Delay.check_var(self, dae_t, *args, **kwargs)

        if (dae_t == 0) or (self.rewind is True):
            # Need to reset the output to zero following a rewind
            self.v[:] = 0

        else:
            self.v[:] = (self._v_mem[:, 1] - self._v_mem[:, 0]) / (self.t[1] - self.t[0])
            self.v[np.where(np.abs(self.v) < 1e-8)] = 0


class Sampling(Discrete):
    """
    Sample and hold

    Sample an input variable periodically at the given time interval and hold
    the value until the next sample time.

    For example, this class can be used to implement a 4-second sampling of the
    AGC signal.

    The output of `Sampling` is named ``<INSTANCE_NAME>_v``, where
    `<INSTANCE_NAME>` is the Sampling instance name.

    """

    def __init__(self, u, interval=1.0, offset=0.0, name=None, tex_name=None, info=None):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info)

        self.u = u
        self.interval = interval
        self.offset = offset

        self.export_flags = ['v']
        self.export_flags_tex = ['v']
        self.input_list.extend([self.u])

        self.has_check_var = True

        self.v = np.array([0])
        self._last_t = np.array([0])
        self._last_v = np.array([0])
        self.indices = np.array([0])

        self.rewind = False

    def list2array(self, n):
        """
        Allocate memory and set all ``_last_v`` internal storage to zeros.
        """

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

        - If time progresses and `dae_t` is a multiple of `period`, update `_last_v` and then `v`.
        Record `_last_t`.

        - If time does not progress, update `v`.

        - If rewinds, restore `_last_v` to `v`.

        """

        self.rewind = False

        if dae_t == 0:  # initial step
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


class ShuntAdjust(Discrete):
    """
    Class for adjusting switchable shunts.

    Parameters
    ----------
    v : BaseVar
        Voltage measurement
    lower : BaseParam
        Lower voltage bound
    upper : BaseParam
        Upper voltage bound
    bsw : SwBlock
        SwBlock instance for susceptance
    gsw : SwBlock
        SwBlock instance for conductance
    dt : NumParam
        Delay time
    u : NumParam
        Connection status
    min_iter : int
        Minimum iteration number to enable shunt switching
    err_tol : float
        Minimum iteration tolerance to enable switching
    """

    def __init__(self, *, v, lower, upper, bsw, gsw, dt, u, enable=True,
                 min_iter=2, err_tol=1e-2,
                 name=None, tex_name=None, info=None, no_warn=False):
        Discrete.__init__(self, name=name, tex_name=tex_name, info=info,
                          no_warn=no_warn)

        self.v = v
        self.lower = lower
        self.upper = upper

        self.bsw = bsw
        self.gsw = gsw
        self.dt = dt
        self.u = u
        self.enable = enable
        self.min_iter = min_iter
        self.err_tol = err_tol

        self.has_check_var = True
        self.input_list.extend([self.v])
        self.param_list.extend([self.lower, self.upper, self.u])

        self.t_last = None
        self.t_enable = None
        self.direction = None

    def check_var(self, dae_t, *args, niter=None, err=None, **kwargs):
        """
        Check voltage and perform shunt switching.

        Parameters
        ----------
        niter : int or None
            Current iteration step
        """
        if not self.enable:
            return

        if self.t_last is None:
            self.t_last = np.zeros_like(self.v.v)
            self.t_enable = np.ones_like(self.v.v, dtype=int)
            self.direction = np.zeros_like(self.v.v, dtype=int)

        if not self.check_iter_err(niter=niter, err=err):
            return

        # determine the shunt +/- direction
        self.direction[:] = 0
        self.direction[np.logical_and(self.v.v < self.lower.v,
                                      self.bsw.sel < self.bsw.maxsel)] = 1
        self.direction[np.logical_and(self.v.v > self.upper.v,
                                      self.bsw.sel > 0)] = -1
        self.direction *= self.u.v.astype(int)  # consider online statuses

        if not np.any(self.direction):
            return

        # allow unlimited switching in power flow
        if dae_t == 0.0:
            self.t_enable[:] = 1.0
        # consider delay for time-domain simulation
        else:
            self.t_enable[:] = (dae_t - self.t_last - self.dt.v) >= 0
            self.direction[:] *= self.t_enable

        if not np.any(self.direction):
            return

        logger.debug("--- Shunt Switch at t=%.6g, niter=%d ---", dae_t, niter)
        logger.debug("Delay enable flags=%s, adjusted levels=%s.",
                     self.t_enable, self.direction)
        logger.debug("Bus voltage=%s", self.v.v)
        logger.debug("Before: b=%s, g=%s", self.bsw.v, self.gsw.v)

        self.bsw.adjust(self.direction)
        self.gsw.adjust(self.direction)
        self.t_last[self.direction != 0] = dae_t

        logger.debug("After: b=%s, g=%s", self.bsw.v, self.gsw.v)
