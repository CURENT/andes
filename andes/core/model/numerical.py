"""
Numerical part of a model.
"""

import logging
import pprint
from collections import OrderedDict
from textwrap import wrap
from typing import Callable, Union

import numpy as np
import scipy as sp

from andes.shared import jac_full_names, numba
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)


class ModelNumerical:

    def __init__(self, model) -> None:
        self.model = model

    def l_update_var(self, dae_t, *args, niter=None, err=None, **kwargs):
        """
        Call the ``check_var`` method of discrete components to update the internal status flags.

        The function is variable-dependent and should be called before updating equations.

        Returns
        -------
        None
        """
        for instance in self.discrete.values():
            if instance.has_check_var:
                instance.check_var(dae_t=dae_t, niter=niter, err=err)

    def l_check_eq(self, init=False, niter=0, **kwargs):
        """
        Call the ``check_eq`` method of discrete components to update equation-dependent flags.

        This function should be called after equation updates.
        AntiWindup limiters use it to append pegged states to the ``x_set`` list.

        Returns
        -------
        None
        """
        if init:
            for instance in self.discrete.values():
                if instance.has_check_eq:
                    instance.check_eq(allow_adjust=self.config.allow_adjust,
                                      adjust_lower=self.config.adjust_lower,
                                      adjust_upper=self.config.adjust_upper,
                                      niter=0
                                      )
        else:
            for instance in self.discrete.values():
                if instance.has_check_eq:
                    instance.check_eq(niter=niter)

    def s_update(self):
        """
        Update service equation values.

        This function is only evaluated at initialization. Service values are
        updated sequentially. The ``v`` attribute of services will be assigned
        at a new memory address.
        """
        for name, instance in self.services.items():
            if name in self.calls.s:
                func = self.calls.s[name]
                if callable(func):
                    self.get_inputs(refresh=True)
                    # NOTE:
                    # Use new assignment due to possible size change.
                    # Always make a copy and make the RHS a 1-d array
                    instance.v = np.array(func(*self.s_args[name]),
                                          dtype=instance.vtype).ravel()
                else:
                    instance.v = np.array(func, dtype=instance.vtype).ravel()

                # convert to an array if the return of lambda function is a scalar
                if isinstance(instance.v, (int, float)):
                    instance.v = np.ones(self.n, dtype=instance.vtype) * instance.v

                elif isinstance(instance.v, np.ndarray) and len(instance.v) == 1:
                    instance.v = np.ones(self.n, dtype=instance.vtype) * instance.v

            # --- Very Important ---
            # the numerical call of a `ConstService` should only depend on previously
            #   evaluated variables.
            func = instance.v_numeric
            if func is not None and callable(func):
                kwargs = self.get_inputs(refresh=True)
                instance.v = func(**kwargs).copy().astype(instance.vtype)  # performs type conv.

        if self.flags.s_num is True:
            kwargs = self.get_inputs(refresh=True)
            self.s_numeric(**kwargs)

        # Block-level `s_numeric` not supported.
        self.get_inputs(refresh=True)

    def s_update_var(self):
        """
        Update values of :py:class:`andes.core.service.VarService`.
        """

        if len(self.services_var):
            kwargs = self.get_inputs()
            # evaluate `v_str` functions for sequential VarService
            for name, instance in self.services_var_seq.items():
                if instance.v_str is not None:
                    func = self.calls.s[name]
                    if callable(func):
                        instance.v[:] = func(*self.s_args[name])

            # apply non-sequential services from ``v_str``:w
            if callable(self.calls.sns):
                ret = self.calls.sns(*self.sns_args)
                for idx, instance in enumerate(self.services_var_nonseq.values()):
                    instance.v[:] = ret[idx]

            # Apply individual `v_numeric`
            for instance in self.services_var.values():
                if instance.v_numeric is None:
                    continue
                if callable(instance.v_numeric):
                    instance.v[:] = instance.v_numeric(**kwargs)

        if self.flags.sv_num is True:
            kwargs = self.get_inputs()
            self.s_numeric_var(**kwargs)

        # Block-level `s_numeric_var` not supported.

    def s_update_post(self):
        """
        Update post-initialization services.
        """

        kwargs = self.get_inputs()
        if len(self.services_post):
            for name, instance in self.services_post.items():
                func = self.calls.s[name]
                if callable(func):
                    instance.v[:] = func(*self.s_args[name])

            for instance in self.services_post.values():
                func = instance.v_numeric
                if func is not None and callable(func):
                    instance.v[:] = func(**kwargs)

    def _init_wrap(self, x0, params):
        """
        A wrapper for converting the initialization equations into standard forms g(x) = 0, where x is an array.
        """
        vars_input = []
        for i, _ in enumerate(self.cache.iter_vars.values()):
            vars_input.append(x0[i * self.n: (i + 1) * self.n])

        ret = np.ravel(self.calls.init_std(vars_input, params))
        return ret

    def store_sparse_pattern(self):
        """
        Store rows and columns of the non-zeros in the Jacobians for building the sparsity pattern.

        This function converts the internal 0-indexed equation/variable address to the numerical addresses for
        the loaded system.

        Calling sequence:
        For each Jacobian name, `fx`, `fy`, `gx` and `gy`, store by
        a) generated constant and variable Jacobians
        c) user-provided constant and variable Jacobians,
        d) user-provided block constant and variable Jacobians

        Notes
        -----
        If `self.n == 0`, skipping this function will avoid appending empty lists/arrays and
        non-empty values, which, as a combination, is not accepted by `kvxopt.spmatrix`.
        """

        self.triplets.clear_ijv()
        if self.n == 0:  # do not check `self.in_use` here
            return

        if self.flags.address is False:
            return

        # store model-level user-defined Jacobians
        if self.flags.j_num is True:
            self.j_numeric()

        # store and merge user-defined Jacobians in blocks
        for instance in self.blocks.values():
            if instance.flags.j_num is True:
                instance.j_numeric()
                self.triplets.merge(instance.triplets)

        # for all combinations of Jacobian names (fx, fxc, gx, gxc, etc.)
        for j_name in jac_full_names:
            for idx, val in enumerate(self.calls.vjac[j_name]):

                row_name, col_name = self._jac_eq_var_name(j_name, idx)
                row_idx = self.__dict__[row_name].a
                col_idx = self.__dict__[col_name].a

                if len(row_idx) != len(col_idx):
                    logger.error(f'row {row_name}, row_idx: {row_idx}')
                    logger.error(f'col {col_name}, col_idx: {col_idx}')
                    raise ValueError(f'{self.class_name}: non-matching row_idx and col_idx')

                # Note:
                # n_elem: number of elements in the equation or variable
                # It does not necessarily equal to the number of devices of the model
                # For example, `COI.omega_sub.n` depends on the number of generators linked to the COI
                # and is likely different from `COI.n`

                n_elem = self.__dict__[row_name].n

                if j_name[-1] == 'c':
                    value = val * np.ones(n_elem)
                else:
                    value = np.zeros(n_elem)

                self.triplets.append_ijv(j_name, row_idx, col_idx, value)

    def _jac_eq_var_name(self, j_name, idx):
        """
        Get the equation and variable name for a Jacobian type based on the absolute index.
        """
        var_names_list = list(self.all_vars().keys())

        eq_names = {'f': var_names_list[:len(self.cache.states_and_ext)],
                    'g': var_names_list[len(self.cache.states_and_ext):]}

        row = self.calls.ijac[j_name][idx]
        col = self.calls.jjac[j_name][idx]

        try:
            row_name = eq_names[j_name[0]][row]  # where jname[0] is the equation name in ("f", "g")
            col_name = var_names_list[col]
        except IndexError as e:
            logger.error("Generated code outdated. Run `andes prepare -i` to re-generate.")
            raise e

        return row_name, col_name

    def f_update(self):
        """
        Evaluate differential equations.

        Notes
        -----
        In-place equations: added to the corresponding DAE array.
        Non-in-place equations: in-place set to internal array to
        overwrite old values (and avoid clearing).
        """
        if callable(self.calls.f):
            f_ret = self.calls.f(*self.f_args)
            for i, var in enumerate(self.cache.states_and_ext.values()):
                if var.e_inplace:
                    var.e += f_ret[i]
                else:
                    var.e[:] = f_ret[i]

        kwargs = self.get_inputs()
        # user-defined numerical calls defined in the model
        if self.flags.f_num is True:
            self.f_numeric(**kwargs)

        # user-defined numerical calls in blocks
        for instance in self.blocks.values():
            if instance.flags.f_num is True:
                instance.f_numeric(**kwargs)

    def g_update(self):
        """
        Evaluate algebraic equations.
        """
        if callable(self.calls.g):
            g_ret = self.calls.g(*self.g_args)
            for i, var in enumerate(self.cache.algebs_and_ext.values()):
                if var.e_inplace:
                    var.e += g_ret[i]
                else:
                    var.e[:] = g_ret[i]

        kwargs = self.get_inputs()
        # numerical calls defined in the model
        if self.flags.g_num is True:
            self.g_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            if instance.flags.g_num is True:
                instance.g_numeric(**kwargs)

    def j_update(self):
        """
        Update Jacobian elements.

        Values are stored to ``Model.triplets[jname]``, where ``jname`` is a jacobian name.

        Returns
        -------
        None
        """
        for jname, jfunc in self.calls.j.items():
            ret = jfunc(*self.j_args[jname])

            for idx, fun in enumerate(self.calls.vjac[jname]):
                try:
                    self.triplets.vjac[jname][idx][:] = ret[idx]
                except (ValueError, IndexError, FloatingPointError) as e:
                    row_name, col_name = self._jac_eq_var_name(jname, idx)
                    logger.error('%s: error calculating or storing Jacobian <%s>: j_idx=%s, d%s / d%s',
                                 self.class_name, jname, idx, row_name, col_name)

                    raise e

    def post_init_check(self):
        """
        Post init checking. Warns if values of `InitChecker` are not True.
        """
        self.get_inputs(refresh=True)

        if self.system.config.warn_abnormal:
            for item in self.services_icheck.values():
                item.check()

    def numba_jitify(self, parallel=False, cache=True, nopython=False):
        """
        Convert equation residual calls, Jacobian calls, and variable service
        calls into JIT compiled functions.

        This function can be enabled by setting ``System.config.numba = 1``.
        """

        if self.system.config.numba != 1:
            return

        if self.flags.jited is True:
            return

        kwargs = {'parallel': parallel,
                  'cache': cache,
                  'nopython': nopython,
                  }

        self.calls.f = to_jit(self.calls.f, **kwargs)
        self.calls.g = to_jit(self.calls.g, **kwargs)
        self.calls.sns = to_jit(self.calls.sns, **kwargs)

        for jname in self.calls.j:
            self.calls.j[jname] = to_jit(self.calls.j[jname], **kwargs)

        for name, instance in self.services_var_seq.items():
            if instance.v_str is not None:
                self.calls.s[name] = to_jit(self.calls.s[name], **kwargs)

        self.flags.jited = True

    def precompile(self):
        """
        Trigger numba compilation for this model.

        This function requires the system to be setup, i.e.,
        memory allocated for storage.
        """

        self.get_inputs()
        if self.n == 0:
            self.mock_refresh_inputs()
        self.refresh_inputs_arg()

        if callable(self.calls.f):
            self.calls.f(*self.f_args)

        if callable(self.calls.g):
            self.calls.g(*self.g_args)

        if callable(self.calls.sns):
            self.calls.sns(*self.sns_args)

        for jname, jfunc in self.calls.j.items():
            jfunc(*self.j_args[jname])

        for name, instance in self.services_var_seq.items():
            if instance.v_str is not None:
                self.calls.s[name](*self.s_args[name])

    def init(self, routine):
        """
        Numerical initialization of a model.

        Initialization sequence:
        1. Sequential initialization based on the order of definition
        2. Use Newton-Krylov method for iterative initialization
        3. Custom init
        """

        # evaluate `ConstService` and `VarService`
        self.s_update()
        self.s_update_var()

        # find out if variables need to be initialized for `routine`
        flag_name = routine + '_init'

        if not hasattr(self.flags, flag_name) or getattr(self.flags, flag_name) is None:
            do_init = getattr(self.flags, routine)
        else:
            do_init = getattr(self.flags, flag_name)

        sys_debug = self.system.options.get("init")

        logger.debug('========== %s has <%s> = %s ==========',
                     self.class_name, flag_name, do_init)

        if do_init:
            kwargs = self.get_inputs(refresh=True)

            logger.debug('Initialization sequence:')
            seq_str = ' -> '.join([str(i) for i in self.calls.init_seq])
            logger.debug('\n'.join(wrap(seq_str, 70)))
            logger.debug("%s: assignment initialization phase begins", self.class_name)

            for idx, name in enumerate(self.calls.init_seq):
                debug_flag = sys_debug or (name in self.debug_equations)

                # single variable - do assignment
                if isinstance(name, str):
                    _log_init_debug(debug_flag,
                                    "%s: entering <%s> assignment init",
                                    self.class_name, name)

                    instance = self.__dict__[name]
                    if instance.discrete is not None:
                        _log_init_debug(debug_flag, "%s: evaluate discrete <%s>", name, instance.discrete)
                        _eval_discrete(instance, self.config.allow_adjust,
                                       self.config.adjust_lower, self.config.adjust_upper)

                    if instance.v_str is not None:
                        arg_print = OrderedDict()
                        if debug_flag:
                            for a, b in zip(self.calls.ia_args[name], self.ia_args[name]):
                                arg_print[a] = b

                        if not instance.v_str_add:
                            # assignment is for most variable initialization
                            _log_init_debug(debug_flag, "%s: new values will be assigned (=)", name)
                            instance.v[:] = self.calls.ia[name](*self.ia_args[name])

                        else:
                            # in-place add initial values.
                            # Voltage compensators can set part of the `v` of exciters.
                            # Exciters will then set the bus voltage part.
                            _log_init_debug(debug_flag, "%s: new values will be in-place added (+=)", name)
                            instance.v[:] += self.calls.ia[name](*self.ia_args[name])

                        arg_print[name] = instance.v

                        if debug_flag:
                            for key, val in arg_print.items():
                                if isinstance(val, (int, float, np.floating, np.integer)) or \
                                        isinstance(val, np.ndarray) and val.ndim == 0:

                                    arg_print[key] = val * np.ones_like(instance.v)
                                if isinstance(val, np.ndarray) and val.dtype == complex:
                                    arg_print[key] = [str(i) for i in val]

                            tab = Tab(title="v_str of %s is '%s'" % (name, instance.v_str),
                                      header=["idx", *self.calls.ia_args[name], name],
                                      data=list(zip(self.idx.v, *arg_print.values())),
                                      )
                            _log_init_debug(debug_flag, tab.draw())

                    # single variable iterative solution
                    if name in self.calls.ii:
                        _log_init_debug(debug_flag,
                                        "\n%s: entering <%s> iterative init",
                                        self.class_name,
                                        pprint.pformat(name))

                        self.solve_iter(name, kwargs)

                        _log_init_debug(debug_flag,
                                        "%s new values are %s", name, self.__dict__[name].v)

                # multiple variables, iterative
                else:
                    _log_init_debug(debug_flag, "\n%s: entering <%s> iterative init",
                                    self.class_name, name)

                    for vv in name:
                        instance = self.__dict__[vv]

                        if instance.discrete is not None:
                            _log_init_debug(debug_flag, "%s: evaluate discrete <%s>",
                                            name, instance.discrete)

                            _eval_discrete(instance, self.config.allow_adjust,
                                           self.config.adjust_lower, self.config.adjust_upper)
                        if instance.v_str is not None:
                            _log_init_debug(debug_flag, "%s: v_str = %s", vv, instance.v_str)

                            arg_print = OrderedDict()
                            if debug_flag:
                                for a, b in zip(self.calls.ia_args[vv], self.ia_args[vv]):
                                    arg_print[a] = b
                                _log_init_debug(debug_flag, pprint.pformat(arg_print))

                            instance.v[:] = self.calls.ia[vv](*self.ia_args[vv])

                    _log_init_debug(debug_flag, "\n%s: entering <%s> iterative init", self.class_name,
                                    pprint.pformat(name))
                    self.solve_iter(name, kwargs)

                    for vv in name:
                        instance = self.__dict__[vv]
                        _log_init_debug(debug_flag, "%s new values are %s", vv, instance.v)

            # call custom variable initializer after generated init
            kwargs = self.get_inputs(refresh=True)
            self.v_numeric(**kwargs)

        # call post initialization checking
        self.post_init_check()

        self.flags.initialized = True

    def solve_iter(self, name, kwargs):
        """
        Solve iterative initialization.
        """
        for pos in range(self.n):
            logger.debug("%s: iterative init for %s, device pos = %s",
                         self.class_name, name, pos)

            self.solve_iter_single(name, kwargs, pos)

    def solve_iter_single(self, name, inputs, pos):
        """
        Solve iterative initialization for one given device.
        """

        if isinstance(name, str):
            x0 = inputs[name][pos]
            name_concat = name
        else:
            x0 = np.ravel([inputs[item][pos] for item in name])
            name_concat = '_'.join(name)

        rhs = self.calls.ii[name_concat]
        jac = self.calls.ij[name_concat]
        ii_args = self.ii_args[name_concat]
        ij_args = self.ij_args[name_concat]

        # iteration setup
        maxiter = self.system.TDS.config.max_iter
        eps = self.system.TDS.config.tol
        niter = 0
        solved = False

        while niter < maxiter:
            logger.debug("iteration %s:", niter)

            i_args = [item[pos] for item in ii_args]  # all variables of one device at a time
            j_args = [item[pos] for item in ij_args]

            b = np.ravel(rhs(*i_args))
            A = jac(*j_args)

            logger.debug("A:\n%s", A)
            logger.debug("b:\n%s", b)

            mis = np.max(np.abs(b))
            if mis <= eps:
                solved = True
                break

            inc = - sp.linalg.lu_solve(sp.linalg.lu_factor(A), b)
            if np.isnan(inc).any():
                if self.u.v[pos] != 0:
                    logger.debug("%s: nan ignored in iterations for offline device pos = %s",
                                 self.class_name, pos)
                else:
                    logger.error("%s: nan error detected in iterations for device pos = %s",
                                 self.class_name, pos)
                break

            x0 += inc
            logger.debug("solved x0:\n%s", x0)

            for idx, item in enumerate(name):
                inputs[item][pos] = x0[idx]

            niter += 1

        if solved:
            for idx, item in enumerate(name):
                inputs[item][pos] = x0[idx]
        else:
            logger.warning(f"{self.class_name}: iterative initialization failed for {self.idx.v[pos]}.")

    def register_debug_equation(self, var_name):
        """
        Helper function to register a variable for debugging the initialization.

        This function needs to be called before calling ``TDS.init()``, and
        logging level needs to be set to ``DEBUG``.
        """

        self.debug_equations.append(var_name)


def _eval_discrete(instance, allow_adjust=True,
                   adjust_lower=False, adjust_upper=False):
    """
    Evaluate discrete components associated with a variable instance.
    Calls ``check_var()`` on the discrete components.

    For variables associated with limiters, limiters need to be evaluated
    before variable initialization.
    However, if any limit is hit, initialization is likely to fail.

    Parameters
    ----------
    instance : BaseVar
        instance of a variable
    allow_adjust : bool, optional
        True to enable overall adjustment
    adjust_lower : bool, optional
        True to adjust lower limits to the input values
    adjust_upper : bool, optional
        True to adjust upper limits to the input values

    """

    if not isinstance(instance.discrete, (list, tuple, set)):
        dlist = (instance.discrete,)
    else:
        dlist = instance.discrete
    for d in dlist:
        d.check_var(allow_adjust=allow_adjust,
                    adjust_lower=adjust_lower,
                    adjust_upper=adjust_upper,
                    is_init=True,
                    )
        d.check_eq(allow_adjust=allow_adjust,
                   adjust_lower=adjust_lower,
                   adjust_upper=adjust_upper,
                   is_init=True,
                   )


def _log_init_debug(debug_flag, *args):
    """
    Helper function to log initialization debug message.
    """

    if debug_flag is True:
        logger.debug(*args)


def to_jit(func: Union[Callable, None],
           parallel: bool = False,
           cache: bool = False,
           nopython: bool = False,
           ):
    """
    Helper function for converting a function to a numba jit-compiled function.

    Note that this function will be compiled just-in-time when first called,
    based on the argument types.
    """

    if func is not None:
        return numba.jit(func,
                         parallel=parallel,
                         cache=cache,
                         nopython=nopython,
                         )

    return func
