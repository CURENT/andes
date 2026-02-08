"""
Composition-based evaluator for Model equation and Jacobian updates.
"""

import logging

import numpy as np

from andes.shared import jac_full_names

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator class that encapsulates numerical update methods for a Model.
    """

    def __init__(self, model):
        self.model = model

    def get_inputs(self, refresh=False):
        """
        Get an OrderedDict of the inputs to the numerical function calls.

        Parameters
        ----------
        refresh : bool
            Refresh the values in the dictionary.
            This is only used when the memory addresses of arrays change.
            After initialization, all array assignments are in place.
            To avoid overhead, refresh should not be used after initialization.

        Returns
        -------
        OrderedDict
            The input name and value array pairs in an OrderedDict

        Notes
        -----
        `dae.t` is now a numpy.ndarray which has stable memory.
        There is no need to refresh `dat_t` in this version.

        """
        model = self.model
        if len(model._input) == 0 or refresh:
            self.refresh_inputs()
            self.refresh_inputs_arg()

        return model._input

    def refresh_inputs(self):
        """
        This is the helper function to refresh inputs.

        The functions collects object references into ``OrderedDict``
        `self._input` and `self._input_z`.

        Returns
        -------
        None

        """
        model = self.model
        # The order of inputs: `all_params` and then `all_vars`, finally `config`
        # the below sequence should correspond to `self.cache.all_params_names`
        for instance in model.num_params.values():
            model._input[instance.name] = instance.v

        for instance in model.services.values():
            model._input[instance.name] = instance.v

        for instance in model.services_ext.values():
            model._input[instance.name] = instance.v

        for instance in model.services_ops.values():
            model._input[instance.name] = instance.v

        # discrete flags
        for instance in model.discrete.values():
            for name, val in zip(instance.get_names(), instance.get_values()):
                model._input[name] = val
                model._input_z[name] = val

        # append all variable values
        for instance in model.cache.all_vars.values():
            model._input[instance.name] = instance.v

        # append config variables as arrays
        for key, val in model.config.as_dict(refresh=True).items():
            model._input[key] = np.array(val)

        # zeros and ones
        model._input['__zeros'] = np.zeros(model.n)
        model._input['__ones'] = np.ones(model.n)
        model._input['__falses'] = np.full(model.n, False)
        model._input['__trues'] = np.full(model.n, True)

        # --- below are numpy scalars ---
        # update`dae_t` and `sys_f`, and `sys_mva`
        model._input['sys_f'] = np.array(model.system.config.freq, dtype=float)
        model._input['sys_mva'] = np.array(model.system.config.mva, dtype=float)
        model._input['dae_t'] = model.system.dae.t

    def refresh_inputs_arg(self):
        """
        Refresh inputs for each function with individual argument list.
        """
        model = self.model
        model.f_args = list()
        model.g_args = list()
        model.j_args = dict()
        model.s_args = dict()
        model.ii_args = dict()
        model.ia_args = dict()
        model.ij_args = dict()

        model.f_args = [model._input[arg] for arg in model.calls.f_args]
        model.g_args = [model._input[arg] for arg in model.calls.g_args]
        model.sns_args = [model._input[arg] for arg in model.calls.sns_args]

        # each value below is a dict
        mapping = {
            'j_args': model.j_args,
            's_args': model.s_args,
            'ia_args': model.ia_args,
            'ii_args': model.ii_args,
            'ij_args': model.ij_args,
        }

        for key, val in mapping.items():
            source = model.calls.__dict__[key]
            for name in source:
                val[name] = [model._input[arg] for arg in source[name]]

    def l_update_var(self, dae_t, *args, niter=None, err=None, **kwargs):
        """
        Call the ``check_var`` method of discrete components to update the internal status flags.

        The function is variable-dependent and should be called before updating equations.

        Returns
        -------
        None
        """
        model = self.model
        for instance in model.discrete.values():
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
        model = self.model
        if init:
            for instance in model.discrete.values():
                if instance.has_check_eq:
                    instance.check_eq(allow_adjust=model.config.allow_adjust,
                                      adjust_lower=model.config.adjust_lower,
                                      adjust_upper=model.config.adjust_upper,
                                      niter=0
                                      )
        else:
            for instance in model.discrete.values():
                if instance.has_check_eq:
                    instance.check_eq(niter=niter)

    def s_update(self):
        """
        Update service equation values.

        This function is only evaluated at initialization. Service values are
        updated sequentially. The ``v`` attribute of services will be assigned
        at a new memory address.
        """
        model = self.model
        for name, instance in model.services.items():
            if name in model.calls.s:
                func = model.calls.s[name]
                if callable(func):
                    self.get_inputs(refresh=True)
                    # NOTE:
                    # Use new assignment due to possible size change.
                    # Always make a copy and make the RHS a 1-d array
                    instance.v = np.array(func(*model.s_args[name]),
                                          dtype=instance.vtype).ravel()
                else:
                    instance.v = np.array(func, dtype=instance.vtype).ravel()

                # convert to an array if the return of lambda function is a scalar
                if isinstance(instance.v, (int, float)):
                    instance.v = np.ones(model.n, dtype=instance.vtype) * instance.v

                elif isinstance(instance.v, np.ndarray) and len(instance.v) == 1:
                    instance.v = np.ones(model.n, dtype=instance.vtype) * instance.v

            # --- Very Important ---
            # the numerical call of a `ConstService` should only depend on previously
            #   evaluated variables.
            func = instance.v_numeric
            if func is not None and callable(func):
                kwargs = self.get_inputs(refresh=True)
                instance.v = func(**kwargs).copy().astype(instance.vtype)  # performs type conv.

        if model.flags.s_num is True:
            kwargs = self.get_inputs(refresh=True)
            model.s_numeric(**kwargs)

        # Block-level `s_numeric` not supported.
        self.get_inputs(refresh=True)

    def s_update_var(self):
        """
        Update values of :py:class:`andes.core.service.VarService`.
        """
        model = self.model

        if len(model.services_var):
            kwargs = self.get_inputs()
            # evaluate `v_str` functions for sequential VarService
            for name, instance in model.services_var_seq.items():
                if instance.v_str is not None:
                    func = model.calls.s[name]
                    if callable(func):
                        instance.v[:] = func(*model.s_args[name])

            # apply non-sequential services from ``v_str``:w
            if callable(model.calls.sns):
                ret = model.calls.sns(*model.sns_args)
                for idx, instance in enumerate(model.services_var_nonseq.values()):
                    instance.v[:] = ret[idx]

            # Apply individual `v_numeric`
            for instance in model.services_var.values():
                if instance.v_numeric is None:
                    continue
                if callable(instance.v_numeric):
                    instance.v[:] = instance.v_numeric(**kwargs)

        if model.flags.sv_num is True:
            kwargs = self.get_inputs()
            model.s_numeric_var(**kwargs)

        # Block-level `s_numeric_var` not supported.

    def s_update_post(self):
        """
        Update post-initialization services.
        """
        model = self.model

        kwargs = self.get_inputs()
        if len(model.services_post):
            for name, instance in model.services_post.items():
                func = model.calls.s[name]
                if callable(func):
                    instance.v[:] = func(*model.s_args[name])

            for instance in model.services_post.values():
                func = instance.v_numeric
                if func is not None and callable(func):
                    instance.v[:] = func(**kwargs)

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
        model = self.model

        model.triplets.clear_ijv()
        if model.n == 0:  # do not check `self.in_use` here
            return

        if model.flags.address is False:
            return

        # skip models where all devices have been replaced
        if model._all_replaced:
            return

        # store model-level user-defined Jacobians
        if model.flags.j_num is True:
            model.j_numeric()

        # store and merge user-defined Jacobians in blocks
        for instance in model.blocks.values():
            if instance.flags.j_num is True:
                instance.j_numeric()
                model.triplets.merge(instance.triplets)

        # for all combinations of Jacobian names (fx, fxc, gx, gxc, etc.)
        for j_name in jac_full_names:
            for idx, val in enumerate(model.calls.vjac[j_name]):

                row_name, col_name = self._jac_eq_var_name(j_name, idx)
                row_idx = model.__dict__[row_name].a
                col_idx = model.__dict__[col_name].a

                if len(row_idx) != len(col_idx):
                    logger.error(f'row {row_name}, row_idx: {row_idx}')
                    logger.error(f'col {col_name}, col_idx: {col_idx}')
                    raise ValueError(f'{model.class_name}: non-matching row_idx and col_idx')

                # Note:
                # n_elem: number of elements in the equation or variable
                # It does not necessarily equal to the number of devices of the model
                # For example, `COI.omega_sub.n` depends on the number of generators linked to the COI
                # and is likely different from `COI.n`

                n_elem = model.__dict__[row_name].n

                if j_name[-1] == 'c':
                    value = val * np.ones(n_elem)
                else:
                    value = np.zeros(n_elem)

                model.triplets.append_ijv(j_name, row_idx, col_idx, value)

    def _jac_eq_var_name(self, j_name, idx):
        """
        Get the equation and variable name for a Jacobian type based on the absolute index.
        """
        model = self.model
        var_names_list = list(model.cache.all_vars.keys())

        eq_names = {'f': var_names_list[:len(model.cache.states_and_ext)],
                    'g': var_names_list[len(model.cache.states_and_ext):]}

        row = model.calls.ijac[j_name][idx]
        col = model.calls.jjac[j_name][idx]

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
        model = self.model
        if model._all_replaced:
            return

        if callable(model.calls.f):
            f_ret = model.calls.f(*model.f_args)
            for i, var in enumerate(model.cache.states_and_ext.values()):
                if var.e_inplace:
                    var.e += f_ret[i]
                else:
                    var.e[:] = f_ret[i]

        kwargs = self.get_inputs()
        # user-defined numerical calls defined in the model
        if model.flags.f_num is True:
            model.f_numeric(**kwargs)

        # user-defined numerical calls in blocks
        for instance in model.blocks.values():
            if instance.flags.f_num is True:
                instance.f_numeric(**kwargs)

    def g_update(self):
        """
        Evaluate algebraic equations.
        """
        model = self.model
        if model._all_replaced:
            return

        if callable(model.calls.g):
            g_ret = model.calls.g(*model.g_args)
            for i, var in enumerate(model.cache.algebs_and_ext.values()):
                if var.e_inplace:
                    var.e += g_ret[i]
                else:
                    var.e[:] = g_ret[i]

        kwargs = self.get_inputs()
        # numerical calls defined in the model
        if model.flags.g_num is True:
            model.g_numeric(**kwargs)

        # numerical calls in blocks
        for instance in model.blocks.values():
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
        model = self.model
        if model._all_replaced:
            return

        for jname, jfunc in model.calls.j.items():
            ret = jfunc(*model.j_args[jname])

            for idx, _ in enumerate(model.calls.vjac[jname]):
                try:
                    model.triplets.vjac[jname][idx][:] = ret[idx]
                except (ValueError, IndexError, FloatingPointError) as e:
                    row_name, col_name = self._jac_eq_var_name(jname, idx)
                    logger.error('%s: error calculating or storing Jacobian <%s>: j_idx=%s, d%s / d%s',
                                 model.class_name, jname, idx, row_name, col_name)

                    raise e

    def e_clear(self):
        """
        Clear equation value arrays associated with all internal variables.
        """
        model = self.model
        for instance in model.cache.all_vars.values():
            if instance.e_inplace:
                continue
            instance.e[:] = 0
