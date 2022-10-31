#  [ANDES] (C)2015-2022 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#

"""
Module for ANDES models.
"""

import logging
import pprint
from collections import OrderedDict
from textwrap import wrap
from typing import Callable, Iterable, Union

import numpy as np
import scipy as sp
from andes.core.block import Block
from andes.core.common import Config, JacTriplet, ModelFlags
from andes.core.discrete import Discrete
from andes.core.documenter import Documenter
from andes.core.model.modelcache import ModelCache
from andes.core.model.modelcall import ModelCall
from andes.core.param import ExtParam
from andes.core.service import (ApplyFunc, BackRef, BaseService, ConstService,
                                DeviceFinder, ExtService, FlagValue,
                                InitChecker, NumReduce, NumRepeat, NumSelect,
                                ParamCalc, PostInitService, RandomService,
                                Replace, SubsService, SwBlock, VarService)
from andes.core.symprocessor import SymProcessor
from andes.core.var import Algeb, BaseVar, ExtAlgeb, ExtState, State
from andes.shared import jac_full_names, numba
from andes.utils.func import list_flatten
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)


class Model:
    r"""
    Base class for power system DAE models.


    After subclassing `ModelData`, subclass `Model`` to complete a DAE model.
    Subclasses of `Model` define DAE variables, services, and other types of
    parameters, in the constructor ``__init__``.

    Attributes
    ----------
    num_params : OrderedDict
        {name: instance} of numerical parameters, including internal and
        external ones

    Examples
    --------
    Take the static PQ as an example, the subclass of `Model`, `PQ`, should look
    like ::

        class PQ(PQData, Model):
            def __init__(self, system, config):
                PQData.__init__(self) Model.__init__(self, system, config)

    Since `PQ` is calling the base class constructors, it is meant to be the
    final class and not further derived. It inherits from `PQData` and `Model`
    and must call constructors in the order of `PQData` and `Model`. If the
    derived class of `Model` needs to be further derived, it should only derive
    from `Model` and use a name ending with `Base`. See
    :py:class:`andes.models.synchronous.genbase.GENBase`.

    Next, in `PQ.__init__`, set proper flags to indicate the routines in which
    the model will be used ::

        self.flags.update({'pflow': True})

    Currently, flags `pflow` and `tds` are supported. Both are `False` by
    default, meaning the model is neither used in power flow nor in time-domain
    simulation. **A very common pitfall is forgetting to set the flag**.

    Next, the group name can be provided. A group is a collection of models with
    common parameters and variables. Devices' idx of all models in the same
    group must be unique. To provide a group name, use ::

        self.group = 'StaticLoad'

    The group name must be an existing class name in
    :py:mod:`andes.models.group`. The model will be added to the specified group
    and subject to the variable and parameter policy of the group. If not
    provided with a group class name, the model will be placed in the
    `Undefined` group.

    Next, additional configuration flags can be added. Configuration flags for
    models are load-time variables, specifying the behavior of a model. They can
    be exported to an `andes.rc` file and automatically loaded when creating the
    `System`. Configuration flags can be used in equation strings, as long as
    they are numerical values. To add config flags, use ::

        self.config.add(OrderedDict((('pq2z', 1), )))

    It is recommended to use `OrderedDict` instead of `dict`, although the
    syntax is verbose. Note that booleans should be provided as integers (1 or
    0), since `True` or `False` is interpreted as a string when loaded from the
    `rc` file and will cause an error.

    Next, it's time for variables and equations! The `PQ` class does not have
    internal variables itself. It uses its `bus` parameter to fetch the
    corresponding `a` and `v` variables of buses. Equation wise, it imposes an
    active power and a reactive power load equation.

    To define external variables from `Bus`, use ::

            self.a = ExtAlgeb(model='Bus', src='a',
                              indexer=self.bus, tex_name=r'\theta')
            self.v = ExtAlgeb(model='Bus', src='v',
                              indexer=self.bus, tex_name=r'V')

    Refer to the subsection Variables for more details.

    The simplest `PQ` model will impose constant P and Q, coded as ::

            self.a.e_str = "u * p"
            self.v.e_str = "u * q"

    where the `e_str` attribute is the equation string attribute. `u` is the
    connectivity status. Any parameter, config, service or variable can be used
    in equation strings.

    Three additional scalars can be used in equations: - ``dae_t`` for the
    current simulation time (can be used if the model has flag `tds`). -
    ``sys_f`` for system frequency (from ``system.config.freq``). - ``sys_mva``
    for system base mva (from ``system.config.mva``).

    The above example is overly simplified. Our `PQ` model wants a feature to
    switch itself to a constant impedance if the voltage is out of the range
    `(vmin, vmax)`. To implement this, we need to introduce a discrete component
    called `Limiter`, which yields three arrays of binary flags, `zi`, `zl`, and
    `zu` indicating in-range, below lower-limit, and above upper-limit,
    respectively.

    First, create an attribute `vcmp` as a `Limiter` instance ::

            self.vcmp = Limiter(u=self.v, lower=self.vmin, upper=self.vmax,
                                 enable=self.config.pq2z)

    where `self.config.pq2z` is a flag to turn this feature on or off. After
    this line, we can use `vcmp_zi`, `vcmp_zl`, and `vcmp_zu` in other equation
    strings. ::

            self.a.e_str = "u * (p0 * vcmp_zi + " \
                           "p0 * vcmp_zl * (v ** 2 / vmin ** 2) + " \
                           "p0 * vcmp_zu * (v ** 2 / vmax ** 2))"

            self.v.e_str = "u * (q0 * vcmp_zi + " \
                           "q0 * vcmp_zl * (v ** 2 / vmin ** 2) + "\
                           "q0 * vcmp_zu * (v ** 2 / vmax ** 2))"

    Note that `PQ.a.e_str` can use the three variables from `vcmp` even before
    defining `PQ.vcmp`, as long as `PQ.vcmp` is defined, because `vcmp_zi` is
    just a string literal in `e_str`.

    The two equations above implement a piece-wise power injection equation. It
    selects the original power demand if within range, and uses the calculated
    power when out of range.

    Finally, to let ANDES pick up the model, the model name needs to be added to
    `models/__init__.py`. Follow the examples in the `OrderedDict`, where the
    key is the file name, and the value is the class name.
    """

    def __init__(self, system=None, config=None):
        self.system = system

        # duplicate attributes from ModelData. Keep for now.
        self.n = 0
        self.group = 'Undefined'

        # params and vars that exist in the group but not in this model
        # normally empty but can be used in special cases to bypass
        # shared param/var checking
        self.group_param_exception = list()
        self.group_var_exception = list()

        if not hasattr(self, 'num_params'):
            self.num_params = OrderedDict()
        if not hasattr(self, 'cache'):
            self.cache = ModelCache()

        # variables
        self.states = OrderedDict()  # internal states
        self.states_ext = OrderedDict()  # external states
        self.algebs = OrderedDict()  # internal algebraic variables
        self.algebs_ext = OrderedDict()  # external algebraic vars
        self.vars_decl_order = OrderedDict()  # variable in the order of declaration

        self.params_ext = OrderedDict()  # external parameters

        self.discrete = OrderedDict()  # discrete comp.
        self.blocks = OrderedDict()  # blocks

        self.services = OrderedDict()  # service/temporary variables
        self.services_var = OrderedDict()  # variable services updated each step/iter
        self.services_var_seq = OrderedDict()
        self.services_var_nonseq = OrderedDict()
        self.services_post = OrderedDict()  # post-initialization storage services
        self.services_subs = OrderedDict()  # to-be-substituted services
        self.services_icheck = OrderedDict()  # post-initialization check services
        self.services_ref = OrderedDict()  # BackRef
        self.services_fnd = OrderedDict()  # services to find/add devices
        self.services_ext = OrderedDict()  # external services (to be retrieved)
        self.services_ops = OrderedDict()  # operational services (for special usages)

        self.tex_names = OrderedDict((('dae_t', 't_{dae}'),
                                      ('sys_f', 'f_{sys}'),
                                      ('sys_mva', 'S_{b,sys}'),
                                      ))

        # Model behavior flags
        self.flags = ModelFlags()

        # `in_use` is used by models with `BackRef` when not reference
        self.in_use = True  # True if this model is in use, False removes this model from all calls

        self.config = Config(name=self.class_name)  # `config` that can be exported
        if config is not None:
            self.config.load(config)

        # basic configs
        self.config.add(OrderedDict((('allow_adjust', 1),
                                    ('adjust_lower', 0),
                                    ('adjust_upper', 1),
                                     )))

        self.config.add_extra("_help",
                              allow_adjust='allow adjusting upper or lower limits',
                              adjust_lower='adjust lower limit',
                              adjust_upper='adjust upper limit',
                              )

        self.config.add_extra("_alt",
                              allow_adjust=(0, 1),
                              adjust_lower=(0, 1),
                              adjust_upper=(0, 1),
                              )

        self.calls = ModelCall()  # callback and LaTeX string storage
        self.triplets = JacTriplet()  # Jacobian triplet storage
        self.syms = SymProcessor(self)  # symbolic processor instance
        self.docum = Documenter(self)

        # cached class attributes
        self.cache.add_callback('all_vars', self._all_vars)
        self.cache.add_callback('iter_vars', self._iter_vars)
        self.cache.add_callback('input_vars', self._input_vars)
        self.cache.add_callback('output_vars', self._output_vars)

        self.cache.add_callback('all_vars_names', self._all_vars_names)
        self.cache.add_callback('all_params', self._all_params)
        self.cache.add_callback('all_params_names', self._all_params_names)
        self.cache.add_callback('algebs_and_ext', self._algebs_and_ext)
        self.cache.add_callback('states_and_ext', self._states_and_ext)
        self.cache.add_callback('services_and_ext', self._services_and_ext)
        self.cache.add_callback('vars_ext', self._vars_ext)
        self.cache.add_callback('vars_int', self._vars_int)
        self.cache.add_callback('v_getters', self._v_getters)
        self.cache.add_callback('v_adders', self._v_adders)
        self.cache.add_callback('v_setters', self._v_setters)
        self.cache.add_callback('e_adders', self._e_adders)
        self.cache.add_callback('e_setters', self._e_setters)

        self._input = OrderedDict()  # cached dictionary of inputs
        self._input_z = OrderedDict()  # discrete flags, storage only.
        self._rhs_f = OrderedDict()  # RHS of external f
        self._rhs_g = OrderedDict()  # RHS of external g

        self.f_args = []
        self.g_args = []  # argument value lists
        self.j_args = dict()
        self.s_args = OrderedDict()
        self.ia_args = OrderedDict()
        self.ii_args = OrderedDict()
        self.ij_args = OrderedDict()

        self.coeffs = dict()  # pu conversion coefficient storage
        self.bases = dict()   # base storage, such as Vn, Vb, Zn, Zb
        self.debug_equations = list()  # variable names for debugging corresponding equation

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the model instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        Block attributes will be exported and registered recursively.
        """
        if isinstance(value, Algeb):
            self.algebs[key] = value
        elif isinstance(value, ExtAlgeb):
            self.algebs_ext[key] = value
        elif isinstance(value, State):
            self.states[key] = value
        elif isinstance(value, ExtState):
            self.states_ext[key] = value
        elif isinstance(value, ExtParam):
            self.params_ext[key] = value
        elif isinstance(value, Discrete):
            self.discrete[key] = value
        elif isinstance(value, ConstService):  # services with only `v_str`
            self.services[key] = value
            # store VarService in an additional dict
            if isinstance(value, VarService):
                self.services_var[key] = value
                if value.sequential:
                    self.services_var_seq[key] = value
                else:
                    self.services_var_nonseq[key] = value
            elif isinstance(value, PostInitService):
                self.services_post[key] = value
        elif isinstance(value, SubsService):
            self.services_subs[key] = value
        elif isinstance(value, DeviceFinder):
            self.services_fnd[key] = value
        elif isinstance(value, BackRef):
            self.services_ref[key] = value
        elif isinstance(value, ExtService):
            self.services_ext[key] = value
        elif isinstance(value, (NumRepeat, NumReduce, NumSelect,
                                FlagValue, RandomService,
                                SwBlock,
                                ParamCalc, Replace, ApplyFunc)):
            self.services_ops[key] = value
        elif isinstance(value, InitChecker):
            self.services_icheck[key] = value
        elif isinstance(value, Block):
            self.blocks[key] = value
            # pull in sub-variables from control blocks
            if value.namespace == 'local':
                prepend = value.name + '_'
                tex_append = value.tex_name
            else:
                prepend = ''
                tex_append = ''

            for var_name, var_instance in value.export().items():
                var_instance.name = f'{prepend}{var_name}'
                var_instance.tex_name = f'{var_instance.tex_name}_{{{tex_append}}}'
                self.__setattr__(var_instance.name, var_instance)

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names while instantiating the class.

        This function assigns `owner` to the model itself, assigns the name and tex_name.
        """
        if isinstance(value, (BaseVar, BaseService, Discrete, Block)):
            if not value.owner:
                value.owner = self
            if not value.name:
                value.name = key
            if not value.tex_name:
                value.tex_name = key
            if key in self.__dict__:
                logger.warning(f"{self.class_name}: redefinition of member <{key}>. Likely a modeling error.")

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key : str
            name of the attribute
        value : [BaseVar, BaseService, Discrete, Block]
            value of the attribute
        """

        self._check_attribute(key, value)

        # store the variable declaration order
        if isinstance(value, BaseVar):
            value.id = len(self._all_vars())  # NOT in use yet
            self.vars_decl_order[key] = value

        self._register_attribute(key, value)

        super(Model, self).__setattr__(key, value)

    def idx2uid(self, idx):
        """
        Convert idx to the 0-indexed unique index.

        Parameters
        ----------
        idx : array-like, numbers, or str
            idx of devices

        Returns
        -------
        list
            A list containing the unique indices of the devices
        """
        if idx is None:
            logger.debug("idx2uid returned None for idx None")
            return None
        if isinstance(idx, (float, int, str, np.integer, np.floating)):
            return self._one_idx2uid(idx)
        elif isinstance(idx, Iterable):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self._one_idx2uid(i) if i is not None else None
                    for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def _one_idx2uid(self, idx):
        """
        Helper function for checking if an idx exists and
        converting it to uid.
        """

        if idx not in self.uid:
            raise KeyError("<%s>: device not exist with idx=%s." %
                           (self.class_name, idx))

        return self.uid[idx]

    def set_backref(self, name, from_idx, to_idx):
        """
        Helper function for setting idx-es to ``BackRef``.
        """

        if name not in self.services_ref:
            return

        uid = self.idx2uid(to_idx)
        self.services_ref[name].v[uid].append(from_idx)

    def get(self, src: str, idx, attr: str = 'v', allow_none=False, default=0.0):
        """
        Get the value of an attribute of a model property.

        The return value is ``self.<src>.<attr>[idx]``

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        allow_none : bool
            True to allow None values in the indexer
        default : float
            If `allow_none` is true, the default value to use for None indexer.

        Returns
        -------
        array-like
            ``self.<src>.<attr>[idx]``

        """
        uid = self.idx2uid(idx)
        if isinstance(self.__dict__[src].__dict__[attr], list):
            if isinstance(uid, Iterable):
                if not allow_none and (uid is None or None in uid):
                    raise KeyError('None not allowed in uid/idx. Enable through '
                                   '`allow_none` and provide a `default` if needed.')
                return [self.__dict__[src].__dict__[attr][i] if i is not None else default
                        for i in uid]

        return self.__dict__[src].__dict__[attr][uid]

    def set(self, src, idx, attr, value):
        """
        Set the value of an attribute of a model property.

        Performs ``self.<src>.<attr>[idx] = value``. This method will not modify
        the input values from the case file that have not been converted to the
        system base. As a result, changes applied by this method will not affect
        the dumped case file.

        To alter parameters and reflect it in the case file, use :meth:`alter`
        instead.

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The internal attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            New values to be set

        Returns
        -------
        bool
            True when successful.
        """
        uid = self.idx2uid(idx)
        self.__dict__[src].__dict__[attr][uid] = value
        return True

    def alter(self, src, idx, value):
        """
        Alter values of input parameters or constant service.

        If the method operates on an input parameter, the new data should be in
        the same base as that in the input file. This function will convert the
        new value to per unit in the system base.

        The values for storing the input data, i.e., the ``vin`` field of the
        parameter, will be overwritten, thus the update will be reflected in the
        dumped case file.

        Parameters
        ----------
        src : str
            The parameter name to alter
        idx : str, float, int
            The device to alter
        value : float
            The desired value
        """
        instance = self.__dict__[src]

        if hasattr(instance, 'vin') and (instance.vin is not None):
            self.set(src, idx, 'vin', value)
            instance.v[:] = instance.vin * instance.pu_coeff
        else:
            self.set(src, idx, 'v', value)

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
        if len(self._input) == 0 or refresh:
            self.refresh_inputs()
            self.refresh_inputs_arg()

        return self._input

    def refresh_inputs(self):
        """
        This is the helper function to refresh inputs.

        The functions collects object references into ``OrderedDict``
        `self._input` and `self._input_z`.

        Returns
        -------
        None

        """
        # The order of inputs: `all_params` and then `all_vars`, finally `config`
        # the below sequence should correspond to `self.cache.all_params_names`
        for instance in self.num_params.values():
            self._input[instance.name] = instance.v

        for instance in self.services.values():
            self._input[instance.name] = instance.v

        for instance in self.services_ext.values():
            self._input[instance.name] = instance.v

        for instance in self.services_ops.values():
            self._input[instance.name] = instance.v

        # discrete flags
        for instance in self.discrete.values():
            for name, val in zip(instance.get_names(), instance.get_values()):
                self._input[name] = val
                self._input_z[name] = val

        # append all variable values
        for instance in self.cache.all_vars.values():
            self._input[instance.name] = instance.v

        # append config variables as arrays
        for key, val in self.config.as_dict(refresh=True).items():
            self._input[key] = np.array(val)

        # zeros and ones
        self._input['__zeros'] = np.zeros(self.n)
        self._input['__ones'] = np.ones(self.n)
        self._input['__falses'] = np.full(self.n, False)
        self._input['__trues'] = np.full(self.n, True)

        # --- below are numpy scalars ---
        # update`dae_t` and `sys_f`, and `sys_mva`
        self._input['sys_f'] = np.array(self.system.config.freq, dtype=float)
        self._input['sys_mva'] = np.array(self.system.config.mva, dtype=float)
        self._input['dae_t'] = self.system.dae.t

    def mock_refresh_inputs(self):
        """
        Use mock data to fill the inputs.

        This function is used to generate input data of the desired type
        to trigget JIT compilation.
        """

        self.get_inputs()
        mock_arr = np.array([1.])

        for key in self._input.keys():
            try:
                key_ndim = self._input[key].ndim
            except AttributeError:
                logger.error(key)
                logger.error(self.class_name)

            key_type = self._input[key].dtype

            if key_ndim == 0:
                self._input[key] = mock_arr.reshape(()).astype(key_type)
            elif key_ndim == 1:
                self._input[key] = mock_arr.astype(key_type)

            else:
                raise NotImplementedError("Unkonwn input data dimension %s" % key_ndim)

    def refresh_inputs_arg(self):
        """
        Refresh inputs for each function with individual argument list.
        """
        self.f_args = list()
        self.g_args = list()
        self.j_args = dict()
        self.s_args = dict()
        self.ii_args = dict()
        self.ia_args = dict()
        self.ij_args = dict()

        self.f_args = [self._input[arg] for arg in self.calls.f_args]
        self.g_args = [self._input[arg] for arg in self.calls.g_args]
        self.sns_args = [self._input[arg] for arg in self.calls.sns_args]

        # each value below is a dict
        mapping = {
            'j_args': self.j_args,
            's_args': self.s_args,
            'ia_args': self.ia_args,
            'ii_args': self.ii_args,
            'ij_args': self.ij_args,
        }

        for key, val in mapping.items():
            source = self.calls.__dict__[key]
            for name in source:
                val[name] = [self._input[arg] for arg in source[name]]

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
                    instance.v = np.ravel(np.array(func(*self.s_args[name]), dtype=instance.vtype))
                else:
                    instance.v = np.ravel(np.array(func, dtype=instance.vtype))

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
        var_names_list = list(self.cache.all_vars.keys())

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

    def get_init_order(self):
        """
        Get variable initialization order and send to `logger.info`.
        """
        out = []
        for name in self.vars_decl_order.keys():
            out.append(name)

        logger.info(f'Initialization order: \n{", ".join(out)}')

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

    def get_times(self):
        """
        Get event switch_times from `TimerParam`.

        Returns
        -------
        list
            A list containing all switching times defined in TimerParams
        """
        out = []
        if self.n > 0:
            for instance in self.timer_params.values():
                out.append(instance.v)

        return out

    def switch_action(self, dae_t):
        """
        Call the switch actions.

        Parameters
        ----------
        dae_t : float
            Current simulation time

        Returns
        -------
        None

        Warnings
        --------
        Timer exported from blocks are supposed to work
        but have not been tested.
        """
        for timer in self.timer_params.values():
            if timer.callback is not None:
                timer.callback(timer.is_time(dae_t))

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def _all_vars(self):
        """
        An OrderedDict of States, ExtStates, Algebs, ExtAlgebs
        """
        return OrderedDict(list(self.states.items()) +
                           list(self.states_ext.items()) +
                           list(self.algebs.items()) +
                           list(self.algebs_ext.items())
                           )

    def _iter_vars(self):
        """
        Variables to be iteratively initialized
        """
        all_vars = OrderedDict(self.cache.all_vars)
        for name, instance in self.cache.all_vars.items():
            if not instance.v_iter:
                all_vars.pop(name)
        return all_vars

    def _all_vars_names(self):
        out = []
        for instance in self.cache.all_vars.values():
            out += instance.get_names()
        return out

    def _all_params(self):
        # the service stuff should not be moved to variables.
        return OrderedDict(list(self.num_params.items()) +
                           list(self.services.items()) +
                           list(self.services_ext.items()) +
                           list(self.services_ops.items()) +
                           list(self.services_subs.items()) +
                           list(self.discrete.items())
                           )

    def _all_params_names(self):
        out = []
        for instance in self.cache.all_params.values():
            out += instance.get_names()
        return out

    def _algebs_and_ext(self):
        return OrderedDict(list(self.algebs.items()) +
                           list(self.algebs_ext.items()))

    def _states_and_ext(self):
        return OrderedDict(list(self.states.items()) +
                           list(self.states_ext.items()))

    def _services_and_ext(self):
        return OrderedDict(list(self.services.items()) +
                           list(self.services_ext.items()))

    def _vars_ext(self):
        return OrderedDict(list(self.states_ext.items()) +
                           list(self.algebs_ext.items()))

    def _vars_int(self):
        return OrderedDict(list(self.states.items()) +
                           list(self.algebs.items()))

    def _v_getters(self):
        out = OrderedDict()
        for name, var in self.cache.all_vars.items():
            if var.v_inplace:
                continue
            out[name] = var
        return out

    def _v_adders(self):
        out = OrderedDict()
        for name, var in self.cache.all_vars.items():
            if var.v_inplace is True:
                continue
            if var.v_str is None and var.v_iter is None:
                continue
            if var.v_setter is True:
                continue

            out[name] = var
        return out

    def _v_setters(self):
        out = OrderedDict()
        for name, var in self.cache.all_vars.items():
            if var.v_inplace is True:
                continue
            if var.v_str is None and var.v_iter is None:
                continue
            if var.v_setter is False:
                continue

            out[name] = var
        return out

    def _e_adders(self):
        out = OrderedDict()
        for name, var in self.cache.all_vars.items():
            if var.e_inplace is True:
                continue
            if var.e_str is None:
                continue
            if var.e_setter is True:
                continue

            out[name] = var
        return out

    def _e_setters(self):
        out = OrderedDict()
        for name, var in self.cache.all_vars.items():
            if var.e_inplace is True:
                continue
            if var.e_str is None:
                continue
            if var.e_setter is False:
                continue

            out[name] = var
        return out

    def _input_vars(self):
        out = list()
        for name, var in self.cache.all_vars.items():
            if var.is_input:
                out.append(name)
        return out

    def _output_vars(self):
        out = list()
        for name, var in self.cache.all_vars.items():
            if var.is_output:
                out.append(name)
        return out

    def set_in_use(self):
        """
        Set the `in_use` attribute. Called at the end of ``System.collect_ref``.

        This function is overloaded by models with `BackRef` to disable calls when no model is referencing.
        Models with no back references will have internal variable addresses assigned but external addresses
        being empty.

        For internal equations that have external variables, the row indices will be non-zeros, while the col
        indices will be empty, which causes an error when updating Jacobians.

        Setting `self.in_use` to False when `len(back_ref_instance.v) == 0` avoids this error. See COI.
        """
        self.in_use = True

    def list2array(self):
        """
        Convert all the value attributes ``v`` to NumPy arrays.

        Value attribute arrays should remain in the same address afterwards.
        Namely, all assignments to value array should be operated in place (e.g., with [:]).
        """

        for instance in self.num_params.values():
            instance.to_array()

        for instance in self.cache.services_and_ext.values():
            instance.assign_memory(self.n)

        for instance in self.discrete.values():
            instance.list2array(self.n)

    def a_reset(self):
        """
        Reset addresses to empty and reset flags.address to ``False``.
        """
        for var in self.cache.all_vars.values():
            var.reset()
        self.flags.address = False
        self.flags.initialized = False

    def e_clear(self):
        """
        Clear equation value arrays associated with all internal variables.
        """
        for instance in self.cache.all_vars.values():
            if instance.e_inplace:
                continue
            instance.e[:] = 0

    def v_numeric(self, **kwargs):
        """
        Custom variable initialization function.
        """
        pass

    def g_numeric(self, **kwargs):
        """
        Custom gcall functions. Modify equations directly.
        """
        pass

    def f_numeric(self, **kwargs):
        """
        Custom fcall functions. Modify equations directly.
        """
        pass

    def s_numeric(self, **kwargs):
        """
        Custom service value functions. Modify ``Service.v`` directly.
        """
        pass

    def s_numeric_var(self, **kwargs):
        """
        Custom variable service value functions. Modify ``VarService.v`` directly.

        This custom numerical function is evaluated at each step/iteration before equation update.
        """
        pass

    def j_numeric(self, **kwargs):
        """
        Custom numeric update functions.

        This function should append indices to `_ifx`, `_jfx`, and append anonymous functions to `_vfx`.
        It is only called once by `store_sparse_pattern`.
        """
        pass

    def doc(self, max_width=78, export='plain'):
        """
        Retrieve model documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def prepare(self, quick=False, pycode_path=None, yapf_pycode=False):
        """
        Symbolic processing and code generation.
        """

        logger.debug("Generating code for %s", self.class_name)

        self.calls.md5 = self.get_md5()

        self.syms.generate_symbols()
        self.syms.generate_subs_expr()
        self.syms.generate_equations()
        self.syms.generate_services()
        self.syms.generate_jacobians()
        self.syms.generate_init()

        self.syms.generate_pycode(pycode_path=pycode_path,
                                  yapf_pycode=yapf_pycode,
                                  )
        if quick is False:
            self.syms.generate_pretty_print()

    def get_md5(self):
        """
        Return the md5 hash of concatenated equation strings.
        """
        import hashlib
        md5 = hashlib.md5()

        for name in self.cache.all_params.keys():
            md5.update(str(name).encode())

        for name in self.config.as_dict().keys():
            md5.update(str(name).encode())

        for name, item in self.cache.all_vars.items():
            md5.update(str(name).encode())

            if item.v_str is not None:
                md5.update(str(item.v_str).encode())
            if item.v_iter is not None:
                md5.update(str(item.v_iter).encode())
            if item.e_str is not None:
                md5.update(str(item.e_str).encode())
            if item.diag_eps is not None:
                md5.update(str(item.diag_eps).encode())

        for name, item in self.services.items():
            md5.update(str(name).encode())

            if item.v_str is not None:
                md5.update(str(item.v_str).encode())

            md5.update(str(int(item.sequential)).encode())

        for name, item in self.discrete.items():
            md5.update(str(name).encode())
            md5.update(str(','.join(item.export_flags)).encode())

        return md5.hexdigest()

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

    def __repr__(self):
        dev_text = 'device' if self.n == 1 else 'devices'

        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'

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

    def externalize(self):
        """
        Externalize internal data as a snapshot.
        """
        pass

    def internalize(self):
        """
        Internalize snapshot data.
        """
        pass

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


def _log_init_debug(debug_flag, *args):
    """
    Helper function to log initialization debug message.
    """

    if debug_flag is True:
        logger.debug(*args)
