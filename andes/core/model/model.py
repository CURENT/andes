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
from collections import OrderedDict
from typing import Iterable

import numpy as np
from andes.core.block import Block
from andes.core.common import Config, ModelFlags
from andes.core.discrete import Discrete
from andes.core.service import BaseService
from andes.core.var import BaseVar
from andes.utils.func import list_flatten

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

    def create_config(self, name, config_obj=None):
        """
        Create a Config object for this model by loading the ConfigParser object
        and inserting this model's default configs.
        """

        config = Config(name)

        if config_obj is not None:
            config.load(config_obj)

        # basic configs
        config.add(OrderedDict((('allow_adjust', 1),
                                ('adjust_lower', 0),
                                ('adjust_upper', 1),
                                )))

        config.add_extra("_help",
                         allow_adjust='allow adjusting upper or lower limits',
                         adjust_lower='adjust lower limit',
                         adjust_upper='adjust upper limit',
                         )

        config.add_extra("_alt",
                         allow_adjust=(0, 1),
                         adjust_lower=(0, 1),
                         adjust_upper=(0, 1),
                         )

        return config

    def set_config(self, config_manager):
        """
        Store a ConfigManager object and register the model's config creation
        method.
        """

        self.config = config_manager

        config_manager.register(self.class_name, self.create_config)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the model instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        Block attributes will be exported and registered recursively.
        """
        # pull in sub-variables from control blocks

        if isinstance(value, Block):
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
                logger.warning(f"{self.class_name}: redefinition of member <{key}>.")

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
        for instance in self.all_vars().values():
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
        to trigger JIT compilation.
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

    # def get_init_order(self):
    #     """
    #     Get variable initialization order and send to `logger.info`.
    #     """
    #     out = []
    #     for name in self.vars_decl_order.keys():
    #         out.append(name)

    #     logger.info(f'Initialization order: \n{", ".join(out)}')

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

    def set_in_use(self):
        """
        Set the `in_use` attribute. Called at the end of ``System.collect_ref``.

        This function is overloaded by models with `BackRef` to disable calls
        when no model is referencing. Models with no back references will have
        internal variable addresses assigned but external addresses being empty.

        For internal equations that have external variables, the row indices
        will be non-zeros, while the col indices will be empty, which causes an
        error when updating Jacobians.

        Setting `self.in_use` to False when `len(back_ref_instance.v) == 0`
        avoids this error. See COI.
        """
        self.in_use = True

    def list2array(self):
        """
        Convert all the value attributes ``v`` to NumPy arrays.

        Value attribute arrays should remain in the same address afterwards.
        Namely, all assignments to value array should be operated in place
        (e.g., with [:]).
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
        for var in self.all_vars().values():
            var.reset()
        self.flags.address = False
        self.flags.initialized = False

    def e_clear(self):
        """
        Clear equation value arrays associated with all internal variables.
        """
        for instance in self.all_vars().values():
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

    # TODO: move to top level
    # def doc(self, max_width=78, export='plain'):
    #     """
    #     Retrieve model documentation as a string.
    #     """
    #     return self.docum.get(max_width=max_width, export=export)

    def __repr__(self):
        dev_text = 'device' if self.n == 1 else 'devices'

        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'
