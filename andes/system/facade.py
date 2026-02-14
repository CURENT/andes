"""
System class for power system data and methods.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: system/facade.py

import logging
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Tuple, Union

import andes.io
from andes.core import AntiWindup, Model, ConnMan
from andes.core.service import ConstService
from andes.io.streaming import Streaming
from andes.shared import NCPUS_PHYSICAL, jac_names, np, spmatrix
from andes.system.codegen import CodegenManager
from andes.system.config_runtime import SystemConfigRuntime
from andes.system.dae_compactor import DAECompactor
from andes.system.registry import RegistryLoader
from andes.utils.misc import elapsed
from andes.utils.tab import Tab
from andes.variables import DAE, FileMan

logger = logging.getLogger(__name__)


class ExistingModels:
    """
    Storage class for existing models
    """

    def __init__(self):
        self.pflow = OrderedDict()
        self.tds = OrderedDict()   # if a model needs to be initialized before TDS, set `flags.tds = True`
        self.pflow_tds = OrderedDict()


class System:
    """
    System contains models and routines for modeling and simulation.

    System contains a several special `OrderedDict` member attributes for housekeeping.
    These attributes include `models`, `groups`, `routines` and `calls` for loaded models, groups,
    analysis routines, and generated numerical function calls, respectively.

    Parameters
    ----------
    no_undill : bool, optional, default=False
        True to disable the call to ``System.undill()`` at the end of object creation.
        False by default.

    autogen_stale : bool, optional, default=True
        True to automatically generate code for stale models.

    Notes
    -----
    System stores model and routine instances as attributes.
    Model and routine attribute names are the same as their class names.
    For example, `Bus` is stored at ``system.Bus``, the power flow calculation routine is at
    ``system.PFlow``, and the numerical DAE instance is at ``system.dae``. See attributes for the list of
    attributes.

    Attributes
    ----------
    dae : andes.variables.dae.DAE
        Numerical DAE storage
    files : andes.variables.fileman.FileMan
        File path storage
    config : andes.core.Config
        System config storage
    models : OrderedDict
        model name and instance pairs
    groups : OrderedDict
        group name and instance pairs
    routines : OrderedDict
        routine name and instance pairs
    """

    def __init__(self,
                 case: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 default_config: Optional[bool] = False,
                 options: Optional[Dict] = None,
                 no_undill: Optional[bool] = False,
                 autogen_stale: Optional[bool] = True,
                 **kwargs
                 ):
        self.name = name
        self.options = {}
        if options is not None:
            self.options.update(options)
        if kwargs:
            self.options.update(kwargs)
        self.calls = OrderedDict()           # a dictionary with model names (keys) and their ``calls`` instance
        self.models = OrderedDict()          # model names and instances
        self.model_aliases = OrderedDict()   # alias: model instance
        self.groups = OrderedDict()          # group names and instances
        self.routines = OrderedDict()        # routine names and instances
        self.switch_times = np.array([])     # an array of ordered event switching times
        self.switch_dict = OrderedDict()     # time: OrderedDict of associated models
        self.with_calls = False              # if generated function calls have been loaded
        self.n_switches = 0                  # number of elements in `self.switch_times`
        self.exit_code = 0                   # command-line exit code, 0 - normal, others - error.

        self.config_runtime = SystemConfigRuntime(self)
        self.config_runtime.initialize(config=config,
                                       config_path=config_path,
                                       default_config=default_config)

        self.exist = ExistingModels()

        self.files = FileMan(case=case, **self.options)    # file path manager
        self.dae = DAE(system=self)                        # numerical DAE storage
        self.streaming = Streaming(self)                   # Dime2 streaming
        self.conn = ConnMan(system=self)                   # connectivity manager
        self.registry = RegistryLoader(self)               # model/group/routine loading
        self.codegen = CodegenManager(self)                # symbolic code generation/loading
        self.dae_compactor = DAECompactor(self)            # DAE algebraic compaction

        # dynamic imports of groups, models and routines
        self.registry.load_all()

        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._adders = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())
        self.antiwindups = list()
        self.no_check_init = list()  # states for which initialization check is omitted
        self.call_stats = defaultdict(dict)  # call statistics storage

        # status propagation graph (built during setup)
        self._status_children = {}     # parent_group -> set of child BackRef names
        self._status_parent_map = {}   # child_model_name -> (param_attr_name, parent_group_name)

        # internal flags
        self.is_setup = False        # if system has been setup

        if not no_undill:
            self.undill(autogen_stale=autogen_stale)

    def _update_config_object(self):
        """
        Deprecated wrapper to :class:`andes.system.config_runtime.SystemConfigRuntime`.
        """
        warnings.warn(
            "System._update_config_object() is deprecated and will be removed in v3.0. "
            "Use system.config_runtime.update_config_object() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.config_runtime.update_config_object()

    def reload(self, case, **kwargs):
        """
        Reload a new case in the same System object.
        """

        self.options.update(kwargs)
        self.files.set(case=case, **kwargs)
        # TODO: clear all flags and empty data
        andes.io.parse(self)
        self.setup()

    def _clear_adder_setter(self):
        """
        Clear adders and setters storage
        """
        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._adders = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())

    def prepare(self, quick=False, incremental=False, models=None, nomp=False, ncpu=NCPUS_PHYSICAL):
        """
        Delegate to :class:`andes.system.codegen.CodegenManager`.
        """
        return self.codegen.prepare(quick=quick,
                                    incremental=incremental,
                                    models=models,
                                    nomp=nomp,
                                    ncpu=ncpu)

    def _mp_prepare(self, models, quick, pycode_path, ncpu):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._mp_prepare() is deprecated and will be removed in v3.0. "
            "Use system.codegen._mp_prepare() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._mp_prepare(models=models,
                                        quick=quick,
                                        pycode_path=pycode_path,
                                        ncpu=ncpu)

    def _finalize_pycode(self, pycode_path):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._finalize_pycode() is deprecated and will be removed in v3.0. "
            "Use system.codegen._finalize_pycode() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._finalize_pycode(pycode_path)

    def _find_stale_models(self):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._find_stale_models() is deprecated and will be removed in v3.0. "
            "Use system.codegen._find_stale_models() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._find_stale_models()

    def _to_orddct(self, model_list):
        """
        Helper function to convert a list of model names to OrderedDict with
        name as keys and model instances as values.
        """

        if isinstance(model_list, OrderedDict):
            return model_list
        if isinstance(model_list, list):
            out = OrderedDict()
            for name in model_list:
                if name not in self.models:
                    logger.error("Model <%s> does not exist. Check your inputs.", name)
                    continue
                out[name] = self.models[name]
            return out
        else:
            raise TypeError("Type %s not recognized" % type(model_list))

    def setup(self):
        """
        Set up system for studies.

        This function is to be called after adding all device data.
        """
        ret = True
        t0, _ = elapsed()

        if self.is_setup:
            logger.warning('System has been setup. Calling setup twice is not allowed.')
            ret = False
            return ret

        self.collect_ref()
        self._build_status_graph()
        self._list2array()     # `list2array` must come before `link_ext_param`
        if not self.link_ext_param():
            ret = False

        self.find_devices()    # find or add required devices
        self._report_param_corrections()

        # === no device addition or removal after this point ===
        self.calc_pu_coeff()   # calculate parameters in system per units
        self.store_existing()  # store models with routine flags

        # assign address at the end before adding devices and processing parameters
        self.set_address(self.exist.pflow)
        self.set_dae_names(self.exist.pflow)        # needs perf. optimization
        self.store_sparse_pattern(self.exist.pflow)
        self.store_adder_setter(self.exist.pflow)

        # init connectivity manager
        self.conn.init()

        if ret is True:
            self.is_setup = True  # set `is_setup` if no error occurred
        else:
            logger.error("System setup failed. Please resolve the reported issue(s).")
            self.exit_code += 1

        _, s = elapsed(t0)
        logger.info('System internal structure set up in %s.', s)

        return ret

    def store_existing(self):
        """
        Store existing models in `System.existing`.

        TODO: Models with `TimerParam` will need to be stored anyway.
        This will allow adding switches on the fly.
        """

        self.exist.pflow = self.find_models('pflow')
        self.exist.tds = self.find_models('tds')
        self.exist.pflow_tds = self.find_models(('tds', 'pflow'))

    def reset(self, force=False):
        """
        Reset to the state after reading data and setup (before power flow).

        Warnings
        --------
        If TDS is initialized, reset will lead to unpredictable state.
        """
        if self.TDS.initialized is True and not force:
            logger.error('Reset failed because TDS is initialized. \nPlease reload the test case to start over.')
            return
        self.dae.reset()
        self.call_models('a_reset', models=self.models)
        self.e_clear(models=self.models)
        self._p_restore()
        self.is_setup = False
        self.setup()

    def add(self, model, param_dict=None, **kwargs):
        """
        Add a device instance for an existing model.

        This methods calls the ``add`` method of `model` and registers the device `idx` to group.
        """
        if model not in self.models and (model not in self.model_aliases):
            logger.warning("<%s> is not an existing model.", model)
            return

        if self.is_setup:
            raise NotImplementedError("Adding devices are not allowed after setup.")

        group_name = self.__dict__[model].group
        group = self.groups[group_name]

        if param_dict is None:
            param_dict = {}
        if kwargs is not None:
            param_dict.update(kwargs)

        # remove `uid` field
        param_dict.pop('uid', None)

        idx = param_dict.pop('idx', None)
        if idx is not None and (not isinstance(idx, str) and np.isnan(idx)):
            idx = None

        idx = group.get_next_idx(idx=idx, model_name=model)
        self.__dict__[model].add(idx=idx, **param_dict)
        group.add(idx=idx, model=self.__dict__[model])

        return idx

    def find_devices(self):
        """
        Add dependent devices for all model based on `DeviceFinder`.
        """

        for mdl in self.models.values():
            if len(mdl.services_fnd) == 0:
                continue

            for fnd in mdl.services_fnd.values():
                fnd.find_or_add(self)

    def set_address(self, models):
        """
        Set addresses for differential and algebraic variables.
        """

        # --- Phase 1: set internal variable addresses ---
        for mdl in models.values():
            if mdl.flags.address is True:
                logger.debug('%s internal address exists', mdl.class_name)
                continue
            if mdl.n == 0:
                continue

            logger.debug('Setting internal address for %s', mdl.class_name)

            collate = mdl.flags.collate
            ndevice = mdl.n

            # get and set internal variable addresses
            xaddr = self.dae.request_address('x', ndevice=ndevice,
                                             nvar=len(mdl.states),
                                             collate=mdl.flags.collate,
                                             )
            yaddr = self.dae.request_address('y', ndevice=ndevice,
                                             nvar=len(mdl.algebs),
                                             collate=mdl.flags.collate,
                                             )

            for idx, item in enumerate(mdl.states.values()):
                item.set_address(xaddr[idx], contiguous=not collate)
            for idx, item in enumerate(mdl.algebs.values()):
                item.set_address(yaddr[idx], contiguous=not collate)

            # observable variable addresses (in dae.b)
            if len(mdl.observables) > 0:
                baddr = self.dae.request_address('b', ndevice=ndevice,
                                                 nvar=len(mdl.observables),
                                                 collate=mdl.flags.collate,
                                                 )
                for idx, item in enumerate(mdl.observables.values()):
                    item.set_address(baddr[idx], contiguous=not collate)

        # --- Phase 2: set external variable addresses ---
        # NOTE:
        # This step will retrieve the number of variables (item.n) for Phase 3.

        for mdl in models.values():
            # handle external groups
            for instance in mdl.cache.vars_ext.values():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError('<%s> is not a model or group name.' % ext_name)

                try:
                    instance.link_external(ext_model)
                except (IndexError, KeyError) as e:
                    logger.error('Error: <%s> cannot retrieve <%s> from <%s> using <%s>:\n  %s',
                                 mdl.class_name, instance.name, instance.model,
                                 instance.indexer.name, repr(e))

        #  --- Phase 3: set external variable RHS addresses ---
        for mdl in models.values():
            if mdl.flags.address is True:
                logger.debug('%s RHS address exists', mdl.class_name)
                continue
            if mdl.n == 0:
                continue

            for item in mdl.states_ext.values():
                # skip if no equation, i.e., no RHS value
                if item.e_str is None:
                    continue
                item.set_address(np.arange(self.dae.p, self.dae.p + item.n))
                self.dae.p += item.n
            for item in mdl.algebs_ext.values():
                if item.e_str is None:
                    continue
                item.set_address(np.arange(self.dae.q, self.dae.q + item.n))
                self.dae.q += item.n

            mdl.flags.address = True

        # allocate memory for DAE arrays
        self.dae.resize_arrays()

        # set `v` and `e` in variables
        self.set_var_arrays(models=models)

        self.dae.alloc_or_extend_names()

    def set_dae_names(self, models):
        """
        Set variable names for differential and algebraic variables,
        right-hand side of external equations, and discrete flags.
        """

        for mdl in models.values():
            _set_xy_name(mdl, mdl.states, (self.dae.x_name, self.dae.x_tex_name))
            _set_xy_name(mdl, mdl.algebs, (self.dae.y_name, self.dae.y_tex_name))

            _set_hi_name(mdl, mdl.states_ext, (self.dae.h_name, self.dae.h_tex_name))
            _set_hi_name(mdl, mdl.algebs_ext, (self.dae.i_name, self.dae.i_tex_name))

            _set_xy_name(mdl, mdl.observables, (self.dae.b_name, self.dae.b_tex_name))

            # build reverse map: addr → (model, var)
            for var in mdl.states.values():
                for addr in var.a:
                    self.dae.x_map[int(addr)] = (mdl, var)
            for var in mdl.algebs.values():
                for addr in var.a:
                    self.dae.y_map[int(addr)] = (mdl, var)

            # add discrete flag names
            if self.TDS.config.store_z == 1:
                _set_z_name(mdl, self.dae, (self.dae.z_name, self.dae.z_tex_name))

    def set_var_arrays(self, models, inplace=True, alloc=True):
        """
        Set arrays (`v` and `e`) for internal variables to access dae arrays in
        place.

        This function needs to be called after de-serializing a System object,
        where the internal variables are incorrectly assigned new memory.

        Parameters
        ----------
        models : OrderedDict, list, Model, optional
            Models to execute.
        inplace : bool
            True to retrieve arrays that share memory with dae
        alloc : bool
            True to allocate for arrays internally
        """

        for mdl in models.values():
            if mdl.n == 0:
                continue

            for var in mdl.cache.vars_int.values():
                var.set_arrays(self.dae, inplace=inplace, alloc=alloc)

            for var in mdl.cache.vars_ext.values():
                var.set_arrays(self.dae, inplace=inplace, alloc=alloc)

            for var in mdl.observables.values():
                var.set_arrays(self.dae, inplace=inplace, alloc=alloc)

    def _init_numba(self, models: OrderedDict):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._init_numba() is deprecated and will be removed in v3.0. "
            "Use system.codegen._init_numba() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._init_numba(models)

    def precompile(self,
                   models: Union[OrderedDict, None] = None,
                   nomp: bool = False,
                   ncpu: int = NCPUS_PHYSICAL):
        """
        Delegate to :class:`andes.system.codegen.CodegenManager`.
        """
        return self.codegen.precompile(models=models, nomp=nomp, ncpu=ncpu)

    def init(self, models: OrderedDict, routine: str):
        """
        Initialize the variables for each of the specified models.

        For each model, the initialization procedure is:

        - Get values for all `ExtService`.
        - Call the model `init()` method, which initializes internal variables.
        - Copy variables to DAE and then back to the model.
        """

        self.codegen._init_numba(models)

        for mdl in models.values():
            # link externals services first
            for instance in mdl.services_ext.values():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError('<%s> is not a model or group name.' % ext_name)

                try:
                    instance.link_external(ext_model)
                except (IndexError, KeyError) as e:
                    logger.error('Error: <%s> cannot retrieve <%s> from <%s> using <%s>:\n  %s',
                                 mdl.class_name, instance.name, instance.model,
                                 instance.indexer.name, repr(e))

            # initialize variables second
            mdl.init(routine=routine)

            self.vars_to_dae(mdl)
            self.vars_to_models()

        self.s_update_post(models)

        # store time constants associated with differential equations
        self._store_tf(models)

    def store_getters(self, models):
        """
        Build ``_getters`` and ``_setters`` so that ``vars_to_models`` and
        ``vars_to_dae`` work during model initialization.

        This is the lightweight subset of ``store_adder_setter`` needed
        before ``system.init()``.  The full ``store_adder_setter`` is
        called after DAE compaction to build ``_adders`` and
        ``antiwindups`` as well.
        """
        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())

        for mdl in models.values():
            if not mdl.n:
                continue
            for var in mdl.cache.v_getters.values():
                self._getters[var.v_code].append(var)
            for var in mdl.cache.v_setters.values():
                self._setters[var.v_code].append(var)
            for var in mdl.cache.e_setters.values():
                self._setters[var.e_code].append(var)

    def store_adder_setter(self, models):
        """
        Store non-inplace adders and setters for variables and equations.
        """
        self._clear_adder_setter()

        for mdl in models.values():
            # Note:
            #   We assume that a Model with no device is not addressed and, therefore,
            #   contains no value in each variable.
            #   It is always true for the current architecture.
            if not mdl.n:
                continue

            # skip models where all devices have been replaced by dynamic models
            if mdl._all_replaced:
                continue

            # Fixes an issue if the cache was manually built but stale
            # after assigning addresses for simulation
            # Assigning memory will affect the cache of `v_adders` and `e_adders`.

            mdl.cache.refresh()

            # ``getters` that retrieve variable values from DAE
            for var in mdl.cache.v_getters.values():
                self._getters[var.v_code].append(var)

            # ``adders`` that add variable values to the DAE array
            for var in mdl.cache.v_adders.values():
                self._adders[var.v_code].append(var)
            for var in mdl.cache.e_adders.values():
                self._adders[var.e_code].append(var)

            # ``setters`` that force set variable values in the DAE array
            for var in mdl.cache.v_setters.values():
                self._setters[var.v_code].append(var)
            for var in mdl.cache.e_setters.values():
                self._setters[var.e_code].append(var)

            # ``antiwindups`` stores all AntiWindup instances
            for item in mdl.discrete.values():
                if isinstance(item, AntiWindup):
                    self.antiwindups.append(item)

        return

    def store_no_check_init(self, models):
        """
        Store differential variables with ``check_init == False``.
        """
        self.no_check_init = list()
        for mdl in models.values():
            if mdl.n == 0:
                continue

            for var in mdl.states.values():
                if var.check_init is False:
                    self.no_check_init.extend(var.a)

    def link_ext_param(self, model=None):
        """
        Retrieve values for ``ExtParam`` for the given models.
        """
        if model is None:
            models = self.models
        else:
            models = self._get_models(model)

        ret = True
        for model in models.values():
            # get external parameters with `link_external` and then calculate the pu coeff
            for instance in model.params_ext.values():
                ext_name = instance.model
                ext_model = self.__dict__[ext_name]

                try:
                    instance.link_external(ext_model)
                except (IndexError, KeyError) as e:
                    logger.error('Error: <%s> cannot retrieve <%s> from <%s> using <%s>:\n  %s',
                                 model.class_name, instance.name, instance.model,
                                 instance.indexer.name, repr(e))
                    ret = False
        return ret

    def calc_pu_coeff(self):
        """
        Perform per unit value conversion.

        This function calculates the per unit conversion factors, stores input
        parameters to `vin`, and perform the conversion.
        """
        # `Sb`, `Vb` and `Zb` are the system base, bus base values
        # `Sn`, `Vn` and `Zn` are the device bases

        Sb = self.config.mva

        for mdl in self.models.values():
            # default Sn to Sb if not provided. Some controllers might not have Sn or Vn.
            if 'Sn' in mdl.__dict__:
                Sn = mdl.Sn.v
            else:
                Sn = Sb

            # If both Vn and Vn1 are not provided, default to Vn = Vb = 1
            # test if is shunt-connected or series-connected to bus, or unconnected to bus
            Vb, Vn = 1, 1
            if 'bus' in mdl.__dict__:
                Vb = self.Bus.get(src='Vn', idx=mdl.bus.v, attr='v')
                Vn = mdl.Vn.v if 'Vn' in mdl.__dict__ else Vb
            elif 'bus1' in mdl.__dict__:
                Vb = self.Bus.get(src='Vn', idx=mdl.bus1.v, attr='v')
                Vn = mdl.Vn1.v if 'Vn1' in mdl.__dict__ else Vb

            Zn = Vn ** 2 / Sn
            Zb = Vb ** 2 / Sb

            # process dc parameter pu conversion
            Vdcb, Vdcn, Idcn = 1, 1, 1
            if 'node' in mdl.__dict__:
                Vdcb = self.Node.get(src='Vdcn', idx=mdl.node.v, attr='v')
                Vdcn = mdl.Vdcn.v if 'Vdcn' in mdl.__dict__ else Vdcb
                Idcn = mdl.Idcn.v if 'Idcn' in mdl.__dict__ else (Sb / Vdcb)
            elif 'node1' in mdl.__dict__:
                Vdcb = self.Node.get(src='Vdcn', idx=mdl.node1.v, attr='v')
                Vdcn = mdl.Vdcn1.v if 'Vdcn1' in mdl.__dict__ else Vdcb
                Idcn = mdl.Idcn.v if 'Idcn' in mdl.__dict__ else (Sb / Vdcb)
            Idcb = Sb / Vdcb
            Rb = Vdcb / Idcb
            Rn = Vdcn / Idcn

            coeffs = {'voltage': Vn / Vb,
                      'power': Sn / Sb,
                      'ipower': Sb / Sn,
                      'current': (Sn / Vn) / (Sb / Vb),
                      'z': Zn / Zb,
                      'y': Zb / Zn,
                      'dc_voltage': Vdcn / Vdcb,
                      'dc_current': Idcn / Idcb,
                      'r': Rn / Rb,
                      'g': Rb / Rn,
                      }

            for prop, coeff in coeffs.items():
                for p in mdl.find_param(prop).values():
                    p.set_pu_coeff(coeff)

            # store coeffs and bases back in models.
            mdl.coeffs = coeffs
            mdl.bases = {'Sn': Sn, 'Sb': Sb, 'Vn': Vn, 'Vb': Vb, 'Zn': Zn, 'Zb': Zb}

    def l_update_var(self, models: OrderedDict, niter=0, err=None):
        """
        Update variable-based limiter discrete states by calling ``l_update_var`` of models.

        This function is must be called before any equation evaluation.
        """
        self.call_models('l_update_var', models,
                         dae_t=self.dae.t, niter=niter, err=err)

    def l_update_eq(self, models:  OrderedDict, init=False, niter=0):
        """
        Update equation-dependent limiter discrete components by calling ``l_check_eq`` of models.
        Force set equations after evaluating equations.

        This function is must be called after differential equation updates.
        """
        self.call_models('l_check_eq', models, init=init, niter=niter)

    def s_update_var(self, models: OrderedDict):
        """
        Update variable services by calling ``s_update_var`` of models.

        This function is must be called before any equation evaluation after
        limiter update function `l_update_var`.
        """
        self.call_models('s_update_var', models)

    def s_update_post(self, models: OrderedDict):
        """
        Update variable services by calling ``s_update_post`` of models.

        This function is called at the end of `System.init()`.
        """
        self.call_models('s_update_post', models)

    def fg_to_dae(self):
        """
        Collect equation values into the DAE arrays.

        Additionally, the function resets the differential equations associated with variables pegged by
        anti-windup limiters.
        """
        self._e_to_dae(('f', 'g'))

        # reset mismatches for islanded buses
        self.g_islands()

        # update variable values set by anti-windup limiters
        for item in self.antiwindups:
            if len(item.x_set) > 0:
                for key, val, _ in item.x_set:
                    np.put(self.dae.x, key, val)

    def f_update(self, models: OrderedDict):
        """
        Call the differential equation update method for models in sequence.

        Notes
        -----
        Updated equation values remain in models and have not been collected into DAE at the end of this step.
        """
        try:
            self.call_models('f_update', models)
        except TypeError as e:
            logger.error("f_update failed. Have you run `andes prepare -i` after updating?")
            raise e

    def g_update(self, models: OrderedDict):
        """
        Call the algebraic equation update method for models in sequence.

        Notes
        -----
        Like `f_update`, updated values have not collected into DAE at the end of the step.
        """
        try:
            self.call_models('g_update', models)
        except TypeError as e:
            logger.error("g_update failed. Have you run `andes prepare -i` after updating?")
            raise e

    def b_update(self, models: OrderedDict):
        """
        Call the observable variable update method for models in sequence.

        This evaluates all Observable variables post-solve and stores
        values in ``dae.b``.
        """
        try:
            self.call_models('b_update', models)
        except TypeError as e:
            logger.error("b_update failed. Have you run `andes prepare -i` after updating?")
            raise e

    def g_islands(self):
        """
        Reset algebraic mismatches for islanded buses.
        """
        if self.Bus.n_islanded_buses == 0:
            return

        self.dae.g[self.Bus.islanded_a] = 0.0
        self.dae.g[self.Bus.islanded_v] = 0.0

    def j_update(self, models: OrderedDict, info=None):
        """
        Call the Jacobian update method for models in sequence.

        The procedure is
        - Restore the sparsity pattern with :py:func:`andes.variables.dae.DAE.restore_sparse`
        - For each sparse matrix in (fx, fy, gx, gy), evaluate the Jacobian function calls and add values.

        Notes
        -----
        Updated Jacobians are immediately reflected in the DAE sparse matrices (fx, fy, gx, gy).
        """
        self.call_models('j_update', models)

        self.dae.restore_sparse()
        # collect sparse values into sparse structures
        for j_name in jac_names:
            j_size = self.dae.get_size(j_name)

            for mdl in models.values():
                for rows, cols, vals in mdl.triplets.zip_ijv(j_name):
                    try:
                        if self.config.ipadd:
                            self.dae.__dict__[j_name].ipadd(vals, rows, cols)
                        else:
                            self.dae.__dict__[j_name] += spmatrix(vals, rows, cols, j_size, 'd')
                    except TypeError as e:
                        logger.error("Error adding Jacobian triplets to existing sparsity pattern.")
                        logger.error(f'{mdl.class_name}: j_name {j_name}, row={rows}, col={cols}, val={vals}, '
                                     f'j_size={j_size}')
                        raise e

        self.j_islands()

        if info:
            logger.debug("Jacobian updated at t=%.6f: %s.", self.dae.t, info)
        else:
            logger.debug("Jacobian updated at t=%.6f.", self.dae.t)

    def j_islands(self):
        """
        Set gy diagonals to eps for `a` and `v` variables of islanded buses.
        """
        if self.Bus.n_islanded_buses == 0:
            return

        aidx = self.Bus.islanded_a
        vidx = self.Bus.islanded_v

        if self.config.ipadd:
            self.dae.gy.ipset(self.config.diag_eps, aidx, aidx)
            self.dae.gy.ipset(0.0, aidx, vidx)

            self.dae.gy.ipset(self.config.diag_eps, vidx, vidx)
            self.dae.gy.ipset(0.0, vidx, aidx)
        else:
            avals = [-self.dae.gy[int(idx), int(idx)] + self.config.diag_eps for idx in aidx]
            vvals = [-self.dae.gy[int(idx), int(idx)] + self.config.diag_eps for idx in vidx]

            self.dae.gy += spmatrix(avals, aidx, aidx, self.dae.gy.size, 'd')
            self.dae.gy += spmatrix(vvals, vidx, vidx, self.dae.gy.size, 'd')

    def store_sparse_pattern(self, models: OrderedDict):
        """
        Collect and store the sparsity pattern of Jacobian matrices.

        This is a runtime function specific to cases.

        Notes
        -----
        For `gy` matrix, always make sure the diagonal is reserved.
        It is a safeguard if the modeling user omitted the diagonal
        term in the equations.
        """
        self.call_models('store_sparse_pattern', models)

        # add variable jacobian values
        for jname in jac_names:
            ii, jj, vv = list(), list(), list()

            # for `gy`, reserve memory for the main diagonal
            if jname == 'gy':
                ii.extend(np.arange(self.dae.m))
                jj.extend(np.arange(self.dae.m))
                vv.extend(np.zeros(self.dae.m))

            for mdl in models.values():
                for row, col, val in mdl.triplets.zip_ijv(jname):
                    ii.extend(row)
                    jj.extend(col)
                    vv.extend(np.zeros_like(row))
                for row, col, val in mdl.triplets.zip_ijv(jname + 'c'):
                    # process constant Jacobians separately
                    ii.extend(row)
                    jj.extend(col)
                    vv.extend(val * np.ones_like(row))

            if len(ii) > 0:
                ii = np.array(ii, dtype=int)
                jj = np.array(jj, dtype=int)
                vv = np.array(vv, dtype=float)

            self.dae.store_sparse_ijv(jname, ii, jj, vv)
            self.dae.build_pattern(jname)

        # set sink diagonal to 1.0 for compacted replaced devices
        si = getattr(self, '_y_sink_idx', None)
        if si is not None:
            self.dae.tpl['gy'][si, si] = 1.0
            self.dae.gy[si, si] = 1.0

    def vars_to_dae(self, model):
        """
        Copy variables values from models to `System.dae`.

        This function clears `DAE.x` and `DAE.y` and collects values from models.
        """
        self._v_to_dae('x', model)
        self._v_to_dae('y', model)

    def vars_to_models(self):
        """
        Copy variable values from `System.dae` to models.
        """

        for var in self._getters['y']:
            if var.n > 0:
                var.v[:] = self.dae.y[var.a]

        for var in self._getters['x']:
            if var.n > 0:
                var.v[:] = self.dae.x[var.a]

    def connectivity(self, info=True):
        """
        Delegate to :meth:`ConnMan.check_connectivity`.
        """
        return self.conn.check_connectivity(info=info)

    def to_ipysheet(self, model: str, vin: bool = False):
        """
        Return an ipysheet object for editing in Jupyter Notebook.
        """

        from ipysheet import from_dataframe

        return from_dataframe(self.models[model].as_df(vin=vin))

    def from_ipysheet(self, model: str, sheet, vin: bool = False):
        """
        Set an ipysheet object back to model.
        """

        from ipysheet import to_dataframe

        df = to_dataframe(sheet)
        self.models[model].update_from_df(df, vin=vin)

    def summary(self):
        """
        Delegate to :meth:`ConnMan.summary`.
        """
        return self.conn.summary()

    def _v_to_dae(self, v_code, model):
        """
        Helper function for collecting variable values into ``dae``
        structures `x` and `y`.

        This function must be called with ``dae.x`` and ``dae.y``
        both being zeros.
        Otherwise, adders will be summed again, causing an error.

        Parameters
        ----------
        v_code : 'x' or 'y'
            Variable type name
        """
        if model.n == 0:
            return
        if model.flags.initialized is False:
            return

        for var in model.cache.v_adders.values():
            if var.v_code != v_code:
                continue
            np.add.at(self.dae.__dict__[v_code], var.a, var.v)

        for var in self._setters[v_code]:
            if var.owner.flags.initialized is False:
                continue
            if var.n > 0:
                np.put(self.dae.__dict__[v_code], var.a, var.v)

    def _e_to_dae(self, eq_name: Union[str, Tuple] = ('f', 'g')):
        """
        Helper function for collecting equation values into `System.dae.f` and `System.dae.g`.

        Parameters
        ----------
        eq_name : 'x' or 'y' or tuple
            Equation type name
        """
        if isinstance(eq_name, str):
            eq_name = [eq_name]

        for name in eq_name:
            for var in self._adders[name]:
                np.add.at(self.dae.__dict__[name], var.a, var.e)
            for var in self._setters[name]:
                np.put(self.dae.__dict__[name], var.a, var.e)

    def get_z(self, models: OrderedDict):
        """
        Get all discrete status flags in a numpy array.
        Values are written to ``dae.z`` in place.

        Returns
        -------
        numpy.array
        """
        if self.TDS.config.store_z != 1:
            return None

        if len(self.dae.z) != self.dae.o:
            self.dae.z = np.zeros(self.dae.o, dtype=float)

        ii = 0
        for mdl in models.values():
            if mdl.n == 0 or len(mdl._input_z) == 0:
                continue
            for zz in mdl._input_z.values():
                self.dae.z[ii:ii + mdl.n] = zz
                ii += mdl.n

        return self.dae.z

    def find_models(self, flag: Optional[Union[str, Tuple]], skip_zero: bool = True):
        """
        Find models with at least one of the flags as True.

        Warnings
        --------
        Checking the number of devices has been centralized into this function.
        ``models`` passed to most System calls must be retrieved from here.

        Parameters
        ----------
        flag : list, str
            Flags to find

        skip_zero : bool
            Skip models with zero devices

        Returns
        -------
        OrderedDict
            model name : model instance

        """
        if isinstance(flag, str):
            flag = [flag]

        out = OrderedDict()
        for name, mdl in self.models.items():

            if skip_zero is True:
                if (mdl.n == 0) or (mdl.in_use is False):
                    continue

            for f in flag:
                if mdl.flags.__dict__[f] is True:
                    out[name] = mdl
                    break
        return out

    def undill(self, autogen_stale=True):
        """
        Delegate to :class:`andes.system.codegen.CodegenManager`.
        """
        return self.codegen.undill(autogen_stale=autogen_stale)

    def _load_calls(self):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._load_calls() is deprecated and will be removed in v3.0. "
            "Use system.codegen._load_calls() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._load_calls()

    def _expand_pycode(self, pycode_module):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._expand_pycode() is deprecated and will be removed in v3.0. "
            "Use system.codegen._expand_pycode() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._expand_pycode(pycode_module)

    def _get_models(self, models):
        """
        Helper function for sanitizing the ``models`` input.

        The output is an OrderedDict of model names and instances.
        """
        out = OrderedDict()

        if isinstance(models, OrderedDict):
            out.update(models)

        elif models is None:
            out.update(self.exist.pflow)

        elif isinstance(models, str):
            out[models] = self.__dict__[models]

        elif isinstance(models, Model):
            out[models.class_name] = models

        elif isinstance(models, list):
            for item in models:
                if isinstance(item, Model):
                    out[item.class_name] = item
                elif isinstance(item, str):
                    out[item] = self.__dict__[item]
                else:
                    raise TypeError(f'Unknown type {type(item)}')

        return out

    def _store_tf(self, models):
        """
        Store the inverse time constant associated with equations.
        """
        for mdl in models.values():
            for var in mdl.cache.states_and_ext.values():
                if var.t_const is not None:
                    np.put(self.dae.Tf, var.a, var.t_const.v)

    def call_models(self, method: str, models: OrderedDict, *args, **kwargs):
        """
        Call methods on the given models.

        Parameters
        ----------
        method : str
            Name of the model method to be called
        models : OrderedDict, list, str
            Models on which the method will be called
        args
            Positional arguments to be passed to the model method
        kwargs
            Keyword arguments to be passed to the model method

        Returns
        -------
        The return value of the models in an OrderedDict

        """
        ret = OrderedDict()
        for name, mdl in models.items():
            ret[name] = getattr(mdl, method)(*args, **kwargs)

            if self.config.save_stats:
                if method not in self.call_stats[name]:
                    self.call_stats[name][method] = 1
                else:
                    self.call_stats[name][method] += 1

        return ret

    def check_group_common(self):
        """
        Delegate to :class:`andes.system.registry.RegistryLoader`.
        """
        return self.registry.check_group_common()

    def collect_ref(self):
        """
        Collect indices into `BackRef` for all models.
        """
        models_and_groups = list(self.models.values()) + list(self.groups.values())

        # create an empty list of lists for all `BackRef` instances
        for model in models_and_groups:
            for ref in model.services_ref.values():
                ref.v = [list() for _ in range(model.n)]

        # `model` is the model who stores `IdxParam`s to other models
        # `BackRef` is declared at other models specified by the `model` parameter
        # of `IdxParam`s.

        for model in models_and_groups:
            if model.n == 0:
                continue

            # skip: a group is not allowed to link to other groups
            if not hasattr(model, "idx_params"):
                continue

            for idxp in model.idx_params.values():
                if (idxp.model not in self.models) and (idxp.model not in self.groups):
                    continue
                dest = self.__dict__[idxp.model]

                if dest.n == 0:
                    continue

                for name in (model.class_name, model.group):
                    # `BackRef` not requested by the linked models or groups
                    if name not in dest.services_ref:
                        continue

                    for model_idx, dest_idx in zip(model.idx.v, idxp.v):
                        if dest_idx not in dest.uid:
                            continue

                        dest.set_backref(name,
                                         from_idx=model_idx,
                                         to_idx=dest_idx)

            # set model ``in_use`` flag
            if isinstance(model, Model):
                model.set_in_use()

    def _build_status_graph(self):
        """
        Build the status propagation graph from ``IdxParam`` declarations
        with ``status_parent=True``.

        Populates ``self._status_children`` and ``self._status_parent_map``
        used by :meth:`set_status` for recursive propagation.
        """
        self._status_children = {}
        self._status_parent_map = {}

        for mdl_name, mdl in self.models.items():
            if mdl.n == 0:
                continue
            if not hasattr(mdl, 'idx_params'):
                continue

            for p_name, p_instance in mdl.idx_params.items():
                if not getattr(p_instance, 'status_parent', False):
                    continue

                parent_group = p_instance.model
                if parent_group is None:
                    continue

                # Record child -> parent mapping
                self._status_parent_map[mdl_name] = (p_name, parent_group)

                # Record parent -> child BackRef names to follow
                if parent_group not in self._status_children:
                    self._status_children[parent_group] = set()

                parent_obj = self.__dict__.get(parent_group)
                if parent_obj is None:
                    continue

                child_group_name = mdl.group
                child_model_name = mdl.class_name

                if child_group_name in parent_obj.services_ref:
                    self._status_children[parent_group].add(child_group_name)
                if child_model_name in parent_obj.services_ref:
                    self._status_children[parent_group].add(child_model_name)

        # Validate: warn if model has status_parent but no ue ConstService
        for mdl_name in self._status_parent_map:
            mdl = self.models[mdl_name]
            has_ue = (hasattr(mdl, 'ue') and
                      isinstance(getattr(mdl, 'ue'), ConstService))
            if not has_ue:
                logger.debug(
                    '<%s> declares status_parent but has no ConstService '
                    'named "ue". Status propagation will update u.v only.',
                    mdl_name,
                )

    def _get_effective_status(self, mdl, uid):
        """
        Return the effective status value for a device.

        Returns ``ue.v[uid]`` if the model has a ``ConstService`` named
        ``ue``, otherwise returns ``u.v[uid]``.
        """
        if hasattr(mdl, 'ue') and isinstance(mdl.ue, ConstService):
            return mdl.ue.v[uid]
        return mdl.u.v[uid]

    def _get_parent_ue(self, mdl, uid):
        """
        Get the parent's effective status for a device.

        Returns 1.0 if the model has no status parent.
        """
        mdl_name = mdl.class_name
        if mdl_name not in self._status_parent_map:
            return 1.0

        p_name, parent_group_name = self._status_parent_map[mdl_name]
        parent_idx = mdl.__dict__[p_name].v[uid]

        if parent_idx is None:
            return 1.0

        parent_obj = self.__dict__.get(parent_group_name)
        if parent_obj is None:
            return 1.0

        # Resolve to concrete model via group
        if hasattr(parent_obj, 'idx2model'):
            parent_mdl = parent_obj.idx2model(parent_idx)
        else:
            parent_mdl = parent_obj

        parent_uid = parent_mdl.idx2uid(parent_idx)
        return self._get_effective_status(parent_mdl, parent_uid)

    def _propagate_status(self, mdl, idx, uid, ue_val):
        """
        Recursively propagate effective status to children via BackRef.
        """
        group_name = mdl.group
        if group_name not in self._status_children:
            return

        group = self.groups.get(group_name)
        if group is None:
            return

        # Get group-level uid for this device
        if idx not in group.uid:
            return
        group_uid = group.idx2uid(idx)

        for backref_name in self._status_children[group_name]:
            if backref_name not in group.services_ref:
                continue

            child_idx_list = group.services_ref[backref_name].v[group_uid]

            for child_idx in child_idx_list:
                # Resolve child idx to concrete model
                if backref_name in self.groups:
                    child_group = self.groups[backref_name]
                    try:
                        child_mdl = child_group.idx2model(child_idx)
                    except KeyError:
                        continue
                elif backref_name in self.models:
                    child_mdl = self.models[backref_name]
                else:
                    continue

                child_uid = child_mdl.idx2uid(child_idx)
                child_ue = child_mdl.u.v[child_uid] * ue_val

                # Update ue.v if the child has a ConstService named ue
                if hasattr(child_mdl, 'ue') and isinstance(child_mdl.ue, ConstService):
                    child_mdl.ue.v[child_uid] = child_ue

                # Recurse to children of this child
                self._propagate_status(child_mdl, child_idx, child_uid, child_ue)

    def set_status(self, model_or_group, idx, value):
        """
        Set the online status of a device and propagate to dependents.

        Sets ``u.v`` (the device's own status) and recomputes ``ue.v``
        (effective status) for the device and all downstream dependents
        recursively via BackRef.

        Parameters
        ----------
        model_or_group : str
            Model name or group name containing the device.
        idx : str, int, float
            Device idx.
        value : int or float
            New status value (0 or 1).
        """
        if model_or_group in self.groups:
            grp = self.groups[model_or_group]
            mdl = grp.idx2model(idx)
        elif model_or_group in self.models:
            mdl = self.models[model_or_group]
        else:
            raise KeyError(f"'{model_or_group}' is not a model or group name.")

        uid = mdl.idx2uid(idx)

        # Set u.v (device's own status)
        mdl.u.v[uid] = value

        # Recompute ue.v for this device
        parent_ue = self._get_parent_ue(mdl, uid)
        ue_val = value * parent_ue

        if hasattr(mdl, 'ue') and isinstance(mdl.ue, ConstService):
            mdl.ue.v[uid] = ue_val

        # Propagate to children
        self._propagate_status(mdl, idx, uid, ue_val)

    def get_status(self, model_or_group, idx):
        """
        Get the effective status of a device.

        Returns ``ue.v`` if the model has a ``ConstService`` named ``ue``,
        otherwise returns ``u.v``.

        Parameters
        ----------
        model_or_group : str
            Model name or group name.
        idx : str, int, float
            Device idx.

        Returns
        -------
        float
            Effective status value (0 or 1).
        """
        if model_or_group in self.groups:
            grp = self.groups[model_or_group]
            mdl = grp.idx2model(idx)
        elif model_or_group in self.models:
            mdl = self.models[model_or_group]
        else:
            raise KeyError(f"'{model_or_group}' is not a model or group name.")

        uid = mdl.idx2uid(idx)
        return self._get_effective_status(mdl, uid)

    def import_groups(self):
        """
        Delegate to :class:`andes.system.registry.RegistryLoader`.
        """
        warnings.warn(
            "System.import_groups() is deprecated and will be removed in v3.0. "
            "Use system.registry.import_groups() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.registry.import_groups()

    def import_models(self):
        """
        Delegate to :class:`andes.system.registry.RegistryLoader`.
        """
        warnings.warn(
            "System.import_models() is deprecated and will be removed in v3.0. "
            "Use system.registry.import_models() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.registry.import_models()

    def import_routines(self):
        """
        Delegate to :class:`andes.system.registry.RegistryLoader`.
        """
        warnings.warn(
            "System.import_routines() is deprecated and will be removed in v3.0. "
            "Use system.registry.import_routines() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.registry.import_routines()

    def store_switch_times(self, models, eps=1e-4):
        """
        Store event switching time in a sorted Numpy array in ``System.switch_times``
        and an OrderedDict ``System.switch_dict``.

        ``System.switch_dict`` has keys as event times and values as the OrderedDict
        of model names and instances associated with the event.

        Parameters
        ----------
        models : OrderedDict
            model name : model instance
        eps : float
            The small time step size to use immediately before
            and after the event

        Returns
        -------
        array-like
            self.switch_times
        """
        out = np.array([], dtype=float)

        if self.options.get('flat') is True:
            return out

        names = []
        for instance in models.values():
            times = np.array(instance.get_times()).ravel()
            out = np.append(out, times)
            out = np.append(out, times - eps)
            out = np.append(out, times + eps)
            names.extend([instance.class_name] * (3 * len(times)))

        # sort
        sort_idx = np.argsort(out).astype(int)
        out = out[sort_idx]
        names = [names[i] for i in sort_idx]

        # select t > current time
        ltzero_idx = np.where(out >= self.dae.t)[0]
        out = out[ltzero_idx]
        names = [names[i] for i in ltzero_idx]

        # make into an OrderedDict with unique keys and model names combined
        for i, j in zip(out, names):
            if i not in self.switch_dict:
                self.switch_dict[i] = {j: self.models[j]}
            else:
                self.switch_dict[i].update({j: self.models[j]})

        self.switch_times = np.array(list(self.switch_dict.keys()))

        # self.switch_times = out
        self.n_switches = len(self.switch_times)
        return self.switch_times

    def switch_action(self, models: OrderedDict):
        """
        Invoke the actions associated with switch times.

        This function will not be called if ``flat=True`` is passed to system.
        """
        for instance in models.values():
            instance.switch_action(self.dae.t)

        # TODO: generalize below for any models with timeseries data.
        self.TimeSeries.apply_exact(self.dae.t)

    def _p_restore(self):
        """
        Restore parameters stored in `pin`.
        """
        for model in self.models.values():
            for param in model.num_params.values():
                param.restore()

    def e_clear(self, models: OrderedDict):
        """
        Clear equation arrays in DAE and model variables.

        This step must be called before calling `f_update` or `g_update` to flush existing values.
        """
        self.dae.clear_fg()
        self.call_models('e_clear', models)

    def remove_pycapsule(self):
        """
        Remove PyCapsule objects in solvers.
        """
        for r in self.routines.values():
            r.solver.clear()

    def _store_calls(self, models: OrderedDict):
        """
        Deprecated wrapper to :class:`andes.system.codegen.CodegenManager`.
        """
        warnings.warn(
            "System._store_calls() is deprecated and will be removed in v3.0. "
            "Use system.codegen._store_calls() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.codegen._store_calls(models)

    def _list2array(self):
        """
        Helper function to call models' ``list2array`` method, which usually
        performs memory preallocation.
        """
        self.call_models('list2array', self.models)

    def _report_param_corrections(self):
        """
        Report parameter corrections for all models.

        Emits grouped warnings for parameter values that were corrected
        during device addition (e.g., non-zero, non-positive, non-negative violations).
        """
        for mdl in self.models.values():
            mdl.report_corrections()

    def set_config(self, config=None):
        """
        Delegate to :class:`andes.system.config_runtime.SystemConfigRuntime`.
        """
        return self.config_runtime.set_config(config=config)

    def collect_config(self):
        """
        Delegate to :class:`andes.system.config_runtime.SystemConfigRuntime`.
        """
        return self.config_runtime.collect_config()

    def save_config(self, file_path=None, overwrite=False):
        """
        Delegate to :class:`andes.system.config_runtime.SystemConfigRuntime`.
        """
        return self.config_runtime.save_config(file_path=file_path,
                                               overwrite=overwrite)

    def supported_models(self, export='plain'):
        """
        Return the support group names and model names in a table.

        Returns
        -------
        str
            A table-formatted string for the groups and models
        """

        def rst_ref(name, export):
            """
            Refer to the model in restructuredText mode so that
            it renders as a hyperlink.
            """

            if export == 'rest':
                return ":ref:`" + name + '`'
            else:
                return name

        pairs = list()
        for g in self.groups:
            models = list()
            for m in self.groups[g].models:
                models.append(rst_ref(m, export))
            if len(models) > 0:
                pairs.append((rst_ref(g, export), ', '.join(models)))

        tab = Tab(title='Supported Groups and Models',
                  header=['Group', 'Models'],
                  data=pairs,
                  export=export,
                  )

        return tab.draw()

    def as_dict(self, vin=False, skip_empty=True):
        """
        Return system data as a dict where the keys are model names and values
        are dicts. Each dict has parameter names as keys and corresponding data
        in an array as values.

        Returns
        -------
        OrderedDict

        """
        out = OrderedDict()

        for name, instance in self.models.items():
            if skip_empty and instance.n == 0:
                continue
            out[name] = instance.as_dict(vin=vin)

        return out

    def set_output_subidx(self, models):
        """
        Process :py:class:`andes.models.misc.Output` data and store the
        sub-indices into ``dae.xy``.

        Parameters
        ----------
        models : OrderedDict
            Models currently in use for the routine
        """

        export_vars = dict(x=list(), y=list())  # indices of export x and y

        for model, var, dev in zip(self.Output.model.v,
                                   self.Output.varname.v,
                                   self.Output.dev.v):

            # check validity of model name
            if model not in models:
                logger.info("Output model <%s> invalid or contains no device. Skipped.",
                            model)
                continue
            mdl_instance = models[model]
            mdl_all_vars = mdl_instance.cache.all_vars

            # check validity of var name
            if var is not None and (var not in mdl_all_vars):
                logger.info("Output model <%s> contains no variable <%s>. Skipped.",
                            model, var)
                continue

            # check validity of dev idx
            if (dev is not None) and (dev not in mdl_instance.idx.v):
                logger.info("Output model <%s> contains no device <%s>. Skipped.",
                            model, dev)
                continue

            # TODO: dev-based indexing is not fully supported
            # for multi-index variables, such as those in COI.

            if var is None:
                for item in mdl_all_vars.values():
                    if dev is None:
                        export_vars[item.v_code].extend(item.a)
                    else:
                        uid = mdl_instance.idx2uid(dev)
                        export_vars[item.v_code].append(item.a[uid])
            else:  # with variable name
                item = mdl_all_vars[var]
                if dev is None:
                    export_vars[item.v_code].extend(item.a)
                else:  # with exact index
                    uid = mdl_instance.idx2uid(dev)
                    export_vars[item.v_code].append(item.a[uid])

        self.Output.xidx = sorted(np.unique(export_vars['x']))
        self.Output.yidx = sorted(np.unique(export_vars['y']))

    # --------------- DAE Compaction ---------------

    @staticmethod
    def _break_var_contiguity(var):
        """Backward-compatible wrapper to :class:`andes.system.dae_compactor.DAECompactor`."""
        DAECompactor._break_var_contiguity(var)

    def compact_dae(self):
        """
        Delegate to :class:`andes.system.dae_compactor.DAECompactor`.
        """
        return self.dae_compactor.compact_dae()

    def _detect_replaced_devices(self):
        """
        Delegate to :class:`andes.system.dae_compactor.DAECompactor`.
        """
        return self.dae_compactor._detect_replaced_devices()

    def _build_y_compaction_map(self, excluded):
        """
        Delegate to :class:`andes.system.dae_compactor.DAECompactor`.
        """
        return self.dae_compactor._build_y_compaction_map(excluded)

    def _compact_dae_y(self, old_to_new, new_m, need_sink,
                       all_replaced_models):
        """
        Delegate to :class:`andes.system.dae_compactor.DAECompactor`.
        """
        return self.dae_compactor._compact_dae_y(old_to_new, new_m, need_sink,
                                                 all_replaced_models)


# --------------- Helper Functions ---------------

def _config_numpy(seed='None', divide='warn', invalid='warn'):
    """
    Backward-compatible wrapper to
    :meth:`andes.system.config_runtime.SystemConfigRuntime.configure_numpy`.
    """
    warnings.warn(
        "andes.system.facade._config_numpy() is deprecated and will be removed in v3.0. "
        "Use andes.system.config_runtime.SystemConfigRuntime.configure_numpy() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return SystemConfigRuntime.configure_numpy(seed=seed,
                                               divide=divide,
                                               invalid=invalid)


def load_config_rc(conf_path=None):
    """
    Backward-compatible wrapper to
    :meth:`andes.system.config_runtime.SystemConfigRuntime.load_config_rc`.
    """
    warnings.warn(
        "andes.system.facade.load_config_rc() is deprecated and will be removed in v3.0. "
        "Use andes.system.config_runtime.SystemConfigRuntime.load_config_rc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return SystemConfigRuntime.load_config_rc(conf_path=conf_path)


def fix_view_arrays(system, models=None):
    """
    Point NumPy arrays without OWNDATA (termed "view arrays" here) to the source
    array.

    This function properly sets ``v`` and ``e`` arrays of internal variables as
    views of the corresponding DAE arrays.

    Inputs will be refreshed for each model.

    Parameters
    ----------
    system : andes.system.System
        System object to be fixed
    models : OrderedDict, optional
        Subset of models to fix.  Defaults to ``system.models`` (all models).
    """

    if models is None:
        models = system.models

    system.set_var_arrays(models)

    for model in models.values():
        if model.n > 0:
            model.get_inputs(refresh=True)

    return True


def import_pycode(user_pycode_path=None):
    """
    Backward-compatible wrapper to :func:`andes.system.codegen.import_pycode`.
    """
    from andes.system.codegen import import_pycode as _import_pycode
    return _import_pycode(user_pycode_path=user_pycode_path)


def _import_pycode_from(pycode_path):
    """
    Backward-compatible wrapper to :func:`andes.system.codegen._import_pycode_from`.
    """
    from andes.system.codegen import _import_pycode_from as _import_pycode_from_impl
    return _import_pycode_from_impl(pycode_path)


def reload_submodules(module_name):
    """
    Backward-compatible wrapper to :func:`andes.system.codegen.reload_submodules`.
    """
    from andes.system.codegen import reload_submodules as _reload_submodules
    return _reload_submodules(module_name)


def _append_model_name(model_name, idx):
    """
    Helper function for appending ``idx`` to model names.
    Removes duplicate model name strings.
    """

    out = ''
    if isinstance(idx, str) and (model_name in idx):
        out = idx
    else:
        out = f'{model_name} {idx}'

    # replaces `_` with space for LaTeX to continue
    out = out.replace('_', ' ')
    return out


def _set_xy_name(mdl, vars_dict, dests):
    """
    Helper function for setting algebraic and state variable names.
    """

    mdl_name = mdl.class_name
    idx = mdl.idx
    for name, item in vars_dict.items():
        for idx_item, addr in zip(idx.v, item.a):
            dests[0][addr] = f'{name} {_append_model_name(mdl_name, idx_item)}'
            dests[1][addr] = rf'${item.tex_name}$ {_append_model_name(mdl_name, idx_item)}'


def _set_hi_name(mdl, vars_dict, dests):
    """
    Helper function for setting names of external equations.
    """

    mdl_name = mdl.class_name
    idx = mdl.idx
    for item in vars_dict.values():
        if len(item.r) != len(idx.v):
            idxall = item.indexer.v
        else:
            idxall = idx.v

        for idx_item, addr in zip(idxall, item.r):
            dests[0][addr] = f'{item.ename} {_append_model_name(mdl_name, idx_item)}'
            dests[1][addr] = rf'${item.tex_ename}$ {_append_model_name(mdl_name, idx_item)}'


def _set_z_name(mdl, dae, dests):
    """
    Helper function for addng and setting discrete flag names.
    """

    for item in mdl.discrete.values():
        if mdl.flags.initialized:
            continue
        mdl_name = mdl.class_name

        for name, tex_name in zip(item.get_names(), item.get_tex_names()):
            for idx_item in mdl.idx.v:
                dests[0].append(f'{name} {_append_model_name(mdl_name, idx_item)}')
                dests[1].append(rf'${item.tex_name}$ {_append_model_name(mdl_name, idx_item)}')
                dae.o += 1


def example(setup=True, no_output=True, **kwargs):
    """
    Return an :py:class:`andes.system.System` object for the
    ``ieee14_linetrip.xlsx`` as an example.

    This function is useful when a user wants to quickly get a
    System object for testing.

    Returns
    -------
    System
        An example :py:class:`andes.system.System` object.
    """

    return andes.load(andes.get_case("ieee14/ieee14_linetrip.xlsx"),
                      setup=setup, no_output=no_output, **kwargs)
