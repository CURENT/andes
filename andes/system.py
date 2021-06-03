"""
System class for power system data and methods
"""

#  [ANDES] (C)2015-2021 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  File name: system.py
#  Last modified: 8/16/20, 7:26 PM

import configparser
import importlib
import logging
import os
import sys
import inspect
import dill
from collections import OrderedDict
from typing import Dict, Tuple, Union, Optional

import andes.io
from andes import __version__
from andes.models import file_classes
from andes.models.group import GroupBase
from andes.variables import FileMan, DAE
from andes.routines import all_routines
from andes.utils.tab import Tab
from andes.utils.misc import elapsed
from andes.utils.paths import confirm_overwrite
from andes.utils.paths import get_config_path, get_dot_andes_path
from andes.utils.paths import get_pycode_path, get_pkl_path
from andes.core import Config, Model, AntiWindup
from andes.io.streaming import Streaming

from andes.shared import np, jac_names, dilled_vars, IP_ADD
from andes.shared import matrix, spmatrix, sparse

logger = logging.getLogger(__name__)
sys.path.append(get_dot_andes_path())
pycode = None


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
                 **kwargs
                 ):
        self.name = name
        self.options = {}
        if options is not None:
            self.options.update(options)
        if kwargs:
            self.options.update(kwargs)
        self.calls = OrderedDict()         # a dictionary with model names (keys) and their ``calls`` instance
        self.models = OrderedDict()        # model names and instances
        self.groups = OrderedDict()        # group names and instances
        self.routines = OrderedDict()      # routine names and instances
        self.switch_times = np.array([])   # an array of ordered event switching times
        self.switch_dict = OrderedDict()   # time: OrderedDict of associated models
        self.n_switches = 0                # number of elements in `self.switch_times`
        self.exit_code = 0                 # command-line exit code, 0 - normal, others - error.

        # get and load default config file
        self._config_path = get_config_path()
        if config_path is not None:
            self._config_path = config_path
        if default_config is True:
            self._config_path = None

        self._config_object = self.load_config(self._config_path)
        self.config = Config(self.__class__.__name__, dct=config)
        self.config.load(self._config_object)

        # custom configuration for system goes after this line
        self.config.add(OrderedDict((('freq', 60),
                                     ('mva', 100),
                                     ('store_z', 0),
                                     ('ipadd', 1),
                                     ('seed', 'None'),
                                     ('diag_eps', 1e-8),
                                     ('warn_limits', 1),
                                     ('warn_abnormal', 1),
                                     ('dime_enabled', 0),
                                     ('dime_name', 'andes'),
                                     ('dime_address', 'ipc:///tmp/dime2'),
                                     ('numba', 0),
                                     ('numba_parallel', 0),
                                     ('yapf_pycode', 0),
                                     ('np_divide', 'warn'),
                                     ('np_invalid', 'warn'),
                                     ('pickle_path', get_pkl_path())
                                     )))
        self.config.add_extra("_help",
                              freq='base frequency [Hz]',
                              mva='system base MVA',
                              store_z='store limiter status in TDS output',
                              ipadd='use spmatrix.ipadd if available',
                              seed='seed (or None) for random number generator',
                              diag_eps='small value for Jacobian diagonals',
                              warn_limits='warn variables initialized at limits',
                              warn_abnormal='warn initialization out of normal values',
                              numba='use numba for JIT compilation',
                              numba_parallel='enable parallel for numba.jit',
                              yapf_pycode='format generated code with yapf',
                              np_divide='treatment for division by zero',
                              np_invalid='treatment for invalid floating-point ops.',
                              pickle_path='path models should be (un)dilled to/from',
                              )
        self.config.add_extra("_alt",
                              freq="float",
                              mva="float",
                              store_z=(0, 1),
                              ipadd=(0, 1),
                              seed='int or None',
                              warn_limits=(0, 1),
                              warn_abnormal=(0, 1),
                              numba=(0, 1),
                              numba_parallel=(0, 1),
                              yapf_pycode=(0, 1),
                              np_divide={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                              np_invalid={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                              )
        self.config.check()
        self._set_numpy()

        self.exist = ExistingModels()

        self.files = FileMan(case=case, **self.options)    # file path manager
        self.dae = DAE(system=self)                        # numerical DAE storage
        self.streaming = Streaming(self)                   # Dime2 streaming

        # dynamic imports of groups, models and routines
        self.import_groups()
        self.import_models()
        self.import_routines()  # routine imports come after models

        self._getters = dict(f=list(), g=list(), x=list(), y=list())
        self._adders = dict(f=list(), g=list(), x=list(), y=list())
        self._setters = dict(f=list(), g=list(), x=list(), y=list())
        self.antiwindups = list()
        self.no_check_init = list()  # states for which initialization check is omitted

        # internal flags
        self.is_setup = False        # if system has been setup

    def _set_numpy(self):
        """
        Configure NumPy based on Config.
        """
        # set up numpy random seed
        if isinstance(self.config.seed, int):
            np.random.seed(self.config.seed)
            logger.debug("Random seed set to <%d>.", self.config.seed)

        np.seterr(divide=self.config.np_divide,
                  invalid=self.config.np_invalid,
                  )

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

    def prepare(self, quick=False, incremental=False):
        """
        Generate numerical functions from symbolically defined models.

        All procedures in this function must be independent of test case.

        Parameters
        ----------
        quick : bool, optional
            True to skip pretty-print generation to reduce code generation time.
        incremental : bool, optional
            True to generate only for modified models, incrementally.

        Notes
        -----
        Option ``incremental`` compares the md5 checksum of all var and
        service strings, and only regenerate for updated models.

        Examples
        --------
        If one needs to print out LaTeX-formatted equations in a Jupyter Notebook, one need to generate such
        equations with ::

            import andes
            sys = andes.prepare()

        Alternatively, one can explicitly create a System and generate the code ::

            import andes
            sys = andes.System()
            sys.prepare()

        Warnings
        --------
        Generated lambda functions will be serialized to file, but pretty prints (SymPy objects) can only exist in
        the System instance on which prepare is called.
        """
        import math

        # info
        if incremental is True:
            mode_text = 'rapid incremental mode'
        elif quick is True:
            mode_text = 'quick mode'
        else:
            mode_text = 'full mode'

        logger.info('Numerical code generation (%s) started...', mode_text)

        t0, _ = elapsed()

        # consistency check for group parameters and variables
        self._check_group_common()

        # get `pycode` folder path without automatic creation
        pycode_path = get_pycode_path(self.options.get("pycode_path"), mkdir=False)

        loaded_calls = self._load_pkl()
        if loaded_calls is None and incremental:
            incremental = False
            logger.debug('calls.pkl does not exist. Incremental codegen disabled.')

        total = len(self.models)
        width = math.ceil(math.log(total, 10))
        for idx, (name, model) in enumerate(self.models.items()):
            if incremental and \
                    name in loaded_calls and \
                    hasattr(loaded_calls[name], 'md5'):
                if loaded_calls[name].md5 == model.get_md5():
                    model.calls = loaded_calls[name]
                    print(f"\r\x1b[K Code generation skipped for {name} ({idx + 1:>{width}}/{total:>{width}}).",
                          end='\r', flush=True)
                    continue

            print(f"\r\x1b[K Generating code for {name} ({idx+1:>{width}}/{total:>{width}}).",
                  end='\r', flush=True)

            model.prepare(quick=quick, pycode_path=pycode_path)

        # -- store pycode --
        logger.info('Storing generated pycode to "%s"', pycode_path)

        # write `__init__.py` that imports all to the `pycode` package
        init_path = os.path.join(pycode_path, '__init__.py')
        with open(init_path, 'w') as f:
            f.write(f"__version__ = '{__version__}'\n\n")
            for name in self.models.keys():
                f.write(f"from . import {name}  # NOQA\n")
            f.write('\n')

        # RELOAD REQUIRED as the generated Jacobian arguments may be in a different order
        if pycode is not None:
            importlib.reload(pycode)

        self._store_calls()
        self.dill()

        _, s = elapsed(t0)
        logger.info('Successfully generated numerical code in %s.', s)

    def _prepare_mp(self, quick=False):
        """
        Code generation with multiprocessing. NOT WORKING NOW.

        Warnings
        --------
        Function is not working. Serialization failed for `conj`.
        """
        from andes.shared import Pool
        dill.settings['recurse'] = True

        # consistency check for group parameters and variables
        self._check_group_common()

        def _prep_model(model: Model):
            model.prepare(quick=quick)
            return model

        model_list = list(self.models.values())

        # TODO: failed when serializing.
        ret = Pool().map(_prep_model, model_list)

        for idx, name in enumerate(self.models.keys()):
            self.models[name] = ret[idx]

        self._store_calls()
        self.dill()

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
        self._list2array()     # `list2array` must come before `link_ext_param`
        if not self.link_ext_param():
            ret = False

        self.find_devices()    # find or add required devices

        # === no device addition or removal after this point ===
        self.calc_pu_coeff()   # calculate parameters in system per units
        self.store_existing()  # store models with routine flags

        # assign address at the end before adding devices and processing parameters
        self.set_address(self.exist.pflow)
        self.set_dae_names(self.exist.pflow)
        self.store_sparse_pattern(self.exist.pflow)
        self.store_adder_setter(self.exist.pflow)

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
        if model not in self.models:
            logger.warning(f"<{model}> is not an existing model.")
            return
        if self.is_setup:
            raise NotImplementedError("Adding devices are not allowed after setup.")
        group_name = self.__dict__[model].group
        group = self.groups[group_name]

        if param_dict is None:
            param_dict = {}
        if kwargs is not None:
            param_dict.update(kwargs)

        idx = param_dict.pop('idx', None)
        if idx is np.nan:
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
        for mdl in models.values():
            if mdl.flags.address is True:
                logger.debug('%s internal address exists', mdl.class_name)
                continue
            if mdl.n == 0:
                continue

            # set internal variable addresses
            logger.debug('Setting internal address for %s', mdl.class_name)
            n = mdl.n
            m0 = self.dae.m
            n0 = self.dae.n
            m_end = m0 + len(mdl.algebs) * n
            n_end = n0 + len(mdl.states) * n
            collate = mdl.flags.collate

            if not collate:
                for idx, item in enumerate(mdl.algebs.values()):
                    item.set_address(np.arange(m0 + idx * n, m0 + (idx + 1) * n), contiguous=True)
                for idx, item in enumerate(mdl.states.values()):
                    item.set_address(np.arange(n0 + idx * n, n0 + (idx + 1) * n), contiguous=True)
            else:
                for idx, item in enumerate(mdl.algebs.values()):
                    item.set_address(np.arange(m0 + idx, m_end, len(mdl.algebs)), contiguous=False)
                for idx, item in enumerate(mdl.states.values()):
                    item.set_address(np.arange(n0 + idx, n_end, len(mdl.states)), contiguous=False)

            self.dae.m = m_end
            self.dae.n = n_end

        # set external variable addresses
        for mdl in models.values():
            # handle external groups
            for name, instance in mdl.cache.vars_ext.items():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError(f'<{ext_name}> is not a model or group name.')

                try:
                    instance.link_external(ext_model)
                except (IndexError, KeyError) as e:
                    logger.error('Error: <%s> cannot retrieve <%s> from <%s> using <%s>:\n  %s',
                                 mdl.class_name, instance.name, instance.model,
                                 instance.indexer.name, repr(e))

        # set external variable RHS addresses
        for mdl in models.values():
            if mdl.flags.address is True:
                logger.debug('%s RHS address exists', mdl.class_name)
                continue
            if mdl.n == 0:
                continue

            for item in mdl.states_ext.values():
                item.set_address(np.arange(self.dae.p, self.dae.p + item.n))
                self.dae.p = self.dae.p + item.n
            for item in mdl.algebs_ext.values():
                item.set_address(np.arange(self.dae.q, self.dae.q + item.n))
                self.dae.q = self.dae.q + item.n

            mdl.flags.address = True

        # allocate memory for DAE arrays
        self.dae.resize_arrays()

        # set `v` and `e` in variables
        self.set_var_arrays(models=models)

        # pre-allocate for names
        if len(self.dae.y_name) == 0:
            self.dae.x_name = [''] * self.dae.n
            self.dae.y_name = [''] * self.dae.m
            self.dae.h_name = [''] * self.dae.p
            self.dae.i_name = [''] * self.dae.q
            self.dae.x_tex_name = [''] * self.dae.n
            self.dae.y_tex_name = [''] * self.dae.m
            self.dae.h_tex_name = [''] * self.dae.p
            self.dae.i_tex_name = [''] * self.dae.q
        else:
            self.dae.x_name.extend([''] * (self.dae.n - len(self.dae.x_name)))
            self.dae.y_name.extend([''] * (self.dae.m - len(self.dae.y_name)))
            self.dae.h_name.extend([''] * (self.dae.p - len(self.dae.h_name)))
            self.dae.i_name.extend([''] * (self.dae.q - len(self.dae.i_name)))
            self.dae.x_tex_name.extend([''] * (self.dae.n - len(self.dae.x_tex_name)))
            self.dae.y_tex_name.extend([''] * (self.dae.m - len(self.dae.y_tex_name)))
            self.dae.h_tex_name.extend([''] * (self.dae.p - len(self.dae.h_tex_name)))
            self.dae.i_tex_name.extend([''] * (self.dae.q - len(self.dae.i_tex_name)))

    def set_dae_names(self, models):
        """
        Set variable names for differential and algebraic variables, and discrete flags.
        """
        def append_model_name(model_name, idx):
            out = ''
            if isinstance(idx, str) and (model_name in idx):
                out = idx
            else:
                out = f'{model_name} {idx}'

            # replaces `_` with space for LaTeX to continue
            out = out.replace('_', ' ')
            return out

        for mdl in models.values():
            mdl_name = mdl.class_name
            idx = mdl.idx
            for name, item in mdl.algebs.items():
                for id, addr in zip(idx.v, item.a):
                    self.dae.y_name[addr] = f'{name} {append_model_name(mdl_name, id)}'
                    self.dae.y_tex_name[addr] = rf'${item.tex_name}$ {append_model_name(mdl_name, id)}'
            for name, item in mdl.states.items():
                for id, addr in zip(idx.v, item.a):
                    self.dae.x_name[addr] = f'{name} {append_model_name(mdl_name, id)}'
                    self.dae.x_tex_name[addr] = rf'${item.tex_name}$ {append_model_name(mdl_name, id)}'

            # add discrete flag names
            if self.config.store_z == 1:
                for item in mdl.discrete.values():
                    if mdl.flags.initialized:
                        continue
                    for name, tex_name in zip(item.get_names(), item.get_tex_names()):
                        for id in idx.v:
                            self.dae.z_name.append(f'{name} {append_model_name(mdl_name, id)}')
                            self.dae.z_tex_name.append(rf'${item.tex_name}$ {append_model_name(mdl_name, id)}')
                            self.dae.o += 1

    def set_var_arrays(self, models, inplace=True, alloc=True):
        """
        Set arrays (`v` and `e`) in internal variables.

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

    def _init_numba(self, models: OrderedDict):
        """
        Helper function to compile all functions with Numba before init.
        """
        if self.config.numba:
            use_parallel = True if (self.config.numba_parallel == 1) else False
            use_cache = True if (pycode is not None) else False

            logger.info("Numba compilation initiated, parallel=%s, cache=%s.",
                        use_parallel, use_cache)
            for mdl in models.values():
                mdl.numba_jitify(parallel=use_parallel, cache=use_cache)

    def init(self, models: OrderedDict, routine: str):
        """
        Initialize the variables for each of the specified models.

        For each model, the initialization procedure is:

        - Get values for all `ExtService`.
        - Call the model `init()` method, which initializes internal variables.
        - Copy variables to DAE and then back to the model.
        """
        try:
            self._init_numba(models)
        except ImportError:
            logger.warning("Numba not found. JIT compilation is skipped.")

        for mdl in models.values():
            # link externals first
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

        # store the inverse of time constants
        self._store_tf(models)

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

        This function calculates the per unit conversion factors, stores input parameters to `vin`, and perform
        the conversion.
        """
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

    def l_update_var(self, models: OrderedDict, niter=None, err=None):
        """
        Update variable-based limiter discrete states by calling ``l_update_var`` of models.

        This function is must be called before any equation evaluation.
        """
        self.call_models('l_update_var', models,
                         dae_t=self.dae.t, niter=niter, err=err)

    def l_update_eq(self, models:  OrderedDict):
        """
        Update equation-dependent limiter discrete components by calling ``l_check_eq`` of models.
        Force set equations after evaluating equations.

        This function is must be called after differential equation updates.
        """
        self.call_models('l_check_eq', models)

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
                        if self.config.ipadd and IP_ADD:
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

        if self.config.ipadd and IP_ADD:
            self.dae.gy.ipset(self.config.diag_eps, aidx, aidx)
            self.dae.gy.ipset(self.config.diag_eps, vidx, vidx)
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
        Perform connectivity check for system.

        Parameters
        ----------
        info : bool
            True to log connectivity summary.
        """
        logger.debug("Entering connectivity check.")

        self.Bus.n_islanded_buses = 0
        self.Bus.islanded_buses = list()
        self.Bus.island_sets = list()
        self.Bus.nosw_island = list()
        self.Bus.msw_island = list()
        self.Bus.islands = list()

        n = self.Bus.n

        # collect from-bus and to-bus indices
        fr, to, u = list(), list(), list()

        # collect from Line
        fr.extend(self.Line.a1.a.tolist())
        to.extend(self.Line.a2.a.tolist())
        u.extend(self.Line.u.v.tolist())
        os = [0] * len(u)

        # find islanded buses
        diag = list(matrix(spmatrix(u, to, os, (n, 1), 'd') +
                           spmatrix(u, fr, os, (n, 1), 'd')))

        nib = self.Bus.n_islanded_buses = diag.count(0)
        for idx in range(n):
            if diag[idx] == 0:
                self.Bus.islanded_buses.append(idx)

        # store `a` and `v` indices for zeroing out residuals
        self.Bus.islanded_a = np.array(self.Bus.islanded_buses)
        self.Bus.islanded_v = self.Bus.n + self.Bus.islanded_a

        # find islanded areas - Goderya's algorithm
        temp = spmatrix(list(u) * 4,
                        fr + to + fr + to,
                        to + fr + fr + to,
                        (n, n),
                        'd')

        cons = temp[0, :]
        nelm = len(cons.J)
        conn = spmatrix([], [], [], (1, n), 'd')
        enum = idx = islands = 0

        while True:
            while True:
                cons = cons * temp
                cons = sparse(cons)  # remove zero values
                new_nelm = len(cons.J)
                if new_nelm == nelm:
                    break
                nelm = new_nelm

            # started with an islanded bus
            if len(conn.J) == 0:
                enum += 1
            # all buses are interconnected
            elif len(cons.J) == n:
                break

            self.Bus.island_sets.append(list(cons.J))
            conn += cons
            islands += 1
            nconn = len(conn.J)
            if nconn >= (n - nib):
                self.Bus.island_sets = [i for i in self.Bus.island_sets if len(i) > 0]
                break

            for element in conn.J[idx:]:
                if not diag[idx]:
                    enum += 1  # skip islanded buses
                if element <= enum:
                    idx += 1
                    enum += 1
                else:
                    break

            cons = temp[enum, :]

        # --- check if all areas have a slack generator ---
        if len(self.Bus.island_sets) > 0:
            for idx, island in enumerate(self.Bus.island_sets):
                nosw = 1
                slack_bus_uid = self.Bus.idx2uid(self.Slack.bus.v)
                slack_u = self.Slack.u.v
                for u, item in zip(slack_u, slack_bus_uid):
                    if (u == 1) and (item in island):
                        nosw -= 1
                if nosw == 1:
                    self.Bus.nosw_island.append(idx)
                elif nosw < 0:
                    self.Bus.msw_island.append(idx)

        # --- Post processing ---
        # extend islanded buses, each in a list
        if len(self.Bus.islanded_buses) > 0:
            self.Bus.islands.extend([[item] for item in self.Bus.islanded_buses])

        if len(self.Bus.island_sets) == 0:
            self.Bus.islands.append(list(range(n)))
        else:
            self.Bus.islands.extend(self.Bus.island_sets)

        if info is True:
            self.summary()

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
        Print out system summary.
        """

        island_sets = self.Bus.island_sets
        nosw_island = self.Bus.nosw_island
        msw_island = self.Bus.msw_island
        n_islanded_buses = self.Bus.n_islanded_buses

        logger.info("-> System connectivity check results:")
        if n_islanded_buses == 0:
            logger.info("  No islanded bus detected.")
        else:
            logger.info("  %d islanded bus detected.", n_islanded_buses)
            logger.debug("  Islanded Bus indices (0-based): %s", self.Bus.islanded_buses)

        if len(island_sets) == 0:
            logger.info("  No island detected.")
        else:
            logger.info("  A total of %d island(s) detected.", len(island_sets))
            logger.debug("  Bus indices in islanded areas (0-based): %s", island_sets)

        if len(nosw_island) > 0:
            logger.warning('  Slack generator is not defined/enabled for %d island(s).',
                           len(nosw_island))
            logger.debug("  Bus indices in no-Slack areas (0-based): %s",
                         [island_sets[item] for item in nosw_island])

        if len(msw_island) > 0:
            logger.warning('  Multiple slack generators are defined/enabled for %d island(s).',
                           len(msw_island))
            logger.debug("  Bus indices in multiple-Slack areas (0-based): %s",
                         [island_sets[item] for item in msw_island])

        if len(self.Bus.nosw_island) == 0 and len(self.Bus.msw_island) == 0:
            logger.info('  Each island has a slack bus correctly defined and enabled.')

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
        if self.config.store_z != 1:
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

    def get_ext_fg(self, model: OrderedDict):
        """
        Get the right-hand side of the external equations.
        """
        pass

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

    def dill(self):
        """
        Serialize generated numerical functions in ``System.calls`` with package ``dill``.

        The serialized file will be stored to ``~/.andes/calls.pkl``, where `~` is the home directory path.

        Notes
        -----
        This function sets `dill.settings['recurse'] = True` to serialize the function calls recursively.

        """
        np_ver = np.__version__.split('.')
        np_ver = tuple([int(i) for i in np_ver])
        if np_ver < (1, 20):
            logger.debug("Dumping calls to calls.pkl with dill")
            dill.settings['recurse'] = True

            with open(self.config.pickle_path, 'wb') as f:
                dill.dump(self.calls, f)
        else:
            logger.warning("Dumping calls to calls.pkl is not supported with NumPy 1.2+")
            logger.warning("ANDES is still fully functional with generated Python code.")

    def undill(self):
        """
        Deserialize the function calls from ``~/.andes/calls.pkl`` with ``dill``.

        If no change is made to models, future calls to ``prepare()`` can be replaced with ``undill()`` for
        acceleration.
        """

        # try to replace equations and jacobian calls with saved code
        calls_loaded = False

        try:
            self._call_from_pycode()
            calls_loaded = True
        except ImportError:
            logger.debug("Pycode not found. Trying to load from `calls.pkl`.")

        if calls_loaded is False:
            self._call_from_pkl()

    def _load_pkl(self):
        """
        Helper function to open and load dill-pickled functions.
        """
        dill.settings['recurse'] = True

        if os.path.isfile(self.config.pickle_path):
            with open(self.config.pickle_path, 'rb') as f:

                try:
                    loaded_calls = dill.load(f)
                    return loaded_calls
                except (IOError, EOFError, AttributeError):
                    pass

        return None

    def _call_from_pkl(self):
        """
        Helper function for loading ModelCall from pickle file.
        """

        loaded_calls = self._load_pkl()

        if loaded_calls is not None:
            ver = loaded_calls.get('__version__')
            if ver == __version__:
                self.calls = loaded_calls
                logger.info('Undilled calls from "%s" is up-to-date.', self.config.pickle_path)
            else:
                logger.info('Undilled calls are for version %s, regenerating...', ver)
                self.prepare(quick=True, incremental=True)

        else:
            logger.info('Generating numerical calls at the first launch.')
            self.prepare(quick=True)

        for name, model_call in self.calls.items():
            if name in self.__dict__:
                self.__dict__[name].calls = model_call

        logger.debug("Using undilled lambda functions.")

    def _call_from_pycode(self):
        """
        Helper function to load generated pycode
        """
        pycode = None
        try:
            import pycode  # NOQA
        except ImportError:
            globals()['pycode'] = pycode

        try:
            from andes import pycode  # NOQA
        except ImportError:
            globals()['pycode'] = pycode

        if pycode is None:
            raise ImportError("Generated pycode not found.")

        for name in self.models:
            if name not in pycode.__dict__:
                raise ImportError("Generated pycode is incomplete.")

        for model in self.models.values():
            pycode_model = pycode.__dict__[model.class_name]

            # reload stored variables
            for item in dilled_vars:
                model.calls.__dict__[item] = pycode_model.__dict__[item]

            # equations
            model.calls.f = pycode_model.__dict__.get("f_update")
            model.calls.g = pycode_model.__dict__.get("g_update")

            # services
            for instance in model.services.values():
                if instance.v_str is not None:
                    sv_name = f'{instance.name}_svc'
                    model.calls.s[instance.name] = pycode_model.__dict__[sv_name]

            # load initialization; assignment
            for instance in model.cache.all_vars.values():
                if instance.v_str is not None:
                    ia_name = f'{instance.name}_ia'
                    model.calls.ia[instance.name] = pycode_model.__dict__[ia_name]

            # load initialization: iterative
            for item in model.calls.init_seq:
                if isinstance(item, list):
                    name_concat = '_'.join(item)
                    model.calls.ii[name_concat] = pycode_model.__dict__[name_concat + '_ii']
                    model.calls.ij[name_concat] = pycode_model.__dict__[name_concat + '_ij']

            # load Jacobian functions
            for jname in model.calls.j_names:
                model.calls.j[jname] = pycode_model.__dict__.get(f'{jname}_update')

        logger.info("Using generated Python code.")

    def _get_models(self, models):
        """
        Helper function for sanitizing the ``models`` input.

        The output is an OrderedDict of model names and instances.
        """
        if models is None:
            models = self.exist.pflow
        if isinstance(models, str):
            models = {models: self.__dict__[models]}
        elif isinstance(models, Model):
            models = {models.class_name: models}
        elif isinstance(models, list):
            models = OrderedDict()
            for item in models:
                if isinstance(item, Model):
                    models[item.class_name] = item
                elif isinstance(item, str):
                    models[item] = self.__dict__[item]
                else:
                    raise TypeError(f'Unknown type {type(item)}')
        # do nothing for OrderedDict type
        return models

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

        return ret

    def _check_group_common(self):
        """
        Check if all group common variables and parameters are met.

        This function is called at the end of code generation by `prepare`.

        Raises
        ------
        KeyError if any parameter or value is not provided
        """
        for group in self.groups.values():
            for item in group.common_params:
                for model in group.models.values():
                    # the below includes all of BaseParam (NumParam, DataParam and ExtParam)
                    if item not in model.__dict__:
                        raise KeyError(f'Group <{group.class_name}> common param <{item}> does not exist '
                                       f'in model <{model.class_name}>')
            for item in group.common_vars:
                for model in group.models.values():
                    if item not in model.cache.all_vars:
                        raise KeyError(f'Group <{group.class_name}> common param <{item}> does not exist '
                                       f'in model <{model.class_name}>')

    def collect_ref(self):
        """
        Collect indices into `BackRef` for all models.
        """
        models_and_groups = list(self.models.values()) + list(self.groups.values())

        for model in models_and_groups:
            for ref in model.services_ref.values():
                ref.v = [list() for _ in range(model.n)]

        for model in models_and_groups:
            if model.n == 0:
                continue
            if not hasattr(model, "idx_params"):
                # skip: group does not link to another group
                continue

            for ref in model.idx_params.values():
                if ref.model not in self.models and (ref.model not in self.groups):
                    continue
                dest_model = self.__dict__[ref.model]

                if dest_model.n == 0:
                    continue

                for n in (model.class_name, model.group):
                    if n not in dest_model.services_ref:
                        continue

                    for model_idx, dest_idx in zip(model.idx.v, ref.v):
                        if dest_idx not in dest_model.uid:
                            continue
                        uid = dest_model.idx2uid(dest_idx)
                        dest_model.services_ref[n].v[uid].append(model_idx)

            # set model ``in_use`` flag
            if isinstance(model, Model):
                model.set_in_use()

    def import_groups(self):
        """
        Import all groups classes defined in ``devices/group.py``.

        Groups will be stored as instances with the name as class names.
        All groups will be stored to dictionary ``System.groups``.
        """
        module = importlib.import_module('andes.models.group')

        for m in inspect.getmembers(module, inspect.isclass):

            name, cls = m
            if name == 'GroupBase':
                continue
            elif not issubclass(cls, GroupBase):
                # skip other imported classes such as `OrderedDict`
                continue

            self.__dict__[name] = cls()
            self.groups[name] = self.__dict__[name]

    def import_models(self):
        """
        Import and instantiate models as System member attributes.

        Models defined in ``models/__init__.py`` will be instantiated `sequentially` as attributes with the same
        name as the class name.
        In addition, all models will be stored in dictionary ``System.models`` with model names as
        keys and the corresponding instances as values.

        Examples
        --------
        ``system.Bus`` stores the `Bus` object, and ``system.GENCLS`` stores the classical
        generator object,

        ``system.models['Bus']`` points the same instance as ``system.Bus``.
        """
        for file, cls_list in file_classes.items():
            for model_name in cls_list:
                the_module = importlib.import_module('andes.models.' + file)
                the_class = getattr(the_module, model_name)
                self.__dict__[model_name] = the_class(system=self, config=self._config_object)
                self.models[model_name] = self.__dict__[model_name]
                self.models[model_name].config.check()

                # link to the group
                group_name = self.__dict__[model_name].group
                self.__dict__[group_name].add_model(model_name, self.__dict__[model_name])

    def import_routines(self):
        """
        Import routines as defined in ``routines/__init__.py``.

        Routines will be stored as instances with the name as class names.
        All groups will be stored to dictionary ``System.groups``.

        Examples
        --------
        ``System.PFlow`` is the power flow routine instance, and ``System.TDS`` and ``System.EIG`` are
        time-domain analysis and eigenvalue analysis routines, respectively.
        """
        for file, cls_list in all_routines.items():
            for cls_name in cls_list:
                file = importlib.import_module('andes.routines.' + file)
                the_class = getattr(file, cls_name)
                attr_name = cls_name
                self.__dict__[attr_name] = the_class(system=self, config=self._config_object)
                self.routines[attr_name] = self.__dict__[attr_name]
                self.routines[attr_name].config.check()

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

        Switch actions will be disabled if `flat=True` is passed to system.
        """
        for instance in models.values():
            instance.switch_action(self.dae.t)

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

    def _store_calls(self):
        """
        Collect and store model calls into system.
        """
        logger.debug("Collecting Model.calls into System.")

        self.calls['__version__'] = __version__

        for name, mdl in self.models.items():
            self.calls[name] = mdl.calls

    def _list2array(self):
        """
        Helper function to call models' ``list2array`` method, which
        usually performs memory preallocation.
        """
        self.call_models('list2array', self.models)

    def set_config(self, config=None):
        """
        Set configuration for the System object.

        Config for models are routines are passed directly to their constructors.
        """
        if config is not None:
            # set config for system
            if self.__class__.__name__ in config:
                self.config.add(config[self.__class__.__name__])
                logger.debug("Config: set for System")

    def get_config(self):
        """
        Collect config data from models.

        Returns
        -------
        dict
            a dict containing the config from devices; class names are keys and configs in a dict are values.
        """
        config_dict = configparser.ConfigParser()
        config_dict[self.__class__.__name__] = self.config.as_dict()

        all_with_config = OrderedDict(list(self.routines.items()) +
                                      list(self.models.items()))

        for name, instance in all_with_config.items():
            cfg = instance.config.as_dict()
            if len(cfg) > 0:
                config_dict[name] = cfg
        return config_dict

    @staticmethod
    def load_config(conf_path=None):
        """
        Load config from an rc-formatted file.

        Parameters
        ----------
        conf_path : None or str
            Path to the config file. If is `None`, the function body
            will not run.

        Returns
        -------
        configparse.ConfigParser
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)
        logger.info('Loaded config from file "%s"', conf_path)
        return conf

    def save_config(self, file_path=None, overwrite=False):
        """
        Save all system, model, and routine configurations to an rc-formatted file.

        Parameters
        ----------
        file_path : str, optional
            path to the configuration file default to `~/andes/andes.rc`.
        overwrite : bool, optional
            If file exists, True to overwrite without confirmation.
            Otherwise prompt for confirmation.

        Warnings
        --------
        Saved config is loaded back and populated *at system instance creation time*.
        Configs from the config file takes precedence over default config values.
        """
        if file_path is None:
            andes_path = os.path.join(os.path.expanduser('~'), '.andes')
            os.makedirs(andes_path, exist_ok=True)
            file_path = os.path.join(andes_path, 'andes.rc')

        elif os.path.isfile(file_path):
            if not confirm_overwrite(file_path, overwrite=overwrite):
                return

        conf = self.get_config()
        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config written to "%s"', file_path)
        return file_path

    def supported_models(self, export='plain'):
        """
        Return the support group names and model names in a table.

        Returns
        -------
        str
            A table-formatted string for the groups and models
        """

        pairs = list()
        for g in self.groups:
            models = list()
            for m in self.groups[g].models:
                models.append(m)
            if len(models) > 0:
                pairs.append((g, ', '.join(models)))

        tab = Tab(title='Supported Groups and Models',
                  header=['Group', 'Models'],
                  data=pairs,
                  export=export,
                  )

        return tab.draw()

    def as_dict(self, vin=False, skip_empty=True):
        """
        Return system data as a dict where the keys are model names
        and values are dicts. Each dict has parameter names as keys
        and corresponding data in an array as values.

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
