# ANDES, a power system simulation tool for research.
#
# Copyright 2015-2029 Hantao Cui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Power system class
"""
import pathlib
import configparser
import importlib
import logging
import os
import inspect
import numpy as np
from cvxopt import spmatrix
from operator import itemgetter
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional  # NOQA

from . import routines
from .config import System
from .consts import pi, rad2deg
from .models import non_jits, jits, JIT, all_models_list
from .utils import get_config_load_path
from .variables import FileMan, DevMan, DAE, VarName, VarOut, Call, Report

from .variables.dae import DAENew
from andes.programs import all_programs
from andes.devices import non_jit
from andes.core.model import Model
from andes.common.config import Config

try:
    from andes_addon.streaming import Streaming
    STREAMING = True
except ImportError:
    STREAMING = False

IP_ADD = False
if hasattr(spmatrix, 'ipadd'):
    IP_ADD = True


logger = logging.getLogger(__name__)


class SystemNew(object):
    """
    New power system class
    """
    def __init__(self,
                 name: Optional[str] = None,
                 config_path: Optional[str] = None,
                 options: Optional[Dict] = None,
                 **kwargs: Optional[Dict]
                 ):
        self.name = name
        self.options = options  # options from command line or so
        self.calls = OrderedDict()
        self.models = OrderedDict()
        self.groups = OrderedDict()
        self.programs = OrderedDict()

        # get and load default config file
        self.config = Config()
        self._config_path = get_config_load_path(file_name='andes.rc') if not config_path else config_path
        self._config_from_file = self.load_config(self._config_path)
        self.set_config(self._config_from_file)  # only load config for system and routines
        # custom configuration for system goes after this line

        self.dae = DAENew()
        self.group_import()
        self.model_import()
        # routine import comes after model import; routines need to query model flags
        self.routine_import()

        self._models_with_flag = {'pflow': self.get_models_with_flag('pflow'),
                                  'tds': self.get_models_with_flag('tds'),
                                  'pflow_and_tds': self.get_models_with_flag(('tds', 'pflow')),
                                  }

        # ------------------------------
        self.f_adders = []
        self.g_adders = []
        self.f_setters = []
        self.g_setters = []

        self.x_setters = []
        self.y_setters = []
        self.x_adders = []
        self.y_adders = []
        # ------------------------------

    def group_import(self):
        """
        Import groups defined in `devices/group.py`

        Returns
        -------

        """
        module = importlib.import_module('andes.devices.group')
        for m in inspect.getmembers(module, inspect.isclass):
            name = m[0]
            cls = m[1]
            if name == 'GroupBase':
                continue
            self.__dict__[name] = cls()
            self.groups[name] = self.__dict__[name]

    def model_import(self):
        """
        Import and instantiate the non-JIT models and the JIT models.

        Models defined in ``jits`` and ``non_jits`` in ``models/__init__.py``
        will be imported and instantiated accordingly.

        Returns
        -------
        None
        """
        # non-JIT models
        for file, cls_list in non_jit.items():
            for model_name in cls_list:
                the_module = importlib.import_module('andes.devices.' + file)
                the_class = getattr(the_module, model_name)
                self.__dict__[model_name] = the_class(system=self, config=self._config_from_file)
                self.models[model_name] = self.__dict__[model_name]

                # link to the group
                group_name = self.__dict__[model_name].group
                self.__dict__[group_name].add_model(model_name, self.__dict__[model_name])

        # import JIT models
        # for file, pair in jits.items():
        #     for cls, name in pair.items():
        #         self.__dict__[name] = JIT(self, file, cls, name)

    def routine_import(self):
        """
        Dynamically import routines as defined in ``routines/__init__.py``.

        The command-line argument ``--routine`` is defined in ``__cli__`` in
        each routine file. A routine instance will be stored in the system
        instance with the name being all lower case.

        For example, a routine for power flow study should be defined in
        ``routines/pflow.py`` where ``__cli__ = 'pflow'``. The class name
        for the routine should be ``Pflow``. The routine instance will be
        saved to ``PowerSystem.pflow``.

        Returns
        -------
        None
        """
        for file, cls_list in all_programs.items():
            for cls_name in cls_list:
                file = importlib.import_module('andes.programs.' + file)
                the_class = getattr(file, cls_name)
                attr_name = cls_name
                self.__dict__[attr_name] = the_class(system=self, config=self._config_from_file)
                self.programs[attr_name] = self.__dict__[attr_name]

    def get_models_with_flag(self, flag: Optional[Union[str, Tuple]] = None):
        if isinstance(flag, str):
            flag = [flag]

        out = OrderedDict()
        for name, mdl in self.models.items():
            for f in flag:
                if mdl.flags[f] is True:
                    out[name] = mdl
                    break
        return out

    def link_external(self, models=None):
        if models is None:
            models = self._models_with_flag['pflow']

        for mdl in models.values():
            # TODO: handle external groups
            for name, instance in OrderedDict(list(mdl.cache.vars_ext.items()) +
                                              list(mdl.params_ext.items()) +
                                              list(mdl.services_ext.items())
                                              ).items():
                ext_name = instance.model
                if ext_name in self.groups:
                    ext_model = self.groups[ext_name]
                elif ext_name in self.models:
                    ext_model = self.models[ext_name]
                else:
                    raise KeyError

                instance.link_external(ext_model)

    def set_address(self, models=None):
        if models is None:
            models = self._models_with_flag['pflow']

        for mdl in models.values():
            if mdl.flags['address'] is True:
                logger.debug(f'{mdl.class_name} addresses exist.')
                continue

            collate = mdl.flags['collate']

            n = mdl.n
            m0 = self.dae.m
            n0 = self.dae.n
            m_end = m0 + len(mdl.algebs) * n
            n_end = n0 + len(mdl.states) * n

            if not collate:
                for idx, (name, item) in enumerate(mdl.algebs.items()):
                    item.set_address(np.arange(m0 + idx * n, m0 + (idx + 1) * n))
                for idx, (name, item) in enumerate(mdl.states.items()):
                    item.set_address(np.arange(n0 + idx * n, n0 + (idx + 1) * n))
            else:
                for idx, (name, item) in enumerate(mdl.algebs.items()):
                    item.set_address(np.arange(m0 + idx, m_end, len(mdl.algebs)))
                for idx, (name, item) in enumerate(mdl.states.items()):
                    item.set_address(np.arange(n0 + idx, n_end, len(mdl.states)))

            self.dae.m = m_end
            self.dae.n = n_end

            mdl.flags['address'] = True

        # pre-allocate for names
        if len(self.dae.y_name) == 0:
            self.dae.x_name = [''] * self.dae.n
            self.dae.y_name = [''] * self.dae.m
        else:
            self.dae.x_name.extend([''] * (self.dae.n - len(self.dae.x_name)))
            self.dae.y_name.extend([''] * (self.dae.m - len(self.dae.y_name)))

    def set_dae_names(self, models=None):
        # store variable names
        # FIXME: fix the messy code below

        if models is None:
            models = self._models_with_flag['pflow']

        for mdl in models.values():
            mdl_name = mdl.class_name
            for name, item in mdl.algebs.items():
                for uid, addr in enumerate(item.a):
                    self.dae.y_name[addr] = f'{mdl_name}_{item.name}_{uid}'
            for name, item in mdl.states.items():
                for uid, addr in enumerate(item.a):
                    self.dae.x_name[addr] = f'{mdl_name}_{item.name}_{uid}'

    def add(self, model, param_dict=None, **kwargs):
        group_name = self.__dict__[model].group
        group = self.groups[group_name]

        if kwargs is not None:
            param_dict.update(kwargs)

        idx = param_dict.pop('idx', None)
        idx = group.get_next_idx(idx=idx, model_name=model)
        self.__dict__[model].add(idx=idx, **param_dict)
        group.add(idx=idx, model=self.__dict__[model])

    def _finalize_add(self, models=None):
        self._call_models_method('finalize_add', self.models)

    def generate_initializers(self):
        # TODO: consider both JIT and non-JIT models
        self._call_models_method('generate_initializer', self.models)

    def generate_equations(self):
        self._call_models_method('generate_equations', self.models)

    def generate_jacobians(self):
        self._call_models_method('generate_jacobians', self.models)

    def initialize(self, models: Optional[Union[str, List, OrderedDict]] = None, tds=False):
        if tds is False:
            # clear x and y values for power flow only
            self.dae.reset_xy()
        self.vars_to_models()

        self._call_models_method('initialize', models)

        self.vars_to_dae()
        self.vars_to_models()
        return np.hstack((self.dae.x, self.dae.y))

    def vars_to_dae(self):
        self.dae.reset_xy()
        # from variables to dae variables
        for var in self.x_adders:
            if var.v_init is None:
                continue
            np.add.at(self.dae.x, var.a, var.v)
        for var in self.x_setters:
            np.put(self.dae.x, var.a, var.v)

        # NOTE: Need to skip vars that are not initializers for re-entrance
        for var in self.y_adders:
            if var.v_init is None:
                continue
            np.add.at(self.dae.y, var.a, var.v)
        for var in self.y_setters:
            np.put(self.dae.y, var.a, var.v)

    def vars_to_models(self):
        for var in self.y_adders + self.y_setters:
            var.v = self.dae.y[var.a]

        for var in self.x_adders + self.x_setters:
            var.v = self.dae.x[var.a]

    def store_adder_setter(self):
        self.f_adders = []
        self.g_adders = []
        self.f_setters = []
        self.g_setters = []

        self.x_setters = []
        self.y_setters = []
        self.x_adders = []
        self.y_adders = []

        for mdl in self.models.values():
            if not mdl.n:
                continue
            for var in mdl.cache.all_vars.values():
                if var.has_address:
                    if var.e_setter is False:
                        self.__dict__[f'{var.e_code}_adders'].append(var)
                    else:
                        self.__dict__[f'{var.e_code}_setters'].append(var)

                    if var.v_setter is False:
                        self.__dict__[f'{var.v_code}_adders'].append(var)
                    else:
                        self.__dict__[f'{var.v_code}_setters'].append(var)

    def s_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('eval_service', models)

    def l_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        # TODO:
        # This function should somehow return the indices and values
        # of variables that are pegged at the limit.

        # The limited values need to sent to solvers
        # such as `scipy.optimize.newton_krylov` to make the result correct

        self._call_models_method('eval_limiter', models)
        self.vars_to_dae()
        self.vars_to_models()

    def e_clear(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self.dae.reset_fg()
        self._call_models_method('e_clear', models)

    def f_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('f_update', models)

        for var in self.f_adders:
            np.add.at(self.dae.f, var.a, var.e)
        for var in self.f_setters:
            np.put(self.dae.f, var.a, var.e)

        return self.dae.f

    def g_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('g_update', models)

        for var in self.g_adders:
            np.add.at(self.dae.g, var.a, var.e)
        for var in self.g_setters:
            np.put(self.dae.g, var.a, var.e)

        return self.dae.g

    def j_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('j_update', models)

        self.dae.reset_sparse()

        # collect sparse values into sparse structures
        jac_name = ('fx', 'fy', 'gx', 'gy')
        for j_name in jac_name:
            j_size = self.dae.get_size(j_name)

            for mdl in self.models.values():
                for row, col, val in mdl.zip_ijv(j_name):
                    # TODO: use `spmatrix.ipadd` if available
                    # TODO: fix `ipadd` to get rid of type checking
                    if isinstance(val, float) or len(val) > 0:
                        self.dae.__dict__[j_name] += spmatrix(val, row, col, j_size, 'd')

    def store_sparse_pattern(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('store_sparse_pattern', models)
        jac_name = ('fx', 'fy', 'gx', 'gy')

        # add variable jacobian values
        for j_name in jac_name:
            ii, jj, vv = list(), list(), list()

            # for `gy` matrix, always make sure the diagonal is reserved
            # It is a safeguard if the modeling user omitted the diagonal
            # term in the equations
            if j_name == 'gy':
                ii.extend(np.arange(self.dae.m))
                jj.extend(np.arange(self.dae.m))
                vv.extend(np.zeros(self.dae.m))

            for name, mdl in self.models.items():
                row_idx = mdl.row_idx(f'{j_name}')
                col_idx = mdl.col_idx(f'{j_name}')

                ii.extend(row_idx)
                jj.extend(col_idx)
                vv.extend(np.zeros(len(np.array(row_idx))))

                # add the constant jacobian values
                for row, col, val in mdl.zip_ijv(f'{j_name}c'):
                    ii.extend(row)
                    jj.extend(col)

                    if isinstance(val, (float, int)):
                        vv.extend(val * np.ones(len(row)))
                    elif isinstance(val, (list, np.ndarray)):
                        vv.extend(val)
                    else:
                        raise TypeError(f'Unknown type {type(val)} in constant jacobian {jac_name}')

            if len(ii) > 0:
                ii = np.array(ii).astype(int)
                jj = np.array(jj).astype(int)
                vv = np.array(vv).astype(float)

            self.dae.store_sparse_ijv(j_name, ii, jj, vv)
            self.dae.build_pattern(j_name)

    def _store_calls(self):
        for name, mdl in self.models.items():
            self.calls[name] = mdl.calls

    @staticmethod
    def _get_pkl_path():
        pkl_name = 'calls.pkl'
        andes_path = os.path.join(str(pathlib.Path.home()), '.andes')

        if not os.path.exists(andes_path):
            os.makedirs(andes_path)

        pkl_path = os.path.join(andes_path, pkl_name)

        return pkl_path

    def dill_calls(self):
        import dill
        pkl_path = self._get_pkl_path()

        dill.dump(self.calls, open(pkl_path, 'wb'))

    def undill_calls(self):
        import dill
        pkl_path = self._get_pkl_path()

        if not os.path.isfile(pkl_path):
            self.prepare()

        self.calls = dill.load(open(pkl_path, 'rb'))
        logger.debug('System undill: loaded <calls.pkl> file.')
        for name, model_call in self.calls.items():
            self.__dict__[name].calls = model_call

    def _call_models_method(self, method: str, models: Optional[Union[str, list, Model, OrderedDict]]):
        if models is None:
            models = self._models_with_flag['pflow']
        if isinstance(models, str):
            models = {models: self.__dict__[models]}
        elif isinstance(models, Model):
            models = {models.class_name, models}
        elif isinstance(models, list):
            models = OrderedDict()
            for item in models:
                if isinstance(item, Model):
                    models[item.class_name] = item
                elif isinstance(item, str):
                    models[item] = self.__dict__[item]
                else:
                    raise TypeError(f'Unknown type {type(item)}')

        for mdl in models.values():
            getattr(mdl, method)()

    def set_config(self, config=None):
        # NOTE: No need to call `set_config` for models since
        # the config is passed during model import
        if config is not None:
            # set config for system
            if self.__class__.__name__ in config:
                self.config.add(config[self.__class__.__name__])
                logger.debug("Config: set for System")

        else:
            logger.warning('No config provided.')

    def get_config(self):
        """Get config data from models

        Returns
        -------
        dict
            a dict containing the config from devices; class names are the keys
        """
        config_dict = configparser.ConfigParser()
        config_dict[self.__class__.__name__] = self.config.as_dict()

        # get routine config
        for name, instance in self.programs.items():
            cfg = instance.get_config()
            if len(cfg) > 0:
                config_dict[name] = cfg

        # get model config
        for name, instance in self.models.items():
            cfg = instance.get_config()
            if len(cfg) > 0:
                config_dict[name] = cfg
        return config_dict

    @staticmethod
    def load_config(conf_path=None):
        """
        Load config from an ``andes.rc`` file.

        Parameters
        ----------
        conf_path : None or str
            Path to the Andes rc file. If ``None``, the function body
            will not run.

        Returns
        -------
        configparse.ConfigParser
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)
        logger.debug(f'Config: Loaded from file {conf_path}')
        return conf

    def save_config(self, file_path=None):
        """
        Save system and routine configurations to an rc-formatted file.

        Parameters
        ----------
        file_path : str
            path to the configuration file. The user will be prompted if the
            file already exists.

        Returns
        -------
        None
        """
        if file_path is None:
            home_dir = os.path.expanduser('~')
            file_path = os.path.join(home_dir, '.andes', 'andes.rc')

        if os.path.isfile(file_path):
            choice = input(f'Config file {file_path} already exist. Overwrite? [y/N]').lower()
            if len(choice) == 0 or choice[0] != 'y':
                logger.info('No config file overwritten.')
                return

        conf = self.get_config()
        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config: written to {}'.format(file_path))

    def prepare(self):
        """
        Prepare classes and lambda functions

        Anything in this function should be independent of test case
        """
        self.generate_equations()
        self.generate_jacobians()
        self.generate_initializers()
        self._store_calls()
        self.dill_calls()

    def setup(self):
        """
        Set up system for studies

        This function is to be called after all data are added.
        """
        self.set_address()
        self.set_dae_names()
        self._finalize_add()
        self.link_external()
        self.s_update()
        self.store_sparse_pattern()
        self.store_adder_setter()
        self.e_clear()


class PowerSystem(object):
    """
    A power system class to hold models, routines, DAE numerical values,
    file manager, call manager, variable names and valuable values.
    """

    def __init__(self,
                 case='',
                 pid=-1,
                 no_output=False,
                 dump_raw=None,
                 output=None,
                 dynfile=None,
                 addfile=None,
                 config=None,
                 input_path=None,
                 input_format=None,
                 output_format=None,
                 output_path='',
                 gis=None,
                 dime=None,
                 tf=None,
                 **kwargs):
        """
        PowerSystem constructor

        Parameters
        ----------
        case : str, optional
            Path to the case file

        pid : idx, optional
            Process index

        no_output : bool, optional
            ``True`` to disable all updates
        dump_raw : None or str, optional
            Path to the file to dump the raw parameters in the dm format

        output : None or str, optional
            Output case file name, NOT implemented yet

        dynfile : None or str, optional
            Path to the dynamic file for some formats, for example, ``dyr``
            for PSS/E

        addfile : None or str, optional
            Path to the additional dynamic file in the dm format

        config : None or str, optional
            Path to the andes.conf file

        input_format : None or str, optional
            Suggested input file format

        output_format : None or str, optional
            Requested dump file format, NOT implemented yet

        gis : None or str, optional
            Path to the GIS file

        dime : None or str, optional
            DiME server address

        tf : None or float, optional
            Time-domain simulation end time

        kwargs : dict, optional
            Other keyword args
        """
        # set internal flags based on the arguments
        self.pid = pid
        self.files = FileMan(case,
                             input_format=input_format,
                             addfile=addfile,
                             config=config,
                             input_path=input_path,
                             no_output=no_output,
                             dynfile=dynfile,
                             dump_raw=dump_raw,
                             output_path=output_path,
                             output_format=output_format,
                             output=output,
                             **kwargs)

        self.config = System()

        conf_path = get_config_load_path(self.files.config)
        self.load_config(conf_path=conf_path, sys_only=True)

        self.routine_import()
        self.load_config(conf_path=conf_path, routine_only=True)

        self.devman = DevMan(self)
        self.call = Call(self)
        self.dae = DAE(self)
        self.varname = VarName(self)
        self.varout = VarOut(self)
        self.report = Report(self)

        self.loaded_groups = []

        if dime:
            self.config.dime_enable = True
            self.config.dime_server = dime

        if tf:
            self.tds.config.tf = tf

        self.streaming = None
        if not STREAMING:
            if self.config.dime_enable:
                self.config.dime_enable = False
                logger.warning('Missing andes_addon for DiME streaming.')
        else:
            if self.config.dime_enable:
                logger.info('Connecting to DiME at {}'.format(self.config.dime_server))
                self.streaming = Streaming(self)

        self.model_import()

    @property
    def freq(self):
        """System base frequency"""
        return self.config.freq

    @freq.setter
    def freq(self, freq):
        if freq <= 0:
            self.config.freq = 1
        else:
            self.config.freq = freq

    @property
    def wb(self):
        """System base radial frequency"""
        return 2 * pi * self.config.freq

    @property
    def mva(self):
        """System base power in mega voltage-ampere"""
        return self.config.mva

    @mva.setter
    def mva(self, mva):
        self.config.mva = mva

    def setup(self):
        """
        Set up the power system object by executing the following workflow:

         * Sort the loaded models to meet the initialization sequence
         * Create call strings for routines
         * Call the ``setup`` function of the loaded models
         * Assign addresses for the loaded models
         * Call ``dae.setup`` to assign memory for the numerical dae structure
         * Convert model parameters to the system base

        Returns
        -------
        PowerSystem
            The instance of the PowerSystem
        """

        self.devman.sort_device()
        self.call.setup()
        self.model_setup()
        self.xy_addr0()
        self.dae.setup()
        self.to_sysbase()

        return self

    def to_sysbase(self):
        """
        Convert model parameters to system base. This function calls the
        ``data_to_sys_base`` function of the loaded models.

        Returns
        -------
        None
        """
        if self.config.base:
            for item in self.devman.devices:
                self.__dict__[item].data_to_sys_base()

    def to_elembase(self):
        """
        Convert parameters back to element base. This function calls the
        ```data_to_elem_base``` function.

        Returns
        -------
        None
        """
        if self.config.base:
            for item in self.devman.devices:
                self.__dict__[item].data_to_elem_base()

    def group_add(self, name='Ungrouped'):
        """
        Dynamically add a group instance to the system if not exist.

        Parameters
        ----------
        name : str, optional ('Ungrouped' as default)
            Name of the group

        Returns
        -------
        None
        """
        if not hasattr(self, name):
            self.__dict__[name] = Group(self, name)
            self.loaded_groups.append(name)

    def model_import(self):
        """
        Import and instantiate the non-JIT models and the JIT models.

        Models defined in ``jits`` and ``non_jits`` in ``models/__init__.py``
        will be imported and instantiated accordingly.

        Returns
        -------
        None
        """
        # non-JIT models
        for file, pair in non_jits.items():
            for cls, name in pair.items():
                themodel = importlib.import_module('andes.models.' + file)
                theclass = getattr(themodel, cls)
                self.__dict__[name] = theclass(self, name)

                group = self.__dict__[name]._group
                if group is None:
                    raise ValueError("Class definition incomplete. Group not defined for class {}".format(name))
                self.group_add(group)
                self.__dict__[group].register_model(name)

                self.devman.register_device(name)

        # import JIT models
        for file, pair in jits.items():
            for cls, name in pair.items():
                self.__dict__[name] = JIT(self, file, cls, name)

    def routine_import(self):
        """
        Dynamically import routines as defined in ``routines/__init__.py``.

        The command-line argument ``--routine`` is defined in ``__cli__`` in
        each routine file. A routine instance will be stored in the system
        instance with the name being all lower case.

        For example, a routine for power flow study should be defined in
        ``routines/pflow.py`` where ``__cli__ = 'pflow'``. The class name
        for the routine should be ``Pflow``. The routine instance will be
        saved to ``PowerSystem.pflow``.

        Returns
        -------
        None
        """
        for r in routines.__all__:
            file = importlib.import_module('.' + r.lower(), 'andes.routines')
            self.__dict__[r.lower()] = getattr(file, r)(self)

    def model_setup(self):
        """
        Call the ``setup`` function of the loaded models. This function is
        to be called after parsing all the data files during the system set up.

        Returns
        -------
        None
        """
        for device in self.devman.devices:
            try:
                self.__dict__[device].setup()
            except Exception as e:
                raise e

    def xy_addr0(self):
        """
        Assign indicies and variable names for variables used in power flow

        For each loaded model with the ``pflow`` flag as ``True``, the following
        functions are called sequentially:

         * ``_addr()``
         * ``_intf_network()``
         * ``_intf_ctrl()``

        After resizing the ``varname`` instance, variable names from models
        are stored by calling ``_varname()``

        Returns
        -------
        None
        """
        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.varname.resize()

        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if pflow:
                self.__dict__[device]._varname()

    def xy_addr1(self):
        """
        Assign indices and variable names for variables after power flow.
        This function is for loaded models that do not have the ``pflow``
        flag as ``True``.
        """
        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if not pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.varname.resize()

        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if not pflow:
                self.__dict__[device]._varname()

    def rmgen(self, idx):
        """
        Remove the static generators if their dynamic models exist

        Parameters
        ----------
        idx : list
            A list of static generator idx
        Returns
        -------
        None
        """
        stagens = []
        for device, stagen in zip(self.devman.devices, self.call.stagen):
            if stagen:
                stagens.append(device)
        for gen in idx:
            for stagen in stagens:
                if gen in self.__dict__[stagen].uid.keys():
                    self.__dict__[stagen].disable_gen(gen)

    def check_event(self, sim_time):
        """
        Check for event occurrance for``Event`` group models at ``sim_time``

        Parameters
        ----------
        sim_time : float
            The current simulation time

        Returns
        -------
        list
            A list of model names who report (an) event(s) at ``sim_time``
        """
        ret = []
        for model in self.__dict__['Event'].all_models:
            if self.__dict__[model].is_time(sim_time):
                ret.append(model)

        if self.Breaker.is_time(sim_time):
            ret.append('Breaker')

        return ret

    def get_data_example(self, model, n_per_line=5):
        """
        Return a string of example data entry that can be used in a dm input file

        Returns
        -------
        str
            A string containing the example data
        """
        if model not in all_models_list:
            raise KeyError('Model <{}> is invalid'.format(model))

        model_name = self.__dict__[model]._name

        data = OrderedDict()
        data.update({'idx': None, 'name': '"{}"'.format(model_name)})
        data.update(self.__dict__[model]._data)

        out = '{}, '.format(model_name)
        nspace = len(out)
        n_param = len(data)

        for idx, (key, val) in enumerate(data.items()):
            if (idx > 0) and (divmod(idx, n_per_line)[1] == 0):
                out += '\n' + ' ' * nspace
            if idx == n_param - 1:
                out += '{} = {}'.format(key, val)
            else:
                out += '{} = {}, '.format(key, val)

        return out

    def get_event_times(self):
        """
        Return event times of Fault, Breaker and other timed events

        Returns
        -------
        list
            A sorted list of event times
        """
        times = []

        times.extend(self.Breaker.get_times())

        for model in self.__dict__['Event'].all_models:
            times.extend(self.__dict__[model].get_times())

        if times:
            times = sorted(list(set(times)))

        return times

    def load_config(self, conf_path, sys_only=False, routine_only=False):
        """
        Load config from an ``andes.conf`` file.

        This function creates a ``configparser.ConfigParser`` object to read
        the specified conf file and calls the ``load_config`` function of the
        config instances of the system and the routines.

        Parameters
        ----------
        conf_path : None or str
            Path to the Andes config file. If ``None``, the function body
            will not run.

        Returns
        -------
        None
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)
        if sys_only:
            self.config.load_config(conf)
            return
        if routine_only:
            for r in routines.__all__:
                self.__dict__[r.lower()].config.load_config(conf)

        logger.debug('Loaded config file from {}.'.format(conf_path))

    def dump_config(self, file_path):
        """
        Dump system and routine configurations to an rc-formatted file.

        Parameters
        ----------
        file_path : str
            path to the configuration file. The user will be prompted if the
            file already exists.

        Returns
        -------
        None
        """
        if os.path.isfile(file_path):
            logger.debug('File {} alreay exist. Overwrite? [y/N]'.format(file_path))
            choice = input('File {} alreay exist. Overwrite? [y/N]'.format(file_path)).lower()
            if len(choice) == 0 or choice[0] != 'y':
                logger.info('File not overwritten.')
                return

        conf = self.config.dump_conf()
        for r in routines.__all__:
            conf = self.__dict__[r.lower()].config.dump_conf(conf)

        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config written to {}'.format(file_path))

    def check_islands(self, show_info=False):
        """
        Check the connectivity for the ac system

        Parameters
        ----------
        show_info : bool
            Show information when the system has islands. To be used when
            initializing power flow.

        Returns
        -------
        None
        """
        if not hasattr(self, 'Line'):
            logger.error('<Line> device not found.')
            return
        self.Line.connectivity(self.Bus)

        if show_info is True:

            if len(self.Bus.islanded_buses) == 0 and len(
                    self.Bus.island_sets) == 0:
                logger.debug('System is interconnected.')
            else:
                logger.info(
                    'System contains {:d} islands and {:d} islanded buses.'.
                    format(
                        len(self.Bus.island_sets),
                        len(self.Bus.islanded_buses)))

            nosw_island = []  # no slack bus island
            msw_island = []  # multiple slack bus island
            for idx, island in enumerate(self.Bus.island_sets):
                nosw = 1
                for item in self.SW.bus:
                    if self.Bus.uid[item] in island:
                        nosw -= 1
                if nosw == 1:
                    nosw_island.append(idx)
                elif nosw < 0:
                    msw_island.append(idx)

            if nosw_island:
                logger.warning(
                    'Slack bus is not defined for {:g} island(s).'.format(
                        len(nosw_island)))
            if msw_island:
                logger.warning(
                    'Multiple slack buses are defined for {:g} island(s).'.
                    format(len(nosw_island)))

            if (not nosw_island) and (not msw_island):
                logger.debug(
                    'Each island has a slack bus correctly defined.')

    def get_busdata(self, sort_names=False):
        """
        get ac bus data from solved power flow
        """
        if self.pflow.solved is False:
            logger.error('Power flow not solved when getting bus data.')
            return tuple([False] * 8)
        idx = self.Bus.idx
        names = self.Bus.name
        Vm = [self.dae.y[x] for x in self.Bus.v]
        if self.pflow.config.usedegree:
            Va = [self.dae.y[x] * rad2deg for x in self.Bus.a]
        else:
            Va = [self.dae.y[x] for x in self.Bus.a]

        Pg = [self.Bus.Pg[x] for x in range(self.Bus.n)]
        Qg = [self.Bus.Qg[x] for x in range(self.Bus.n)]
        Pl = [self.Bus.Pl[x] for x in range(self.Bus.n)]
        Ql = [self.Bus.Ql[x] for x in range(self.Bus.n)]

        if sort_names:
            ret = (list(x) for x in zip(*sorted(
                zip(idx, names, Vm, Va, Pg, Qg, Pl, Ql), key=itemgetter(0))))
        else:
            ret = idx, names, Vm, Va, Pg, Qg, Pl, Ql

        return ret

    def get_nodedata(self, sort_names=False):
        """
        get dc node data from solved power flow
        """
        if not self.Node.n:
            return
        if not self.pflow.solved:
            logger.error('Power flow not solved when getting bus data.')
            return tuple([False] * 7)
        idx = self.Node.idx
        names = self.Node.name
        V = [self.dae.y[x] for x in self.Node.v]

        if sort_names:
            ret = (list(x)
                   for x in zip(*sorted(zip(idx, names, V), key=itemgetter(0))))
        else:
            ret = idx, names, V

        return ret

    def get_linedata(self, sort_names=False):
        """
        get line data from solved power flow
        """
        if not self.pflow.solved:
            logger.error('Power flow not solved when getting line data.')
            return tuple([False] * 7)
        idx = self.Line.idx
        fr = self.Line.bus1
        to = self.Line.bus2

        Sloss = self.Line.S1 + self.Line.S2

        Pfr = list(self.Line.S1.real())
        Qfr = list(self.Line.S1.imag())
        Pto = list(self.Line.S2.real())
        Qto = list(self.Line.S2.imag())

        Ploss = list(Sloss.real())
        Qloss = list(Sloss.imag())

        if sort_names:
            ret = (list(x) for x in zip(*sorted(
                zip(idx, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss),
                key=itemgetter(0))))
        else:
            ret = idx, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss

        return ret


class GroupMeta(type):
    def __new__(cls, name, base, attr_dict):
        return super(GroupMeta, cls).__new__(cls, name, base, attr_dict)


class Group(metaclass=GroupMeta):
    """
    Group class for registering models and elements.

    Also handles reading and setting attributes.
    """

    def __init__(self, system, name):
        self.system = system
        self.name = name
        self.all_models = []
        self._idx_model = {}
        self._idx = []

    def register_model(self, model):
        """
        Register ``model`` to this group

        :param model: model name
        :return: None
        """

        if not isinstance(model, str):
            # TODO: consider removing this constrain
            raise KeyError("Name of the model must be a string")
        if model not in self.all_models:
            self.all_models.append(model)

    def register_element(self, model, idx):
        """
        Register element with index ``idx`` to ``model``

        :param model: model name
        :param idx: element idx
        :return: final element idx
        """

        if idx is None:
            idx = model + '_' + str(len(self._idx_model))

        self._idx_model[idx] = model
        self._idx.append(idx)

        return idx

    def get_field(self, field, idx):
        """
        Return the field ``field`` of elements ``idx`` in the group

        :param field: field name
        :param idx: element idx
        :return: values of the requested field
        """
        ret = []
        scalar = False

        # TODO: ensure idx is unique in this Group

        if isinstance(idx, (int, float, str)):
            scalar = True
            idx = [idx]

        models = [self._idx_model[i] for i in idx]

        for i, m in zip(idx, models):
            ret.append(self.system.__dict__[m].get_field(field, idx=i))

        if scalar is True:
            return ret[0]
        else:
            return ret

    def set_field(self, field, idx, value):
        """
        Set the field ``field`` of elements ``idx`` to ``value``.

        This function does not if the field is valid for all models.

        :param field: field name
        :param idx: element idx
        :param value: value of fields to set
        :return: None
        """

        if isinstance(idx, (int, float, str)):
            idx = [idx]
        if isinstance(value, (int, float)):
            value = [value]

        models = [self._idx_model[i] for i in idx]

        for i, m, v in zip(idx, models, value):
            if not hasattr(self.system.__dict__[m], field):
                raise KeyError('Field <{}> is not valid for model <{}>'.format(field, m))

            uid = self.system.__dict__[m].get_uid(idx)
            self.system.__dict__[m].__dict__[field][uid] = v
