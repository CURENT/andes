"""
System class for power system data and methods
"""

import pathlib
import configparser
import importlib
import logging
import os
import inspect
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

from andes.utils.paths import get_config_path
from andes.variables.dae import DAE
from andes.routines import all_routines
from andes.models import non_jit
from andes.core.param import BaseParam
from andes.core.model import Model
from andes.core.discrete import AntiWindupLimiter
from andes.core.config import Config
from andes.variables.fileman import FileMan

from andes.shared import np, spmatrix

IP_ADD = False
if hasattr(spmatrix, 'ipadd'):
    IP_ADD = True


logger = logging.getLogger(__name__)


class System(object):
    """
    New power system class
    """
    def __init__(self,
                 name: Optional[str] = None,
                 config_path: Optional[str] = None,
                 options: Optional[Dict] = None,
                 **kwargs
                 ):
        self.name = name
        self.options = {} if options is None else options
        if kwargs:
            self.options.update(kwargs)
        self.calls = OrderedDict()
        self.models = OrderedDict()
        self.groups = OrderedDict()
        self.programs = OrderedDict()
        self.switch_times = np.array([])

        # get and load default config file
        self.config = Config(self.__class__.__name__)
        self._config_path = get_config_path() if not config_path else config_path
        self._config_from_file = self.load_config(self._config_path)
        self.config.load(self._config_from_file)  # only load config for system and routines

        # custom configuration for system goes after this line
        self.config.add(OrderedDict((('freq', 60),
                                     ('mva', 100),
                                     )))

        self.files = FileMan()
        if self.options is not None:
            self.files.set(**self.options)

        self.dae = DAE()
        # routine import comes after model import; routines need to query model flags
        self._group_import()
        self._model_import()
        self._routine_import()

        self._models_with_flag = {'pflow': self.get_models_with_flag('pflow'),
                                  'tds': self.get_models_with_flag('tds'),
                                  'pflow_and_tds': self.get_models_with_flag(('tds', 'pflow')),
                                  }

        # ------------------------------
        # FIXME: reduce clutter with defaultdict `adders` and `setters`, each with `x`, `y`, `f`, and `g`
        self.f_adders, self.f_setters = list(), list()
        self.g_adders, self.g_setters = list(), list()

        self.x_adders, self.x_setters = list(), list()
        self.y_adders, self.y_setters = list(), list()
        self.antiwindups = list()
        # ------------------------------

    def prepare(self, quick=False):
        """
        Prepare classes and lambda functions

        Anything in this function should be independent of test case
        """
        self._generate_symbols()
        self._generate_equations()
        self._generate_jacobians()
        self._generate_initializers()
        if quick is False:
            self._generate_pretty_print()
        self._check_group_common()
        self._store_calls()
        self.dill_calls()

    def setup(self):
        """
        Set up system for studies

        This function is to be called after all data are added.
        """
        self.set_address()
        self.set_dae_names()
        self._collect_ref_param()
        self._list2array()
        self.calc_pu_coeff()
        self.store_sparse_pattern()
        self.store_adder_setter()

    def reset(self):
        """
        Reset to the state after reading data and setup (before power flow)

        Returns
        -------

        """

        self.dae.reset()
        self._call_models_method('a_reset', models=self.models)
        self.e_clear()
        self._p_restore()
        self.setup()

    def add(self, model, param_dict=None, **kwargs):
        if model not in self.models:
            logger.warning(f"<{model}> is not an existing model.")
            return
        group_name = self.__dict__[model].group
        group = self.groups[group_name]

        if param_dict is None:
            param_dict = {}
        if kwargs is not None:
            param_dict.update(kwargs)

        idx = param_dict.pop('idx', None)
        idx = group.get_next_idx(idx=idx, model_name=model)
        self.__dict__[model].add(idx=idx, **param_dict)
        group.add(idx=idx, model=self.__dict__[model])

    def set_address(self, models=None):
        if models is None:
            models = self._models_with_flag['pflow']

        # set internal variable addresses
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
                for idx, item in enumerate(mdl.algebs.values()):
                    item.set_address(np.arange(m0 + idx * n, m0 + (idx + 1) * n))
                for idx, item in enumerate(mdl.states.values()):
                    item.set_address(np.arange(n0 + idx * n, n0 + (idx + 1) * n))
            else:
                for idx, item in enumerate(mdl.algebs.values()):
                    item.set_address(np.arange(m0 + idx, m_end, len(mdl.algebs)))
                for idx, item in enumerate(mdl.states.values()):
                    item.set_address(np.arange(n0 + idx, n_end, len(mdl.states)))

            self.dae.m = m_end
            self.dae.n = n_end

            mdl.flags['address'] = True

        # set external variable addresses
        for mdl in models.values():
            # handle external groups
            for name, instance in mdl.cache.vars_ext.items():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError(f'<{ext_name}> is not a model or group name.')

                instance.link_external(ext_model)

        # pre-allocate for names
        if len(self.dae.y_name) == 0:
            self.dae.x_name = [''] * self.dae.n
            self.dae.y_name = [''] * self.dae.m
            self.dae.x_tex_name = [''] * self.dae.n
            self.dae.y_tex_name = [''] * self.dae.m
        else:
            self.dae.x_name.extend([''] * (self.dae.n - len(self.dae.x_name)))
            self.dae.y_name.extend([''] * (self.dae.m - len(self.dae.y_name)))
            self.dae.x_tex_name.extend([''] * (self.dae.n - len(self.dae.x_tex_name)))
            self.dae.y_tex_name.extend([''] * (self.dae.m - len(self.dae.y_tex_name)))

    def set_dae_names(self, models=None):
        # store variable names

        if models is None:
            models = self._models_with_flag['pflow']

        for mdl in models.values():
            mdl_name = mdl.class_name
            for name, item in mdl.algebs.items():
                for uid, addr in enumerate(item.a):
                    self.dae.y_name[addr] = f'{mdl_name} {name} {uid}'
                    self.dae.y_tex_name[addr] = rf'${item.tex_name}\ {mdl_name}\ {uid}$'
            for name, item in mdl.states.items():
                for uid, addr in enumerate(item.a):
                    self.dae.x_name[addr] = f'{mdl_name} {name} {uid}'
                    self.dae.x_tex_name[addr] = rf'${item.tex_name}\ {mdl_name}\ {uid}$'

    def initialize(self, models: Optional[Union[str, List, OrderedDict]] = None):
        if models is None:
            models = self._models_with_flag['pflow']

        for mdl in models.values():
            # link externals first
            for instance in mdl.services_ext.values():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError(f'<{ext_name}> is not a model or group name.')

                instance.link_external(ext_model)

            # initialize variables second
            mdl.initialize()

            # Might need to relay data back and forth ????
            # send data back and forth
            self.vars_to_dae()
            self.vars_to_models()

    def store_adder_setter(self, models=None):
        models = self._get_models(models)

        self.f_adders, self.f_setters = list(), list()
        self.g_adders, self.g_setters = list(), list()

        self.x_adders, self.x_setters = list(), list()
        self.y_adders, self.y_setters = list(), list()

        for mdl in models.values():
            if not mdl.n:
                continue
            for var in mdl.cache.all_vars.values():
                if var.e_setter is False:
                    self.__dict__[f'{var.e_code}_adders'].append(var)
                else:
                    self.__dict__[f'{var.e_code}_setters'].append(var)

                if var.v_setter is False:
                    self.__dict__[f'{var.v_code}_adders'].append(var)
                else:
                    self.__dict__[f'{var.v_code}_setters'].append(var)
            for item in mdl.discrete.values():
                if isinstance(item, AntiWindupLimiter):
                    self.antiwindups.append(item)

    def calc_pu_coeff(self):
        """
        Calculate per unit conversion factor; store input parameters to `vin`, and perform the conversion
        Returns
        -------

        """
        Sb = self.config.mva

        # for each model, get external parameters with `link_external` and then calculate the pu coeff
        for mdl in self.models.values():
            for instance in mdl.params_ext.values():
                ext_name = instance.model
                try:
                    ext_model = self.__dict__[ext_name]
                except KeyError:
                    raise KeyError(f'<{ext_name}> is not a model or group name.')

                try:
                    instance.link_external(ext_model)
                except IndexError:
                    raise IndexError(f'Model <{mdl.class_name}> param <{instance.name}> link parameter error')

            # default Sn to Sb if not provided. Some controllers might not have Sn or Vn
            if 'Sn' in mdl.params:
                Sn = mdl.Sn.v
            else:
                Sn = Sb

            # If both Vn and Vn1 are not provided, default to Vn = Vb = 1
            # test if is shunt-connected or series-connected to bus, or unconnected to bus
            Vb, Vn = 1, 1
            if 'bus' in mdl.params or 'bus1' in mdl.params:
                if 'bus' in mdl.params:
                    Vb = self.Bus.get(src='Vn', idx=mdl.bus.v, attr='v')
                    Vn = mdl.Vn.v if 'Vn' in mdl.params else Vb
                elif 'bus1' in mdl.params:
                    Vb = self.Bus.get(src='Vn', idx=mdl.bus1.v, attr='v')
                    Vn = mdl.Vn1.v if 'Vn1' in mdl.params else Vb

            Zn = (Vn ** 2 / Sn)
            Zb = (Vb ** 2 / Sb)

            if 'node' in mdl.params or 'node1' in mdl.params:
                raise NotImplementedError('Per unit conversion for DC models is not implemented')
            # TODO: handle DC voltages similarly

            coeffs = {'voltage': Vn / Vb,
                      'power': Sn / Sb,
                      'ipower': Sb / Sn,
                      'current': (Sn / Vn) / (Sb / Vb),
                      'z': Zn / Zb,
                      'y': Zb / Zn,
                      }

            for prop, coeff in coeffs.items():
                for p in mdl.find_param(prop).values():
                    p.set_pu_coeff(coeff)

    def l_update_var(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('l_update_var', models)

    def l_check_eq(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('l_check_eq', models)

    def l_set_eq(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('l_set_eq', models)

    def fg_to_dae(self):
        self._e_to_dae('f')
        self._e_to_dae('g')

        # update variable values set by anti-windup limiters
        for item in self.antiwindups:
            if len(item.x_set) > 0:
                for key, val in item.x_set:
                    np.put(self.dae.x, key, val)

    def f_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('f_update', models)

    def g_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self._call_models_method('g_update', models)

    def j_update(self, models: Optional[Union[str, List, OrderedDict]] = None):
        models = self._get_models(models)
        self._call_models_method('j_update', models)

        self.dae.restore_sparse()
        # collect sparse values into sparse structures
        for j_name in self.dae.jac_name:
            j_size = self.dae.get_size(j_name)

            for mdl in models.values():
                for row, col, val in mdl.zip_ijv(j_name):
                    # TODO: use `spmatrix.ipadd` if available
                    # TODO: fix `ipadd` to get rid of type checking
                    if isinstance(val, np.float64):
                        # Workaround for CVXOPT's handling of np.float64
                        val = float(val)
                    if isinstance(val, (int, float)) or len(val) > 0:
                        try:
                            self.dae.__dict__[j_name] += spmatrix(val, row, col, j_size, 'd')
                        except TypeError as e:
                            logger.error(f'{mdl.class_name}: j_name {j_name}, row={row}, col={col}, val={val}, '
                                         f'j_size={j_size}')
                            raise e

    def store_sparse_pattern(self, models: Optional[Union[str, List, OrderedDict]] = None):
        models = self._get_models(models)
        self._call_models_method('store_sparse_pattern', models)

        # add variable jacobian values
        for j_name in self.dae.jac_name:
            ii, jj, vv = list(), list(), list()

            # for `gy` matrix, always make sure the diagonal is reserved
            # It is a safeguard if the modeling user omitted the diagonal
            # term in the equations
            if j_name == 'gy':
                ii.extend(np.arange(self.dae.m))
                jj.extend(np.arange(self.dae.m))
                vv.extend(np.zeros(self.dae.m))

            # logger.debug(f'Jac <{j_name}>, row={ii}')

            for mdl in models.values():
                row_idx = mdl.row_of(f'{j_name}')
                col_idx = mdl.col_of(f'{j_name}')

                # logger.debug(f'Model <{name}>, row={row_idx}')
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
                        raise TypeError(f'Unknown type {type(val)} in constant jacobian {j_name}')

            if len(ii) > 0:
                ii = np.array(ii).astype(int)
                jj = np.array(jj).astype(int)
                vv = np.array(vv).astype(float)

            self.dae.store_sparse_ijv(j_name, ii, jj, vv)
            self.dae.build_pattern(j_name)

    def vars_to_dae(self):
        """
        From variables to dae variables

        For adders, only those with `v_str` can set the value. ??????

        Returns
        -------

        """
        self.dae.clear_xy()
        self._v_to_dae('x')
        self._v_to_dae('y')

    def vars_to_models(self):
        for var in self.y_adders + self.y_setters:
            if var.n > 0:
                var.v = self.dae.y[var.a]

        for var in self.x_adders + self.x_setters:
            if var.n > 0:
                var.v = self.dae.x[var.a]

    def _v_to_dae(self, v_name):
        """
        Helper function for collecting variable values into dae structures `x` and `y`

        Parameters
        ----------
        v_name

        Returns
        -------

        """
        if v_name not in ('x', 'y'):
            raise KeyError(f'{v_name} is not a valid var name')

        for var in self.__dict__[f'{v_name}_adders']:
            # NOTE: Need to skip vars that are not initializers for re-entrance
            if var.v_str is None or (var.n == 0):
                continue
            if var.owner.flags['initialized'] is False:
                continue
            np.add.at(self.dae.__dict__[v_name], var.a, var.v)
        for var in self.__dict__[f'{v_name}_setters']:
            if var.owner.flags['initialized'] is False:
                continue
            if var.n > 0:
                np.put(self.dae.__dict__[v_name], var.a, var.v)

    def _e_to_dae(self, eq_name):
        """
        Helper function for collecting equation values into dae structures `f` and `g`

        Parameters
        ----------
        eq_name

        Returns
        -------

        """
        if eq_name not in ('f', 'g'):
            raise KeyError(f'{eq_name} is not a valid eq name')

        for var in self.__dict__[f'{eq_name}_adders']:
            if var.n > 0:
                np.add.at(self.dae.__dict__[eq_name], var.a, var.e)
        for var in self.__dict__[f'{eq_name}_setters']:
            if var.n > 0:
                np.put(self.dae.__dict__[eq_name], var.a, var.e)

    @staticmethod
    def get_pkl_path():
        pkl_name = 'calls.pkl'
        andes_path = os.path.join(str(pathlib.Path.home()), '.andes')

        if not os.path.exists(andes_path):
            os.makedirs(andes_path)

        pkl_path = os.path.join(andes_path, pkl_name)

        return pkl_path

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

    def dill_calls(self):
        logger.debug("Dumping calls to andes.pkl with dill")
        import dill
        dill.settings['recurse'] = True

        pkl_path = self.get_pkl_path()
        dill.dump(self.calls, open(pkl_path, 'wb'))

    def undill_calls(self):
        import dill
        dill.settings['recurse'] = True

        pkl_path = self.get_pkl_path()

        if not os.path.isfile(pkl_path):
            self.prepare()

        self.calls = dill.load(open(pkl_path, 'rb'))
        logger.debug(f'System undill: loaded <{pkl_path}> file.')
        for name, model_call in self.calls.items():
            self.__dict__[name].calls = model_call

    def _get_models(self, models):
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
        # do nothing for OrderedDict type
        return models

    def _call_models_method(self, method: str, models: Optional[Union[str, list, Model, OrderedDict]]):
        if not isinstance(models, OrderedDict):
            models = self._get_models(models)
        for mdl in models.values():
            getattr(mdl, method)()

    def _check_group_common(self):
        """
        Check if all group common variables and parameters are met

        Raises
        ------
        KeyError if any parameter or value is not provided

        Returns
        -------
        None
        """
        for group in self.groups.values():
            for item in group.common_params:
                for model in group.models.values():
                    # the below includes all of BaseParam (NumParam, DataParam and ExtParam)
                    if item not in model.__dict__ or not isinstance(model.__dict__[item], BaseParam):
                        raise KeyError(f'Group <{group.class_name}> common param <{item}> does not exist '
                                       f'in model <{model.class_name}>')
            for item in group.common_vars:
                for model in group.models.values():
                    if item not in model.cache.all_vars:
                        raise KeyError(f'Group <{group.class_name}> common param <{item}> does not exist '
                                       f'in model <{model.class_name}>')

    def _collect_ref_param(self):
        """
        Collect indices into `RefParam` for all models

        Returns
        -------

        """
        # FIXME: too many safe-checking here. Even the model can be non-existent.

        for model in self.models.values():
            for ref in model.ref_params.values():
                ref.v = [list() for _ in range(model.n)]

        for model in self.models.values():
            if model.n == 0:
                continue

            for ref in model.idx_params.values():
                if ref.model not in self.models:
                    continue
                dest_model = self.__dict__[ref.model]

                if dest_model.n == 0:
                    continue

                for n in (model.class_name, model.group):
                    if n not in dest_model.ref_params:
                        continue

                    for model_idx, dest_idx in zip(model.idx, ref.v):
                        if dest_idx not in dest_model.idx:
                            continue
                        uid = dest_model.idx2uid(dest_idx)
                        dest_model.ref_params[n].v[uid].append(model_idx)

    def _generate_initializers(self):
        # TODO: consider both JIT and non-JIT models
        self._call_models_method('generate_initializers', self.models)

    def _generate_symbols(self):
        self._call_models_method('generate_symbols', self.models)

    def _generate_pretty_print(self):
        self._call_models_method('generate_pretty_print', self.models)

    def _generate_equations(self):
        self._call_models_method('generate_equations', self.models)

    def _generate_jacobians(self):
        self._call_models_method('generate_jacobians', self.models)

    def _group_import(self):
        """
        Import groups defined in `devices/group.py`

        Returns
        -------

        """
        module = importlib.import_module('andes.models.group')
        for m in inspect.getmembers(module, inspect.isclass):
            name = m[0]
            cls = m[1]
            if name == 'GroupBase':
                continue
            self.__dict__[name] = cls()
            self.groups[name] = self.__dict__[name]

    def _model_import(self):
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
                the_module = importlib.import_module('andes.models.' + file)
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

    def _routine_import(self):
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
        for file, cls_list in all_routines.items():
            for cls_name in cls_list:
                file = importlib.import_module('andes.routines.' + file)
                the_class = getattr(file, cls_name)
                attr_name = cls_name
                self.__dict__[attr_name] = the_class(system=self, config=self._config_from_file)
                self.programs[attr_name] = self.__dict__[attr_name]

    def store_switch_times(self, models=None):
        models = self._get_models(models)
        out = []
        for instance in models.values():
            out.extend(instance.get_times())

        out = np.ravel(np.array(out))
        out = np.unique(out)
        out = out[np.where(out >= 0)]
        out = np.sort(out)

        self.switch_times = out
        return self.switch_times

    def switch_action(self, models=None):
        models = self._get_models(models)
        for instance in models.values():
            instance.switch_action(self.dae.t)

    def _p_restore(self):
        """
        Restore parameters stored in `pin`
        Returns
        -------

        """
        for model in self.models.values():
            for param in model.num_params.values():
                param.restore()

    def e_clear(self, models: Optional[Union[str, List, OrderedDict]] = None):
        self.dae.clear_fg()
        self._call_models_method('e_clear', models)

    def _store_calls(self):
        logger.debug("Collecting Model.calls into System.")
        for name, mdl in self.models.items():
            self.calls[name] = mdl.calls

    def _list2array(self):
        self._call_models_method('list2array', self.models)

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

        all_with_config = OrderedDict(list(self.programs.items()) +
                                      list(self.models.items()))

        for name, instance in all_with_config.items():
            cfg = instance.config.as_dict()
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
        logger.debug(f'Config: Loaded from file <{conf_path}>.')
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

        elif os.path.isfile(file_path):
            choice = input(f'Config file {file_path} already exist. Overwrite? [y/N]').lower()
            if len(choice) == 0 or choice[0] != 'y':
                logger.info('No config file overwritten.')
                return

        conf = self.get_config()
        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info(f'Config: written to {file_path}')
