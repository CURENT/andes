# ANDES, a power system simulation tool for research.
#
# Copyright 2015-2017 Hantao Cui
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
Base class for building ANDES models
"""
from collections import OrderedDict

import importlib
import logging
import numpy as np

from andes.common.config import Config
from andes.core.discrete import Discrete
from andes.core.param import ParamBase, RefParam, IdxParam, DataParam, NumParam, ExtParam, TimerParam
from andes.core.var import VarBase, Algeb, State, Calc, ExtAlgeb, ExtState
from andes.core.block import Block
from andes.core.service import ServiceBase, ServiceConst, ExtService, ServiceOperation, ServiceRandom
from andes.common.utils import list_flatten
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)
pd = None


def load_pd():
    """
    Import pandas to globals() if not exist
    """
    if globals()['pd'] is None:
        try:
            globals()['pd'] = importlib.import_module('pandas')
        except ImportError:
            logger.warning("Pandas import error.")
            return False

    return True


class Cache(object):
    """
    Class for caching the return value of callback functions
    """

    def __init__(self):
        self._callbacks = {}

    def __getattr__(self, item):
        if item not in self.__dict__:
            if item in self._callbacks:
                self.__dict__[item] = self._call(item)

        return self.__getattribute__(item)

    def __setattr__(self, key, value):
        super(Cache, self).__setattr__(key, value)

    def add_callback(self, name: str, callback):
        """
        Add a cache attribute and a callback function to update the attribute

        Parameters
        ----------
        name : str
            name of the cached function return value
        callback : callable
            callback function for updating the cached attribute
        """
        self._callbacks[name] = callback

    def refresh(self, name=None):
        """
        Refresh the cached values

        Parameters
        ----------
        name : str, list, optional
            name or list of cached to refresh, by default None for refreshing all
        """
        if name is None:
            for name in self._callbacks.keys():
                self.__dict__[name] = self._call(name)
        elif isinstance(name, str):
            self.__dict__[name] = self._call(name)
        elif isinstance(name, list):
            for n in name:
                self.__dict__[n] = self._call(n)

    def _call(self, name):
        if name not in self._callbacks:
            return None
        else:
            if callable(self._callbacks[name]):
                return self._callbacks[name]()
            else:
                return self._callbacks[name]


class ModelData(object):
    """
    Class for holding model data.

    This class is used when accessing
    data file and manipulating with the raw data without having
    the `System` class constructed.
    """

    def __init__(self, *args, **kwargs):
        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.ref_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.n = 0
        self.idx = []
        self.uid = {}

        if not hasattr(self, 'cache'):
            self.cache = Cache()
        self.cache.add_callback('dict', self.as_dict)
        self.cache.add_callback('df', self.as_df)
        self.cache.add_callback('df_in', self.as_df_in)

        self.u = NumParam(default=1, info='connection status', unit='bool', tex_name='u')
        self.name = DataParam(info='element name', tex_name='name')

    def __setattr__(self, key, value):
        if isinstance(value, ParamBase):
            value.owner = self
            if not value.name:
                value.name = key

            if key in self.__dict__:
                logger.warning(f"{self.__class__}: redefinition of instance member <{key}>")

            self.params[key] = value

        if isinstance(value, NumParam):
            self.num_params[key] = value
        elif isinstance(value, RefParam):
            self.ref_params[key] = value
        elif isinstance(value, IdxParam):
            self.idx_params[key] = value

        # `TimerParam` is a subclass of `NumParam` and thus tested separately
        if isinstance(value, TimerParam):
            self.timer_params[key] = value

        super(ModelData, self).__setattr__(key, value)

    def add(self, idx=None, **kwargs):
        """
        Add a model element using a set of parameters.

        Parameters
        ----------
        idx : str, optional
            reference-able external index, by default None
        """
        self.idx.append(idx)
        self.uid[idx] = self.n
        self.n += 1

        for name, instance in self.params.items():

            # skip `RefParam` because it is collected outside `add`
            if isinstance(instance, RefParam):
                continue
            value = kwargs.pop(name, None)
            instance.add(value)
        if len(kwargs) > 0:
            logger.warning(f'{self.__class__.__name__}: Unused data {kwargs}')

    def as_dict(self, vin=False):
        """
        Export all variable parameters as a dict


        Returns
        -------
        dict
            a dict with the keys being the `ModelData` parameter names
            and the values being an array-like of data in the order of adding.
            An additional `uid` key is added with the value default to range(n).
        """
        out = dict()
        out['uid'] = np.arange(self.n)
        out['idx'] = self.idx

        for name, instance in self.params.items():
            # skip non-exported parameters
            if instance.export is False:
                continue

            # select origin input if `vin` is True
            if vin is True and hasattr(instance, 'vin'):
                out[name] = instance.vin
            else:
                out[name] = instance.v

        return out

    def as_df(self):
        """
        Export all the data as a `pandas.DataFrame`
        This function utilizes `as_dict` for preparing data.


        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        if not load_pd():
            return None

        out = pd.DataFrame(self.as_dict()).set_index('uid')

        return out

    def as_df_in(self):
        """
        Export all the data from original input (``vin``) as a `pandas.DataFrame`.
        This function utilizes `as_dict` for preparing data.

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        if not load_pd():
            return None

        out = pd.DataFrame(self.as_dict(vin=True)).set_index('uid')

        return out

    def find_param(self, prop):
        """
        Find an OrderedDict of params with the given property

        Parameters
        ----------
        prop : str
            Property name

        Returns
        -------
        OrderedDict
        """
        out = OrderedDict()
        for name, instance in self.params.items():
            if instance.get_property(prop) is True:
                out[name] = instance

        return out


class ModelCall(object):
    def __init__(self):
        # callables to be generated by sympy.lambdify
        self.g_lambdify = None
        self.f_lambdify = None
        self.h_lambdify = None
        self.init_lambdify = None
        self.service_lambdify = None

        self._ifx, self._jfx, self._vfx = list(), list(), list()
        self._ify, self._jfy, self._vfy = list(), list(), list()
        self._igx, self._jgx, self._vgx = list(), list(), list()
        self._igy, self._jgy, self._vgy = list(), list(), list()
        self._itx, self._jtx, self._vtx = list(), list(), list()
        self._irx, self._jrx, self._vrx = list(), list(), list()

        self._ifxc, self._jfxc, self._vfxc = list(), list(), list()
        self._ifyc, self._jfyc, self._vfyc = list(), list(), list()
        self._igxc, self._jgxc, self._vgxc = list(), list(), list()
        self._igyc, self._jgyc, self._vgyc = list(), list(), list()
        self._itxc, self._jtxc, self._vtxc = list(), list(), list()
        self._irxc, self._jrxc, self._vrxc = list(), list(), list()


class Model(object):
    """
    Base class for power system device models


    Attributes
    ----------
    n : int
        The number of loaded elements.

    idx : list
        A list of all element idx.

    num_params : OrderedDict
        {name: instance} of numerical parameters, including internal
        and external onesd
    """

    def __init__(self, system=None, config=None):
        self.system = system

        # duplicate attributes from ModelData. Keep for now.
        self.n = 0
        self.idx = []
        self.group = 'Undefined'

        if not hasattr(self, 'num_params'):
            self.num_params = OrderedDict()
        if not hasattr(self, 'cache'):
            self.cache = Cache()

        # variables
        self.states = OrderedDict()  # internal states
        self.states_ext = OrderedDict()  # external states
        self.algebs = OrderedDict()  # internal algebraic variables
        self.algebs_ext = OrderedDict()  # external algebraic vars
        self.calcs = OrderedDict()   # internal calculated vars
        self.vars_decl_order = OrderedDict()

        # external parameters
        self.params_ext = OrderedDict()

        # discrete and control blocks
        self.discrete = OrderedDict()
        self.blocks = OrderedDict()

        # service/temporary variables
        self.services = OrderedDict()
        self.services_ext = OrderedDict()
        self.services_ops = OrderedDict()

        # cache callback and lambda function storage
        self.calls = ModelCall()

        self.flags = dict(
            pflow=False,
            tds=False,  # if `tds` is False, `dae_t` cannot be used
            sys_base=False,
            address=False,
            collate=True,
            is_series=False,
        )

        self.config = Config()
        if config is not None:
            self.set_config(config)

        self.tex_names = OrderedDict((('dae_t', 't_{dae}'), ))
        self.input_syms = OrderedDict()
        self.vars_syms = OrderedDict()
        self.f_syms, self.g_syms, self.h_syms = list(), list(), list()
        self.f_syms_matrix, self.g_syms_matrix, self.c_syms_matrix = list(), list(), list()
        self.f_print, self.g_print, self.c_print = list(), list(), list()
        self.df_print, self.dg_print = None, None

        # ----- ONLY FOR CUSTOM NUMERICAL JACOBIAN FUNCTIONS -----
        self._ifx, self._jfx, self._vfx = list(), list(), list()
        self._ify, self._jfy, self._vfy = list(), list(), list()
        self._igx, self._jgx, self._vgx = list(), list(), list()
        self._igy, self._jgy, self._vgy = list(), list(), list()
        self._itx, self._jtx, self._vtx = list(), list(), list()
        self._irx, self._jrx, self._vrx = list(), list(), list()

        self._ifxc, self._jfxc, self._vfxc = list(), list(), list()
        self._ifyc, self._jfyc, self._vfyc = list(), list(), list()
        self._igxc, self._jgxc, self._vgxc = list(), list(), list()
        self._igyc, self._jgyc, self._vgyc = list(), list(), list()
        self._itxc, self._jtxc, self._vtxc = list(), list(), list()
        self._irxc, self._jrxc, self._vrxc = list(), list(), list()
        # -------------------------------------------------------

        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()

        self.ifxc, self.jfxc, self.vfxc = list(), list(), list()
        self.ifyc, self.jfyc, self.vfyc = list(), list(), list()
        self.igxc, self.jgxc, self.vgxc = list(), list(), list()
        self.igyc, self.jgyc, self.vgyc = list(), list(), list()

        # cached class attributes
        self.cache.add_callback('all_vars', self._all_vars)
        self.cache.add_callback('all_vars_names', self._all_vars_names)
        self.cache.add_callback('all_params', self._all_params)
        self.cache.add_callback('all_params_names', self._all_params_names)
        self.cache.add_callback('algebs_and_ext', self._algebs_and_ext)
        self.cache.add_callback('states_and_ext', self._states_and_ext)
        self.cache.add_callback('services_and_ext', self._services_and_ext)
        self.cache.add_callback('vars_ext', self._vars_ext)

        # cached dictionary of inputs
        self._input = OrderedDict()

    def __setattr__(self, key, value):
        if isinstance(value, (VarBase, ServiceBase, Discrete, Block)):
            if not value.owner:
                value.owner = self
            if not value.name:
                value.name = key
            if not value.tex_name:
                value.tex_name = key
            if key in self.__dict__:
                logger.warning(f"{self.class_name}: redefinition of member <{key}>")

        # store the variable declaration order
        if isinstance(value, VarBase):
            value.id = len(self._all_vars())  # NOT in use yet
            self.vars_decl_order[key] = value

        # store instances to the corresponding OrderedDict
        if isinstance(value, Algeb):
            self.algebs[key] = value
        elif isinstance(value, ExtAlgeb):
            self.algebs_ext[key] = value
        elif isinstance(value, State):
            self.states[key] = value
        elif isinstance(value, ExtState):
            self.states_ext[key] = value
        elif isinstance(value, Calc):
            self.calcs[key] = value
        elif isinstance(value, ExtParam):
            self.params_ext[key] = value
        elif isinstance(value, Discrete):
            self.discrete[key] = value
        elif isinstance(value, ServiceConst):
            # only services with `v_str`
            self.services[key] = value
        elif isinstance(value, ExtService):
            self.services_ext[key] = value
        elif isinstance(value, (ServiceOperation, ServiceRandom)):
            self.services_ops[key] = value
        elif isinstance(value, Block):
            self.blocks[key] = value
            # pull in sub-variables from control blocks
            for var_name, var_instance in value.export().items():
                var_instance.name = f'{value.name}_{var_name}'
                var_instance.tex_name = rf'{value.tex_name}\ {var_instance.tex_name}'
                self.__setattr__(var_instance.name, var_instance)

        super(Model, self).__setattr__(key, value)

    def idx2uid(self, idx):
        if idx is None:
            logger.error("idx2uid cannot search for None idx")
            return None
        if isinstance(idx, (float, int, str, np.int64, np.float64)):
            return self.uid[idx]
        elif isinstance(idx, (list, np.ndarray)):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self.uid[i] for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def get(self, src: str, idx, attr: str):
        uid = self.idx2uid(idx)
        return self.__dict__[src].__dict__[attr][uid]

    def set(self, src, idx, attr, value):
        uid = self.idx2uid(idx)
        self.__dict__[src].__dict__[attr][uid] = value

    def get_inputs(self, refresh=False):
        if len(self._input) == 0 or refresh:
            self.refresh_inputs()

        # update`dae_t`
        self._input['dae_t'] = self.system.dae.t
        return self._input

    def refresh_inputs(self):
        # The order of inputs: `all_params` and then `all_vars`, finally `config`
        # the below sequence should correspond to `self.all_param_names`
        for instance in self.num_params.values():
            self._input[instance.name] = instance.v

        for instance in self.services.values():
            self._input[instance.name] = instance.v

        for instance in self.services_ext.values():
            self._input[instance.name] = instance.v

        # updated every call
        for instance in self.discrete.values():
            for name, val in zip(instance.get_names(), instance.get_values()):
                self._input[name] = val

        # append all variable values
        for instance in self.cache.all_vars.values():
            self._input[instance.name] = instance.v

        # append config variables
        for key, val in self.config.__dict__.items():
            self._input[key] = val

    def l_update_var(self):
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_var()
            instance.set_var()
        self.refresh_inputs()

    def l_update_eq(self):
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_eq()
            instance.set_eq()
        self.refresh_inputs()

    def s_update(self):
        """
        Evaluate service equations

        This function is only evaluated at initialization, sequentially.
        Returns
        -------

        """
        if self.n == 0:
            return
        logger.debug(f'{self.class_name}: calling eval_service()')

        if self.calls.service_lambdify is not None and len(self.calls.service_lambdify):
            for name, instance in self.services.items():

                func = self.calls.service_lambdify[name]
                if callable(func):
                    kwargs = self.get_inputs(refresh=True)
                    instance.v = func(**kwargs)
                else:
                    instance.v = func

        # NOTE: some numerical calls depend on other service values, so they are evaluated
        #       after lambdified calls

        # Apply both the individual `v_numeric` and Model-level `s_numeric`
        for instance in self.services.values():
            func = instance.v_numeric
            if func is not None and callable(func):
                kwargs = self.get_inputs(refresh=True)
                instance.v = func(**kwargs)

        kwargs = self.get_inputs(refresh=True)
        self.s_numeric(**kwargs)

    def generate_initializers(self):
        """
        Generate lambda functions for initial values
        """
        from sympy import sympify, lambdify
        from sympy.printing import latex

        syms_list = list(self.input_syms)

        init_lambda_list = OrderedDict()
        init_latex = OrderedDict()
        for name, instance in self.cache.all_vars.items():
            if instance.v_init is None:
                init_lambda_list[name] = 0
                init_latex[name] = ''
            else:
                sympified = sympify(instance.v_init, locals=self.input_syms)
                lambdified = lambdify(syms_list, sympified, 'numpy')
                init_lambda_list[name] = lambdified
                init_latex[name] = latex(sympified.subs(self.tex_names))

        self.calls.init_lambdify = init_lambda_list
        self.calls.init_latex = init_latex

    def generate_equations(self):
        logger.debug(f'Generating equations for {self.__class__.__name__}')
        from sympy import Symbol, Matrix, sympify, lambdify
        from sympy.printing import latex

        self.f_syms, self.g_syms, self.h_syms = list(), list(), list()
        self.f_syms_matrix, self.g_syms_matrix, self.c_syms_matrix = list(), list(), list()
        self.f_print, self.g_print, self.c_print = list(), list(), list()

        # -----------------------------------------------------------
        # process tex_names defined in model
        for key in self.tex_names.keys():
            self.tex_names[key] = Symbol(self.tex_names[key])

        for var in self.cache.all_params_names:
            self.input_syms[var] = Symbol(var)

            # replace symbols for printing
            if var in self.__dict__ and self.__dict__[var].tex_name is not None:
                self.tex_names[Symbol(var)] = Symbol(self.__dict__[var].tex_name)

        for var in self.cache.all_vars_names:
            self.vars_syms[var] = Symbol(var)
            self.input_syms[var] = Symbol(var)

            # replace symbols for printing
            if var in self.__dict__ and self.__dict__[var].tex_name is not None:
                self.tex_names[Symbol(var)] = Symbol(self.__dict__[var].tex_name)

        for key in self.config.__dict__:
            self.input_syms[key] = Symbol(key)

        self.input_syms['dae_t'] = Symbol('dae_t')
        self.vars_print = Matrix(list(self.vars_syms.values())).subs(self.tex_names)
        # ------------------------------------------------------------

        syms_list = list(self.input_syms)
        iter_list = [self.states, self.states_ext, self.algebs, self.algebs_ext, self.calcs]
        dest_list = [self.f_syms, self.f_syms, self.g_syms, self.g_syms, self.h_syms]

        for it, dest in zip(iter_list, dest_list):
            for instance in it.values():
                if instance.e_str is None:
                    dest.append(0)
                else:
                    try:
                        sympified_equation = sympify(instance.e_str, locals=self.input_syms)
                    except TypeError as e:
                        logger.error(f'Error sympifying equation <{instance.e_str}> for <{instance.name}>')
                        raise e

                    dest.append(sympified_equation)

        self.f_syms_matrix = Matrix(self.f_syms)
        self.g_syms_matrix = Matrix(self.g_syms)
        self.h_syms_matrix = Matrix(self.h_syms)

        self.f_print = self.f_syms_matrix.subs(self.tex_names)
        self.g_print = self.g_syms_matrix.subs(self.tex_names)
        self.h_print = self.h_syms_matrix.subs(self.tex_names)

        self.calls.g_lambdify = lambdify(syms_list, self.g_syms_matrix, 'numpy')
        self.calls.f_lambdify = lambdify(syms_list, self.f_syms_matrix, 'numpy')
        self.calls.h_lambdify = lambdify(syms_list, self.h_syms_matrix, 'numpy')

        # store latex strings
        nx = len(self.f_print)
        ny = len(self.g_print)
        self.calls.x_latex = [latex(item) for item in self.vars_print[:nx]]
        self.calls.y_latex = [latex(item) for item in self.vars_print[nx:nx + ny]]
        self.calls.c_latex = [latex(item) for item in self.vars_print[nx + ny:]]

        self.calls.f_latex = [latex(item) for item in self.f_print]
        self.calls.g_latex = [latex(item) for item in self.g_print]
        self.calls.h_latex = [latex(item) for item in self.h_print]

        # convert service equations
        # Note: service equations are converted one by one, because service variables
        # can be interdependent
        service_eq_list = OrderedDict()
        service_latex = OrderedDict()
        for name, instance in self.services.items():
            if instance.v_str is not None:
                sympified_equation = sympify(instance.v_str, locals=self.input_syms)
                service_eq_list[name] = lambdify(syms_list, sympified_equation, 'numpy')
                service_latex[name] = latex(sympified_equation.subs(self.tex_names))
            else:
                service_eq_list[name] = 0
                service_latex[name] = '0'

        self.calls.service_lambdify = service_eq_list
        self.calls.service_latex = service_latex

    def generate_jacobians(self):
        logger.debug(f'Generating Jacobians for {self.__class__.__name__}')
        from sympy import SparseMatrix, lambdify, Matrix

        self.calls._ifx, self.calls._jfx, self.calls._vfx = list(), list(), list()
        self.calls._ify, self.calls._jfy, self.calls._vfy = list(), list(), list()
        self.calls._igx, self.calls._jgx, self.calls._vgx = list(), list(), list()
        self.calls._igy, self.calls._jgy, self.calls._vgy = list(), list(), list()

        self.calls._ifxc, self.calls._jfxc, self.calls._vfxc = list(), list(), list()
        self.calls._ifyc, self.calls._jfyc, self.calls._vfyc = list(), list(), list()
        self.calls._igxc, self.calls._jgxc, self.calls._vgxc = list(), list(), list()
        self.calls._igyc, self.calls._jgyc, self.calls._vgyc = list(), list(), list()

        # NOTE: SymPy does not allow getting the derivative of an empty array
        self.dg_syms = Matrix([])
        self.df_syms = Matrix([])
        if len(self.g_syms_matrix) > 0:
            self.dg_syms = self.g_syms_matrix.jacobian(list(self.vars_syms.values()))

        if len(self.f_syms_matrix) > 0:
            self.df_syms = self.f_syms_matrix.jacobian(list(self.vars_syms.values()))

        self.dg_syms_sparse = SparseMatrix(self.dg_syms)
        self.df_syms_sparse = SparseMatrix(self.df_syms)

        self.df_print = self.df_syms_sparse.subs(self.tex_names)
        self.dg_print = self.dg_syms_sparse.subs(self.tex_names)

        vars_syms_list = list(self.vars_syms)
        syms_list = list(self.input_syms)
        algebs_and_ext_list = list(self.cache.algebs_and_ext)
        states_and_ext_list = list(self.cache.states_and_ext)

        fg_sparse = [self.df_syms_sparse, self.dg_syms_sparse]
        for idx, eq_sparse in enumerate(fg_sparse):
            for item in eq_sparse.row_list():
                e_idx = item[0]
                v_idx = item[1]
                e_symbolic = item[2]

                if idx == 0:
                    eq_name = states_and_ext_list[e_idx]
                else:
                    eq_name = algebs_and_ext_list[e_idx]

                var_name = vars_syms_list[v_idx]
                eqn = self.cache.all_vars[eq_name]
                var = self.cache.all_vars[var_name]

                # FIXME: this line takes excessively long to run
                # it will work with is_number
                # but needs some refactor with the value
                #
                # if e_str.is_constant():
                #     jac_name = f'{eqn.e_code}{var.v_code}c'

                # -----------------------------------------------------------------
                # ------ Constant parameters are not known at generation time -----
                # elif len(e_str.atoms(Symbol)) == 1 and \
                #         str(list(e_str.atoms(Symbol))[0]) in self.cache.all_consts_names:
                #     jac_name = f'{eqn.e_code}{var.v_code}c'
                # -----------------------------------------------------------------
                # else:
                jac_name = f'{eqn.e_code}{var.v_code}'

                self.calls.__dict__[f'_i{jac_name}'].append(e_idx)
                self.calls.__dict__[f'_j{jac_name}'].append(v_idx)
                self.calls.__dict__[f'_v{jac_name}'].append(lambdify(syms_list, e_symbolic))

        # NOTE: This will not work: checking if the jacobian
        # element has value and add an epsilon if not.
        # The reason is that even if the jacobian is non-zero at
        # generation time, it can evaluate to zero

        # NOTE:
        # The for loop below is intended to add an epsilon small
        # value to the diagonal of gy matrix.
        # The user should take care of the algebraic equations
        # by using `diag_eps` in Algeb definition

        for var in self.algebs.values():
            if var.diag_eps == 0.0:
                continue
            e_idx = algebs_and_ext_list.index(var.name)
            v_idx = vars_syms_list.index(var.name)
            self.calls.__dict__[f'_igyc'].append(e_idx)
            self.calls.__dict__[f'_jgyc'].append(v_idx)
            self.calls.__dict__[f'_vgyc'].append(var.diag_eps)

    def store_sparse_pattern(self):
        """

        Calling sequence:
        For each jacobian name, `fx`, `fy`, `gx` and `gy`,
        store by
        a) generated constant and variable jacobians
        c) user-provided constant and variable jacobians,
        d) user-provided block constant and variable jacobians
        """

        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()
        self.itx, self.jtx, self.vtx = list(), list(), list()
        self.irx, self.jrx, self.vrx = list(), list(), list()

        self.ifxc, self.jfxc, self.vfxc = list(), list(), list()
        self.ifyc, self.jfyc, self.vfyc = list(), list(), list()
        self.igxc, self.jgxc, self.vgxc = list(), list(), list()
        self.igyc, self.jgyc, self.vgyc = list(), list(), list()
        self.itxc, self.jtxc, self.vtxc = list(), list(), list()
        self.irxc, self.jrxc, self.vrxc = list(), list(), list()

        if (not self.flags['address']) or (self.n == 0):
            # Note:
            # if `self.n` is 0, skipping the processes below will
            # avoid appending empty lists/arrays and non-empty values,
            # which as a combination is not accepted by `cvxopt.spmatrix`
            #
            # If we don't want to check `self.n`, we can check if
            # len(row) == 0 or len(col) ==0 in the code below.
            return

        self.j_numeric()
        # store block jacobians to block instances
        for instance in self.blocks.values():
            instance.j_numeric()

        var_names_list = list(self.cache.all_vars.keys())
        eq_names = {'f': var_names_list[:len(self.cache.states_and_ext)],
                    'g': var_names_list[len(self.cache.states_and_ext):]}

        for j_name in self.system.dae.jac_name:
            for j_type in self.system.dae.jac_type:
                # generated lambda functions
                for row, col, val in zip(self.calls.__dict__[f'_i{j_name}{j_type}'],
                                         self.calls.__dict__[f'_j{j_name}{j_type}'],
                                         self.calls.__dict__[f'_v{j_name}{j_type}']):
                    row_name = eq_names[j_name[0]][row]  # separate states and algebs
                    col_name = var_names_list[col]

                    row_idx = self.__dict__[row_name].a
                    col_idx = self.__dict__[col_name].a
                    if len(row_idx) != len(col_idx):
                        print(f'Model: {self.class_name}')
                        print(f'row {row_name}, row_idx: {row_idx}')
                        print(f'col {col_name}, col_idx: {col_idx}')
                        raise ValueError
                    elif len(row_idx) == 0 and len(col_idx) == 0:
                        continue

                    self.__dict__[f'i{j_name}{j_type}'].append(row_idx)
                    self.__dict__[f'j{j_name}{j_type}'].append(col_idx)
                    if j_type == 'c':
                        self.__dict__[f'v{j_name}{j_type}'].append(val)
                    else:
                        self.__dict__[f'v{j_name}{j_type}'].append(np.zeros(self.n))

                # user-provided numerical jacobians
                for row, col, val in zip(self.__dict__[f'_i{j_name}{j_type}'],
                                         self.__dict__[f'_j{j_name}{j_type}'],
                                         self.__dict__[f'_v{j_name}{j_type}']):

                    if len(row) != len(col):
                        raise ValueError
                    elif len(row) == 0 and len(col) == 0:
                        continue

                    self.__dict__[f'i{j_name}{j_type}'].append(row)
                    self.__dict__[f'j{j_name}{j_type}'].append(col)

                    if j_type == 'c':
                        self.__dict__[f'v{j_name}{j_type}'].append(val)
                    else:
                        self.__dict__[f'v{j_name}{j_type}'].append(np.zeros(self.n))

                # user-provided numerical jacobians in blocks
                for instance in list(self.blocks.values()):
                    for row, col, val in zip(instance.__dict__[f'i{j_name}{j_type}'],
                                             instance.__dict__[f'j{j_name}{j_type}'],
                                             instance.__dict__[f'v{j_name}{j_type}']):
                        self.__dict__[f'i{j_name}{j_type}'].append(row)
                        self.__dict__[f'j{j_name}{j_type}'].append(col)

                        if j_type == 'c':
                            self.__dict__[f'v{j_name}{j_type}'].append(val)
                        else:
                            self.__dict__[f'v{j_name}{j_type}'].append(np.zeros(self.n))

    def initialize(self):
        if self.n == 0:
            return

        # update service values
        self.s_update()

        logger.debug(f'{self.class_name}: calling initialize()')
        for name, instance in self.vars_decl_order.items():
            if instance.v_init is None:
                continue

            kwargs = self.get_inputs(refresh=True)
            init_fun = self.calls.init_lambdify[name]
            if callable(init_fun):
                try:
                    instance.v = init_fun(**kwargs)
                except TypeError:
                    logger.error(f'{self.class_name}: {instance.name} = {instance.v_init} error.'
                                 f'You might have undefined variable in the equation string.')
            else:
                instance.v = init_fun

        # call custom variable initializer after lambdified initializers
        kwargs = self.get_inputs(refresh=True)
        self.v_numeric(**kwargs)

    def f_update(self):
        if self.n == 0:
            return

        # update equations for algebraic variables supplied with `g_numeric`
        # evaluate numerical function calls
        kwargs = self.get_inputs()

        # call lambdified functions with use self.call
        ret = self.calls.f_lambdify(**kwargs)

        for idx, instance in enumerate(self.cache.states_and_ext.values()):
            try:
                instance.e += ret[idx][0]
            except TypeError as e:
                logger.error(f"Error evaluating f for {instance.name} where e_str={instance.e_str}")
                raise e

        # numerical calls defined in the model
        self.f_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.f_numeric(**kwargs)

    def g_update(self):
        if self.n == 0:
            return

        logger.debug(f'{self.class_name}: calling g_update()')
        # update equations for algebraic variables supplied with `g_numeric`
        # evaluate numerical function calls
        kwargs = self.get_inputs()

        # call lambdified functions with use self.call
        ret = self.calls.g_lambdify(**kwargs)

        for idx, instance in enumerate(self.cache.algebs_and_ext.values()):
            try:
                instance.e += ret[idx][0]
            except TypeError as e:
                logger.error(f"Error evaluating g for {instance.name} where e_str={instance.e_str}")
                raise e

        # numerical calls defined in the model
        self.g_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.g_numeric(**kwargs)

    def c_update(self):
        if self.n == 0:
            return
        kwargs = self.get_inputs(refresh=True)
        # call lambdified functions with use self.call
        ret = self.calls.h_lambdify(**kwargs)
        for idx, instance in enumerate(self.calcs.values()):
            instance.v = ret[idx][0]

        # numerical calls defined in the model
        self.c_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.c_numeric(**kwargs)

    def j_update(self):
        if self.n == 0:
            return

        jac_set = ('fx', 'fy', 'gx', 'gy')

        kwargs = self.get_inputs()
        for name in jac_set:
            idx = 0

            # generated lambda jacobian functions first
            fun_list = self.calls.__dict__[f'_v{name}']
            for fun in fun_list:
                self.__dict__[f'v{name}'][idx] = fun(**kwargs)
                idx += 1

            # call numerical jacobian functions for blocks
            for instance in self.blocks.values():
                for fun in instance.__dict__[f'v{name}']:
                    self.__dict__[f'v{name}'][idx] = fun(**kwargs)
                    idx += 1

            # call numerical jacobians for self
            for fun in self.__dict__[f'_v{name}']:
                self.__dict__[f'v{name}'][idx] = fun(**kwargs)
                idx += 1

    def get_times(self):
        """
        Get event switch_times from `TimerParam`

        Returns
        -------
        list
            A list containing all switching times defined in TimerParams
        """
        out = []
        if self.n > 0:
            for name, instance in self.timer_params.items():
                out.append(instance.v)

        return out

    def switch_action(self, dae_t):
        if self.n == 0:
            return

        action = False
        for timer in self.timer_params.values():
            if timer.callback is not None:
                ret = timer.callback(timer.is_time(dae_t))
                if ret is None or (ret is True):
                    action = True

        # TODO: consider `Block` with timer
        if action:
            self.refresh_inputs()

    @property
    def class_name(self):
        """Return class name"""
        return self.__class__.__name__

    def _all_vars(self):
        """An OrderedDict of States, ExtStates, Algebs, ExtAlgebs, and Calcs"""
        return OrderedDict(list(self.states.items()) +
                           list(self.states_ext.items()) +
                           list(self.algebs.items()) +
                           list(self.algebs_ext.items()) +
                           list(self.calcs.items())
                           )

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

    def row_of(self, name):
        """
        Return the row index in a flattened arrays for the specified jacobian matrix

        Parameters
        ----------
        name : str
            name of the jacobian matrix
        """
        return np.ravel(np.array(self.__dict__[f'i{name}']))

    def col_of(self, name):
        """
        Return the col index in a flattened arrays for the specified jacobian matrix

        Parameters
        ----------
        name : str
            name of the jacobian matrix
        """
        return np.ravel(np.array(self.__dict__[f'j{name}']))

    def zip_ijv(self, name):
        """
        Return a zipped iterator for the rows, cols and vals for the specified matrix

        Parameters
        ----------
        name : str
            jac name
        """
        return zip(self.__dict__[f'i{name}'],
                   self.__dict__[f'j{name}'],
                   self.__dict__[f'v{name}'])

    def set_config(self, config):
        if self.class_name in config:
            config_section = config[self.class_name]
            self.config.add(OrderedDict(config_section))

    def get_config(self):
        return self.config.as_dict()

    def list2array(self):
        for _, instance in self.num_params.items():
            instance.to_array()

    def a_reset(self):
        """
        Reset addresses to empty and reset flags['address']
        Returns
        -------

        """
        if self.n == 0:
            return
        for var in self.cache.all_vars.values():
            var.reset()
        self.flags['address'] = False

    def e_clear(self):
        if self.n == 0:
            return
        for instance in self.cache.all_vars.values():
            instance.e = np.zeros(instance.n)

    def v_numeric(self, **kwargs):
        """
        Custom variable initialization function
        """
        pass

    def g_numeric(self, **kwargs):
        """
        Custom gcall functions. Modify equations directly
        """
        pass

    def f_numeric(self, **kwargs):
        """
        Custom fcall functions. Modify equations directly
        """
        pass

    def c_numeric(self, **kwargs):
        pass

    def s_numeric(self, **kwargs):
        pass

    def j_numeric(self, **kwargs):
        """
        Custom numeric update functions.

        This function should append indices to `_ifx`, `_jfx`, and `_vfx`.
        It is only called once in `store_sparse_pattern`.

        Example
        -------
        """
        pass

    def _param_doc(self, max_width=80, export='plain'):
        """
        Export formatted model documentation as a string

        Returns
        -------
        str
        """
        # parameter documentation

        if len(self.params) == 0:
            return ''

        table = Tab(title='Parameters', export=export)
        rows = []
        for p in self.params.values():
            property_list = []
            for key, val in p.property.items():
                if val is True:
                    property_list.append(key)
            property_str = ','.join(property_list)
            rows.append((p.name,
                         p.unit if p.unit else '',
                         p.class_name,
                         p.info if p.info else '',
                         p.default if p.default is not None else '',
                         property_str))

        table.add_rows(rows, header=False)
        table.header(('Name', 'Unit', 'Type', 'Description', 'Default', 'Properties'))

        return table.draw()

    def _var_doc(self, max_width=80, export='plain'):
        # variable documentation
        if len(self.cache.all_vars) == 0:
            return ''

        table = Tab(title='Variables', max_width=max_width, export=export)

        call_store = self.system.calls[self.class_name]
        tex_names = self.math_wrap(call_store.x_latex + call_store.y_latex + call_store.c_latex, export=export)
        tex_init = self.math_wrap(call_store.init_latex.values(), export=export)

        rows = []
        for i, var in enumerate(self.cache.all_vars.values()):
            all_properties = ['v_init', 'v_setter', 'e_setter']
            property_list = []
            for item in all_properties:
                if (var.__dict__[item] is not None) and (var.__dict__[item] is not False):
                    property_list.append(item)
            property_str = ','.join(property_list)

            rows.append((var.name,
                         tex_names[i],
                         var.unit if var.unit else '',
                         tex_init[i],
                         property_str,
                         var.info if var.info else ''))

        table.add_rows(rows, header=False)
        table.header(('Name', 'Symbol', 'Unit', 'Initial Value', 'Properties', 'Description'))

        return table.draw()

    @staticmethod
    def math_wrap(tex_str_list, export):
        # only wrap when export format is ``rest``
        if export != 'rest':
            return list(tex_str_list)

        out = []
        for item in tex_str_list:
            if item is None or item == '':
                out.append('')
            else:
                out.append(rf':math:`{item}`')
        return out

    def _eq_doc(self, max_width=80, export='plain'):
        # equation documentation
        if len(self.cache.all_vars) == 0:
            return ''

        table = Tab(title='Equations', max_width=max_width, export=export)

        call_store = self.system.calls[self.class_name]
        tex_names = self.math_wrap(call_store.x_latex + call_store.y_latex + call_store.c_latex, export=export)
        tex_eqs = self.math_wrap(call_store.f_latex + call_store.g_latex + call_store.h_latex, export=export)

        rows = []
        for i, var in enumerate(self.cache.all_vars.values()):
            rows.append((var.name, tex_names[i], tex_eqs[i], var.class_name))

        table.add_rows(rows, header=False)
        table.header(('Name', 'Symbol', 'Equation (x\'=f or g=0)', 'Type'))

        return table.draw()

    def _service_doc(self, max_width=80, export='plain'):
        if len(self.services) == 0:
            return ''

        table = Tab(title='Services', max_width=max_width, export=export)

        call_store = self.system.calls[self.class_name]
        tex_names = self.math_wrap([item.tex_name for item in self.services.values()], export=export)
        tex_eqs = self.math_wrap(call_store.service_latex.values(), export=export)

        rows = []
        for i, var in enumerate(self.services.values()):
            rows.append((var.name,
                         tex_names[i],
                         tex_eqs[i] if (var.v_str is not None) else var.v_numeric,
                         var.class_name))

        table.add_rows(rows, header=False)
        table.header(('Name', 'Symbol', 'Equation or Callback', 'Type'))
        return table.draw()

    def _discrete_doc(self, max_width=80, export='plain'):
        if len(self.discrete) == 0:
            return ''

        table = Tab(title='Discrete Components', max_width=max_width, export=export)
        rows = []
        for var in self.discrete.values():
            rows.append((var.name, var.class_name,
                         var.var.name if hasattr(var, 'var') else '',
                         var.upper.name if hasattr(var, 'upper') else '',
                         var.lower.name if hasattr(var, 'lower') else ''
                         ))

        table.add_rows(rows, header=False)
        table.header(('Name', 'Type', 'Output', 'Upper Bound', 'Lower Bound'))
        return table.draw()

    def _block_doc(self, max_width=80, export='plain'):
        return ''

    def doc(self, max_width=80, export='plain'):
        out = ''
        if export == 'rest':
            max_width = 0
            model_header = '--------------------------------------------------------------------------------\n'
            out += f'.. _{self.class_name}:\n\n'
        else:
            model_header = ''

        if export == 'rest':
            out += model_header + f'{self.class_name}\n' + model_header
            out += f'\nGroup {self.group}_\n\n'
        else:
            out += model_header + f'Model <{self.class_name}> in Group <{self.group}>\n' + model_header

        if self.__doc__ is not None:
            out += self.__doc__

        # add tables
        out += self._param_doc(max_width=max_width, export=export) + \
            self._var_doc(max_width=max_width, export=export) + \
            self._eq_doc(max_width=max_width, export=export) + \
            self._service_doc(max_width=max_width, export=export) + \
            self._discrete_doc(max_width=max_width, export=export) + \
            self._block_doc(max_width=max_width, export=export)
        return out
