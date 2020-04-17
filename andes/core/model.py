"""
Base class for building ANDES models
"""
import os
import logging
from collections import OrderedDict, defaultdict
from typing import Iterable

from andes.core.config import Config
from andes.core.discrete import Discrete
from andes.core.block import Block
from andes.core.triplet import JacTriplet
from andes.core.param import BaseParam, RefParam, IdxParam, DataParam, NumParam, ExtParam, TimerParam
from andes.core.var import BaseVar, Algeb, State, ExtAlgeb, ExtState
from andes.core.service import BaseService, ConstService
from andes.core.service import ExtService, OperationService, RandomService

from andes.utils.paths import get_pkl_path
from andes.utils.func import list_flatten
from andes.utils.tab import make_doc_table, math_wrap

from andes.shared import np, pd, newton_krylov
from andes.shared import jac_names, jac_types, jac_full_names

logger = logging.getLogger(__name__)


class Cache(object):
    """
    Class for caching the return value of callback functions.
    """
    def __init__(self):
        self._callbacks = {}

    def __getattr__(self, item):
        if item == "_callbacks":
            return self.__getattribute__(item)

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
        """
        Helper function for calling callback functions.

        Parameters
        ----------
        name : str
            attribute name to be updated

        Returns
        -------
        callback result
        """
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

    This class is designed to hold the parameter data separately from model equations.
    Models should inherit this class to define the parameters from input files.
    """

    def __init__(self, *args, **kwargs):
        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.ref_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.n = 0
        self.uid = {}

        if not hasattr(self, 'cache'):
            self.cache = Cache()
        self.cache.add_callback('dict', self.as_dict)
        self.cache.add_callback('dict_in', lambda: self.as_dict(True))
        self.cache.add_callback('df', self.as_df)
        self.cache.add_callback('df_in', self.as_df_in)

        self.idx = DataParam(info='unique device idx')
        self.u = NumParam(default=1, info='connection status', unit='bool', tex_name='u')
        self.name = DataParam(info='device name', tex_name='name')

    def __len__(self):
        return self.n

    def __setattr__(self, key, value):
        if isinstance(value, BaseParam):
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

    def add(self, **kwargs):
        """
        Add a device (an instance) to this model.

        Warnings
        --------
        This function is not intended to be used directly. Use the ``add`` method from System so that the index
        can be registered correctly.

        Parameters
        ----------
        kwargs
            model parameters are collected into the kwargs dictionary
        """
        idx = kwargs['idx']
        self.uid[idx] = self.n
        self.n += 1
        if kwargs.get("name") is None:
            kwargs["name"] = idx

        for name, instance in self.params.items():
            # TODO: Consider making `RefParam` not subclass of `BaseParam`
            # skip `RefParam` because it is collected outside `add`
            if isinstance(instance, RefParam):
                continue
            value = kwargs.pop(name, None)
            instance.add(value)
        if len(kwargs) > 0:
            logger.warning(f'{self.__class__.__name__}: Unused data {kwargs}')

    def as_dict(self, vin=False):
        """
        Export all parameters as a dict.

        Returns
        -------
        dict
            a dict with the keys being the `ModelData` parameter names
            and the values being an array-like of data in the order of adding.
            An additional `uid` key is added with the value default to range(n).
        """
        out = dict()
        out['uid'] = np.arange(self.n)

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
        Export all parameters as a `pandas.DataFrame` object.
        This function utilizes `as_dict` for preparing data.

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        out = pd.DataFrame(self.as_dict()).set_index('uid')

        return out

    def as_df_in(self):
        """
        Export all parameters from original input (``vin``) as a `pandas.DataFrame`.
        This function utilizes `as_dict` for preparing data.

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        """
        out = pd.DataFrame(self.as_dict(vin=True)).set_index('uid')

        return out

    def find_param(self, prop):
        """
        Find params with the given property and return in an OrderedDict.

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

    def find_idx(self, keys, values, allow_missing=False):
        """
        Find `idx` of devices whose values match the given pattern.

        Parameters
        ----------
        keys : str, array-like, Sized
            A string or an array-like of strings containing the names of parameters for the search criteria
        values : array, array of arrays
            Values for the corresponding key to search for. If keys is a str, values should be an array of
            elements. If keys is a list, values should be an array of arrays, each corresponds to the key.
        allow_missing : bool, Sized
            Allow key, value to be not found. Used by groups.

        Returns
        -------
        list
            indices of devices
        """
        if isinstance(keys, str):
            keys = (keys, )
            if not isinstance(values, (int, float, str)) and not isinstance(values, Iterable):
                raise ValueError("value must be a string, scalar or an iterable")
            elif len(values) > 0 and not isinstance(values[0], Iterable):
                values = (values, )
        elif isinstance(keys, Iterable):
            if not isinstance(values, Iterable):
                raise ValueError("value must be an iterable")
            elif len(values) > 0 and not isinstance(values[0], Iterable):
                raise ValueError("if keys is an iterable, values must be an iterable of iterables")
            if len(keys) != len(values):
                raise ValueError("keys and values must have the same length")

        v_attrs = [self.__dict__[key].v for key in keys]

        idxes = []
        for v_search in zip(*values):
            v_idx = None
            for pos, v_attr in enumerate(zip(*v_attrs)):
                if all([i == j for i, j in zip(v_search, v_attr)]):
                    v_idx = self.idx.v[pos]
                    break
            if v_idx is None:
                if allow_missing is False:
                    raise IndexError(f'{keys} = {v_search} not found in {self.__class__.__name__}')
                else:
                    v_idx = False

            idxes.append(v_idx)

        return idxes


class ModelCall(object):
    """
    Class for storing generated function calls and Jacobians.
    """
    def __init__(self):
        # callables to be generated by sympy.lambdify
        self.g_lambdify = None
        self.f_lambdify = None
        self.h_lambdify = None
        self.init_lambdify = None
        self.s_lambdify = None

        self.ijac = defaultdict(list)
        self.jjac = defaultdict(list)
        self.vjac = defaultdict(list)

    def clear_ijv(self):
        for jname in jac_names:
            for jtype in jac_types:
                self.ijac[jname + jtype] = list()
                self.jjac[jname + jtype] = list()
                self.vjac[jname + jtype] = list()

    def append_ijv(self, j_full_name, ii, jj, vv):
        if not isinstance(ii, int):
            raise ValueError("i index must be an integer")
        if not isinstance(jj, int):
            raise ValueError("j index must be an integer")
        if not isinstance(vv, (int, float)) and (not callable(vv)):
            raise ValueError("j index must be numerical or callable")

        self.ijac[j_full_name].append(ii)
        self.jjac[j_full_name].append(jj)
        self.vjac[j_full_name].append(vv)

    def zip_ijv(self, j_full_name):
        """
        Return a zipped iterator for the rows, cols and vals for the specified matrix name.
        """
        return zip(self.ijac[j_full_name],
                   self.jjac[j_full_name],
                   self.vjac[j_full_name])


class Model(object):
    """
    Base class for power system DAE models.

    Attributes
    ----------
    n : int
        The number of loaded elements.

    idx : list
        A list of all element idx.

    num_params : OrderedDict
        {name: instance} of numerical parameters, including internal
        and external ones
    """

    def __init__(self, system=None, config=None):
        self.system = system

        # duplicate attributes from ModelData. Keep for now.
        self.n = 0
        self.group = 'Undefined'

        if not hasattr(self, 'num_params'):
            self.num_params = OrderedDict()
        if not hasattr(self, 'cache'):
            self.cache = Cache()

        # variables
        self.states = OrderedDict()           # internal states
        self.states_ext = OrderedDict()       # external states
        self.algebs = OrderedDict()           # internal algebraic variables
        self.algebs_ext = OrderedDict()       # external algebraic vars
        self.vars_decl_order = OrderedDict()  # variable in the order of declaration

        # external parameters
        self.params_ext = OrderedDict()

        # discrete and control blocks
        self.discrete = OrderedDict()
        self.blocks = OrderedDict()

        # service/temporary variables
        self.services = OrderedDict()
        # time-constant services
        self.services_tc = OrderedDict()
        # external services (to be retrieved)
        self.services_ext = OrderedDict()
        # operational services (for special usages)
        self.services_ops = OrderedDict()

        # cache callback and lambda function storage
        self.calls = ModelCall()

        # class behavior flags
        self.flags = dict(
            collate=True,       # True: collate variables by device; False: by variable
            pflow=False,        # True: called during power flow
            tds=False,          # True if called during tds; if is False, ``dae_t`` cannot be used
            series=False,       # True if is series device
            nr_iter=False,      # True if require iterative initialization
            sys_base=False,     # True if is parameters have been converted to system base
            address=False,      # True if address is assigned
            initialized=False,  # True if variables have been initialized
        )

        # model `config` that can be exported to the `andes.rc` file
        self.config = Config(name=self.class_name)
        if config is not None:
            self.config.load(config)

        self.tex_names = OrderedDict((('dae_t', 't_{dae}'),))
        # symbols that are input to lambda functions
        # including parameters, variables, services, configs and "dae_t"
        self.input_syms = OrderedDict()
        self.vars_syms = OrderedDict()
        self.iter_syms = OrderedDict()
        self.vars_syms_list = list()
        self.non_vars_syms = OrderedDict()  # input_syms - vars_syms
        self.non_iter_syms = OrderedDict()  # input_syms - iter_syms
        self.f_syms, self.g_syms = list(), list()  # symbolic equations in lists
        self.f_matrix, self.g_matrix, self.s_matrix = list(), list(), list()  # equations in matrices

        # pretty print of variables
        self.vars_print = list()   # variables in the order of states, algebs
        self.f_print, self.g_print, self.s_print = list(), list(), list()
        self.df_print, self.dg_print = None, None

        self.triplets = JacTriplet()

        # cached class attributes
        self.cache.add_callback('all_vars', self._all_vars)
        self.cache.add_callback('iter_vars', self._iter_vars)
        self.cache.add_callback('all_vars_names', self._all_vars_names)
        self.cache.add_callback('all_params', self._all_params)
        self.cache.add_callback('all_params_names', self._all_params_names)
        self.cache.add_callback('algebs_and_ext', self._algebs_and_ext)
        self.cache.add_callback('states_and_ext', self._states_and_ext)
        self.cache.add_callback('services_and_ext', self._services_and_ext)
        self.cache.add_callback('vars_ext', self._vars_ext)
        self.cache.add_callback('vars_int', self._vars_int)

        # cached dictionary of inputs
        self._input = OrderedDict()
        # discrete flags in an OrderedDict
        self._input_z = OrderedDict()

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
        elif isinstance(value, ConstService):   # services with only `v_str`
            self.services[key] = value
        elif isinstance(value, ExtService):
            self.services_ext[key] = value
        elif isinstance(value, (OperationService, RandomService)):
            self.services_ops[key] = value

        elif isinstance(value, Block):
            self.blocks[key] = value
            # pull in sub-variables from control blocks
            for var_name, var_instance in value.export().items():
                var_instance.name = f'{value.name}_{var_name}'
                var_instance.tex_name = f'{var_instance.tex_name}_{{{value.tex_name}}}'
                self.__setattr__(var_instance.name, var_instance)

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names.

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
                logger.warning(f"{self.class_name}: redefinition of member <{key}>")

    def __setattr__(self, key, value):
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
            logger.error("idx2uid cannot search for None idx")
            return None
        if isinstance(idx, (float, int, str, np.int32, np.int64, np.float64)):
            return self.uid[idx]
        elif isinstance(idx, Iterable):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self.uid[i] for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def get(self, src: str, idx, attr: str = 'v'):
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

        Returns
        -------
        array-like
            ``self.<src>.<attr>[idx]``

        """
        uid = self.idx2uid(idx)
        if isinstance(self.__dict__[src].__dict__[attr], list):
            if isinstance(uid, Iterable):
                return [self.__dict__[src].__dict__[attr][i] for i in uid]

        return self.__dict__[src].__dict__[attr][uid]

    def set(self, src, idx, attr, value):
        """
        Set the value of an attribute of a model property.

        Performs ``self.<src>.<attr>[idx] = value``

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            Values to be set

        Returns
        -------
        None

        """
        uid = self.idx2uid(idx)
        self.__dict__[src].__dict__[attr][uid] = value

    def alter(self, src, idx, value):
        """
        Alter input parameter value.

        This function converts the new parameter to per unit.
        """
        self.set(src, idx, 'vin', value)
        self.__dict__[src].v[:] = self.__dict__[src].vin * self.__dict__[src].pu_coeff

    def get_inputs(self, refresh=False):
        """
        Get an OrderedDict of the inputs to the numerical function calls.

        Parameters
        ----------
        refresh : bool
            Refresh the values in the dictionary.
            This is only used when the memory address of arrays changed.
            After initialization, all array assignments are inplace.
            To avoid overhead, refresh should not be used after initialization.

        Returns
        -------
        OrderedDict
            The input name and value array pairs in an OrderedDict

        """
        if len(self._input) == 0 or refresh:
            self._refresh_inputs()

        # update`dae_t`
        self._input['dae_t'] = self.system.dae.t
        return self._input

    def _refresh_inputs(self):
        """
        This is the helper function to refresh inputs.

        The functions collects objects into OrderedDict and store to `self._input` and `self._input_z`.

        Returns
        -------
        None

        """
        # The order of inputs: `all_params` and then `all_vars`, finally `config`
        # the below sequence should correspond to `self.all_param_names`
        for instance in self.num_params.values():
            self._input[instance.name] = instance.v

        for instance in self.services.values():
            self._input[instance.name] = instance.v

        for instance in self.services_ext.values():
            self._input[instance.name] = instance.v

        # discrete flags
        for instance in self.discrete.values():
            for name, val in zip(instance.get_names(), instance.get_values()):
                self._input[name] = val
                self._input_z[name] = val

        # append all variable values
        for instance in self.cache.all_vars.values():
            self._input[instance.name] = instance.v

        # append config variables
        for key, val in self.config.as_dict().items():
            self._input[key] = val

    def l_update_var(self):
        """
        Call the ``check_var`` method of discrete components to update the internal status flags.

        The function is variable-dependent and should be called before updating equations.

        Returns
        -------
        None
        """
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_var()

    def l_check_eq(self):
        """
        Call the ``check_eq`` method of discrete components to update equation-dependent flags.

        This function should be called after equation updates.

        Returns
        -------
        None
        """
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_eq()

    def l_set_eq(self):
        """
        Call the ``set_eq`` method of discrete components.

        This function is only used by AntiWindup to append the pegged states to the ``x_set`` list.

        Returns
        -------
        None
        """
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.set_eq()

    def s_update(self):
        """
        Update service equation values.

        This function is only evaluated at initialization.
        Service values are updated sequentially.
        The ``v`` attribute of services will be assigned at a new memory.
        """
        if self.n == 0:
            return

        if (self.calls.s_lambdify is not None) and len(self.calls.s_lambdify):
            for name, instance in self.services.items():
                func = self.calls.s_lambdify[name]
                if callable(func):
                    kwargs = self.get_inputs(refresh=True)
                    # DO NOT use in-place operation since the return can be complex number
                    instance.v = func(**kwargs)
                else:
                    instance.v = func

                if not isinstance(instance.v, np.ndarray):
                    instance.v = instance.v * np.ones(self.n)

        # NOTE:
        # Some numerical calls depend on other service values.
        # They are evaluated after lambdified calls

        # Apply both the individual `v_numeric` and Model-level `s_numeric`
        for instance in self.services.values():
            func = instance.v_numeric
            if callable(func):
                kwargs = self.get_inputs(refresh=True)
                instance.v = func(**kwargs)

        # Evaluate TimeConstant multiplicative inverse
        for instance in self.services_tc.values():
            instance.inverse()

        kwargs = self.get_inputs(refresh=True)
        self.s_numeric(**kwargs)

    def generate_pycode_file(self):
        """
        Create output source code file for generated code
        """
        models_dir = os.path.join(get_pkl_path(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        file = os.path.join(models_dir, self.class_name.lower() + '.py')
        self.code_file = open(file, 'w')

    def generate_initializers(self):
        """
        Generate lambda functions for initial values.
        """
        logger.debug(f'Generating initializers for {self.class_name}')
        from sympy import sympify, lambdify, Matrix
        from sympy.printing import latex

        init_lambda_list = OrderedDict()
        init_latex = OrderedDict()
        init_seq_list = []
        init_g_list = []  # initialization equations in g(x, y) = 0 form

        input_syms_list = list(self.input_syms)

        for name, instance in self.cache.all_vars.items():
            if instance.v_str is None and instance.v_iter is None:
                init_latex[name] = ''
            else:
                if instance.v_str is not None:
                    sympified = sympify(instance.v_str, locals=self.input_syms)
                    self._check_expr_symbols(sympified)
                    lambdified = lambdify(input_syms_list, sympified, 'numpy')
                    init_lambda_list[name] = lambdified
                    init_latex[name] = latex(sympified.subs(self.tex_names))
                    init_seq_list.append(sympify(f'{instance.v_str} - {name}', locals=self.input_syms))

                if instance.v_iter is not None:
                    sympified = sympify(instance.v_iter, locals=self.input_syms)
                    self._check_expr_symbols(sympified)
                    init_g_list.append(sympified)
                    init_latex[name] = latex(sympified.subs(self.tex_names))

        self.init_seq = Matrix(init_seq_list)
        self.init_std = Matrix(init_g_list)
        self.init_dstd = Matrix([])
        if len(self.init_std) > 0:
            self.init_dstd = self.init_std.jacobian(list(self.vars_syms.values()))

        self.calls.init_lambdify = init_lambda_list
        self.calls.init_latex = init_latex
        self.calls.init_std = lambdify((list(self.iter_syms), list(self.non_iter_syms)),
                                       self.init_std,
                                       'numpy')

    def _init_wrap(self, x0, params):
        """
        A wrapper for converting the initialization equations into standard forms g(x) = 0, where x is an array.
        """
        vars_input = []
        for i, _ in enumerate(self.cache.iter_vars.values()):
            vars_input.append(x0[i * self.n: (i + 1) * self.n])

        return np.ravel(self.calls.init_std(vars_input, params))

    def solve_initialization(self):
        """
        Solve the initialization equation using the Newton-Krylov method.
        """
        inputs = self.get_inputs(refresh=True)

        iter_input = OrderedDict()
        non_iter_input = OrderedDict(inputs)  # include non-iter variables and other params/configs

        for name, _ in self.cache.iter_vars.items():
            iter_input[name] = inputs[name]
            non_iter_input.pop(name)

        iter_array = list(iter_input.values())
        non_iter_list = list(non_iter_input.values())

        for i in range(len(iter_array)):
            if isinstance(iter_array[i], (float, int, np.int64, np.float64)):
                iter_array[i] = np.ones(self.n) * iter_array[i]

        iter_array = np.ravel(iter_array)

        def init_wrap(x0):
            return self._init_wrap(x0, non_iter_list)

        sol = newton_krylov(init_wrap, iter_array)

        for i, var in enumerate(self.cache.iter_vars.values()):
            var.v[:] = sol[i * self.n: (i + 1) * self.n]

    def generate_symbols(self):
        """
        Generate symbols for symbolic equation generations.

        This function should run before other generate equations.

        Attributes
        ----------
        input_syms : OrderedDict
            name-symbol pair of all parameters, variables and configs

        vars_syms : OrderedDict
            name-symbol pair of all variables, in the order of (states_and_ext + algebs_and_ext)

        non_vars_syms : OrderedDict
            name-symbol pair of alll non-variables, namely, (input_syms - vars_syms)

        Returns
        -------

        """
        logger.debug(f'Generating symbols for {self.class_name}')
        from sympy import Symbol, Matrix

        # clear symbols storage
        self.f_syms, self.g_syms = list(), list()
        self.f_matrix, self.g_matrix = Matrix([]), Matrix([])

        # process tex_names defined in model
        # -----------------------------------------------------------
        for key in self.tex_names.keys():
            self.tex_names[key] = Symbol(self.tex_names[key])
        for instance in self.discrete.values():
            for name, tex_name in zip(instance.get_names(), instance.get_tex_names()):
                self.tex_names[name] = tex_name
        # -----------------------------------------------------------

        for var in self.cache.all_params_names:
            self.input_syms[var] = Symbol(var)

        for var in self.cache.all_vars_names:
            tmp = Symbol(var)
            self.vars_syms[var] = tmp
            self.input_syms[var] = tmp
            if self.__dict__[var].v_iter is not None:
                self.iter_syms[var] = tmp

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = Symbol(key)
            self.input_syms[key] = tmp
            if key in self.config.tex_names:
                self.tex_names[tmp] = Symbol(self.config.tex_names[key])

        # store tex names for pretty printing replacement later
        for var in self.input_syms:
            if var in self.__dict__ and self.__dict__[var].tex_name is not None:
                self.tex_names[Symbol(var)] = Symbol(self.__dict__[var].tex_name)

        self.input_syms['dae_t'] = Symbol('dae_t')

        # build ``non_vars_syms`` by removing ``vars_syms`` keys from a copy of ``input_syms``
        self.non_vars_syms = OrderedDict(self.input_syms)
        self.non_iter_syms = OrderedDict(self.input_syms)
        for key in self.vars_syms:
            self.non_vars_syms.pop(key)
        for key in self.iter_syms:
            self.non_iter_syms.pop(key)

        self.vars_syms_list = list(self.vars_syms.values())  # useful for ``.jacobian()``

    def _check_expr_symbols(self, expr):
        """Check if expression contains unknown symbols"""
        for item in expr.free_symbols:
            if item not in self.input_syms.values():
                raise ValueError(f'{self.class_name} expression "{expr}" contains unknown symbol "{item}"')

    def generate_equations(self):
        logger.debug(f'Generating equations for {self.__class__.__name__}')
        from sympy import Matrix, sympify, lambdify

        inputs_list = list(self.input_syms)
        iter_list = [self.cache.states_and_ext, self.cache.algebs_and_ext]
        dest_list = [self.f_syms, self.g_syms]

        for it, dest in zip(iter_list, dest_list):
            for instance in it.values():
                if instance.e_str is None:
                    dest.append(0)
                else:
                    try:
                        expr = sympify(instance.e_str, locals=self.input_syms)
                        self._check_expr_symbols(expr)
                        dest.append(expr)
                    except TypeError as e:
                        logger.error(f'Error sympifying <{instance.e_str}> for <{instance.name}>')
                        raise e

        # convert to SymPy matrices
        self.f_matrix = Matrix(self.f_syms)
        self.g_matrix = Matrix(self.g_syms)

        self.calls.g_lambdify = lambdify(inputs_list,
                                         self.g_matrix,
                                         'numpy')
        self.calls.f_lambdify = lambdify(inputs_list,
                                         self.f_matrix,
                                         'numpy')

        # convert service equations
        # Service equations are converted sequentially because Services can be interdependent
        s_syms = OrderedDict()
        s_lambdify = OrderedDict()
        for name, instance in self.services.items():
            if instance.v_str is not None:
                expr = sympify(instance.v_str, locals=self.input_syms)
                self._check_expr_symbols(expr)
                s_syms[name] = expr
                s_lambdify[name] = lambdify(inputs_list,
                                            s_syms[name],
                                            'numpy')
            else:
                s_syms[name] = 0
                s_lambdify[name] = 0

        self.s_matrix = Matrix(list(s_syms.values()))
        self.calls.s_lambdify = s_lambdify

    def generate_jacobians(self):
        """
        Generate Jacobians and store to corresponding triplets.

        The internal indices of equations and variables are stored, alongside the lambda functions.

        For example, dg/dy is a sparse matrix whose elements are ``(row, col, val)``, where ``row`` and ``col``
        are the internal indices, and ``val`` is the numerical lambda function. They will be stored to

            row -> self.calls._igy
            col -> self.calls._jgy
            val -> self.calls._vgy

        Returns
        -------
        None
        """
        logger.debug(f'Generating Jacobians for {self.__class__.__name__}')
        from sympy import SparseMatrix, lambdify, Matrix

        # clear storage
        self.df_syms, self.dg_syms = Matrix([]), Matrix([])
        self.calls.clear_ijv()

        # NOTE: SymPy does not allow getting the derivative of an empty array
        if len(self.g_matrix) > 0:
            self.dg_syms = self.g_matrix.jacobian(self.vars_syms_list)

        if len(self.f_matrix) > 0:
            self.df_syms = self.f_matrix.jacobian(self.vars_syms_list)

        self.df_sparse = SparseMatrix(self.df_syms)
        self.dg_sparse = SparseMatrix(self.dg_syms)

        vars_syms_list = list(self.vars_syms)
        syms_list = list(self.input_syms)
        algebs_and_ext_list = list(self.cache.algebs_and_ext)
        states_and_ext_list = list(self.cache.states_and_ext)

        fg_sparse = [self.df_sparse, self.dg_sparse]
        for idx, eq_sparse in enumerate(fg_sparse):
            for item in eq_sparse.row_list():
                e_idx, v_idx, e_symbolic = item

                if idx == 0:
                    eq_name = states_and_ext_list[e_idx]
                else:
                    eq_name = algebs_and_ext_list[e_idx]

                var_name = vars_syms_list[v_idx]
                eqn = self.cache.all_vars[eq_name]
                var = self.cache.all_vars[var_name]

                jname = f'{eqn.e_code}{var.v_code}'
                self.calls.append_ijv(jname, e_idx, v_idx, lambdify(syms_list, e_symbolic))

        # The for loop below is intended to add an epsilon small value to the diagonal of gy matrix.
        # The user should take care of the algebraic equations by using `diag_eps` in Algeb definition

        for var in self.algebs.values():
            if var.diag_eps == 0.0:
                continue
            e_idx = algebs_and_ext_list.index(var.name)
            v_idx = vars_syms_list.index(var.name)
            self.calls.append_ijv('gyc', e_idx, v_idx, var.diag_eps)

    def generate_pretty_print(self):
        """
        Generate pretty print variables and equations.
        """
        logger.debug(f"Generating pretty prints for {self.class_name}")
        from sympy import Matrix
        from sympy.printing import latex

        # equation symbols for pretty printing
        self.f_print, self.g_print = Matrix([]), Matrix([])

        self.vars_print = Matrix(list(self.vars_syms.values())).subs(self.tex_names)

        # get pretty printing equations by substituting symbols
        self.f_print = self.f_matrix.subs(self.tex_names)
        self.g_print = self.g_matrix.subs(self.tex_names)
        self.s_print = self.s_matrix.subs(self.tex_names)

        # store latex strings
        nx = len(self.f_print)
        ny = len(self.g_print)
        self.calls.x_latex = [latex(item) for item in self.vars_print[:nx]]
        self.calls.y_latex = [latex(item) for item in self.vars_print[nx:nx + ny]]

        self.calls.f_latex = [latex(item) for item in self.f_print]
        self.calls.g_latex = [latex(item) for item in self.g_print]
        self.calls.s_latex = [latex(item) for item in self.s_print]

        self.df_print = self.df_sparse.subs(self.tex_names)
        self.dg_print = self.dg_sparse.subs(self.tex_names)

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
        non-empty values, which, as a combination, is not accepted by `cvxopt.spmatrix`.
        """

        self.triplets.clear_ijv()
        if self.n == 0:
            return
        if self.flags['address'] is False:
            return

        # store model-level user-defined Jacobians
        self.j_numeric()
        # store and merge user-defined Jacobians in blocks
        for instance in self.blocks.values():
            instance.j_numeric()
            self.triplets.merge(instance.triplets)

        var_names_list = list(self.cache.all_vars.keys())
        eq_names = {'f': var_names_list[:len(self.cache.states_and_ext)],
                    'g': var_names_list[len(self.cache.states_and_ext):]}

        # prepare all combinations of Jacobian names (fx, fxc, gx, gxc, etc.)

        for j_full_name in jac_full_names:
            for row, col, val in self.calls.zip_ijv(j_full_name):
                row_name = eq_names[j_full_name[0]][row]  # where jname[0] is the equation name (f, g, r, t)
                col_name = var_names_list[col]

                row_idx = self.__dict__[row_name].a
                col_idx = self.__dict__[col_name].a

                if len(row_idx) != len(col_idx):
                    logger.error(f'row {row_name}, row_idx: {row_idx}')
                    logger.error(f'col {col_name}, col_idx: {col_idx}')
                    raise ValueError(f'{self.class_name}: non-matching row_idx and col_idx')

                if j_full_name[-1] == 'c':
                    value = val * np.ones(self.n)
                else:
                    value = np.zeros(self.n)
                self.triplets.append_ijv(j_full_name, row_idx, col_idx, value)

    def init(self):
        """
        Numerical initialization of a model.

        Initialization sequence:
        1. Sequential initialization based on the order of definition
        2. Use Newton-Krylov method for iterative initialization
        3. Custom init
        """
        if self.n == 0:
            return
        logger.debug(f'{self.class_name:<10s}: calling initialize()')

        # update service values
        self.s_update()

        for name, instance in self.vars_decl_order.items():
            if instance.v_str is None:
                continue

            kwargs = self.get_inputs(refresh=True)
            init_fun = self.calls.init_lambdify[name]
            if callable(init_fun):
                instance.v[:] = init_fun(**kwargs)
            else:
                instance.v[:] = init_fun

        # experimental: user Newton-Krylov solver for dynamic initialization
        # ----------------------------------------
        if self.flags['nr_iter']:
            self.solve_initialization()
        # ----------------------------------------

        # call custom variable initializer after generated initializers
        kwargs = self.get_inputs(refresh=True)
        self.v_numeric(**kwargs)

        self.flags['initialized'] = True

    def get_init_order(self):
        """
        Get variable initialization order and send to `logger.info`.
        """
        out = []
        for name in self.vars_decl_order.keys():
            out.append(name)

        logger.info(f'Initialization order: {",".join(out)}')

    def f_update(self):
        """
        Evaluate differential equations.
        """

        if self.n == 0:
            return
        kwargs = self.get_inputs()

        # call lambda functions in self.call
        ret = self.calls.f_lambdify(**kwargs)
        for idx, instance in enumerate(self.cache.states_and_ext.values()):
            instance.e += ret[idx][0]

        # numerical calls defined in the model
        self.f_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.f_numeric(**kwargs)

    def g_update(self):
        """
        Evaluate algebraic equations.
        """

        if self.n == 0:
            return
        kwargs = self.get_inputs()

        # call lambda functions stored in `self.calls`
        ret = self.calls.g_lambdify(**kwargs)
        for idx, instance in enumerate(self.cache.algebs_and_ext.values()):
            instance.e += ret[idx][0]

        # numerical calls defined in the model
        self.g_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.g_numeric(**kwargs)

    def j_update(self):
        """
        Update Jacobians and store the Jacobian values to ``self.v<JName>``, where ``<JName>`` is the Jacobian
        name.

        Returns
        -------
        None
        """
        if self.n == 0:
            return

        jac_set = ('fx', 'fy', 'gx', 'gy')
        kwargs = self.get_inputs()
        for name in jac_set:
            for idx, fun in enumerate(self.calls.vjac[name]):
                self.triplets.vjac[name][idx] = fun(**kwargs)

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
        """
        if self.n == 0:
            return

        for timer in self.timer_params.values():
            if timer.callback is not None:
                timer.callback(timer.is_time(dae_t))

        # TODO: consider `Block` with timer

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

    def list2array(self):
        """
        Convert all the value attributes ``v`` to NumPy arrays.

        Value attribute arrays should remain in the same address afterwards.
        Namely, all assignments to value array should be operated in place (e.g., with [:]).
        """

        for instance in self.num_params.values():
            instance.to_array()

        for instance in self.cache.services_and_ext.values():
            instance.v = np.zeros(self.n)

        for instance in self.discrete.values():
            instance.list2array(self.n)

    def a_reset(self):
        """
        Reset addresses to empty and reset flags['address'] to ``False``.
        """
        if self.n == 0:
            return
        for var in self.cache.all_vars.values():
            var.reset()
        self.flags['address'] = False
        self.flags['initialized'] = False

    def e_clear(self):
        """
        Clear equation value arrays associated with all internal variables.
        """
        if self.n == 0:
            return
        for instance in self.cache.all_vars.values():
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

    def j_numeric(self, **kwargs):
        """
        Custom numeric update functions.

        This function should append indices to `_ifx`, `_jfx`, and append anonymous functions to `_vfx`.
        It is only called once by `store_sparse_pattern`.
        """
        pass

    def _param_doc(self, max_width=80, export='plain'):
        """
        Export formatted model parameter documentation as a string.

        Parameters
        ----------
        max_width : int, optional = 80
            Maximum table width. If export format is ``rest`` it will be unlimited.

        export : str, optional = 'plain'
            Export format, 'plain' for plain text, 'rest' for restructuredText.

        Returns
        -------
        str
            Tabulated output in a string
        """
        if len(self.params) == 0:
            return ''

        # prepare temporary lists
        names, units, class_names = list(), list(), list()
        info, defaults, properties = list(), list(), list()
        units_rest = list()

        for p in self.params.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            plist = []
            for key, val in p.property.items():
                if val is True:
                    plist.append(key)
            properties.append(','.join(plist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.params.values()]
            symbols = math_wrap(symbols, export=export)
        else:
            symbols = [item.name for item in self.params.values()]

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Default', defaults),
                                  ('Unit', units),
                                  ('Type', class_names),
                                  ('Properties', properties)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Default', defaults),
                                 ('Unit', units_rest),
                                 ('Type', class_names),
                                 ('Properties', properties)])

        # convert to rows and export as table
        return make_doc_table(title='Parameters',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _var_doc(self, max_width=80, export='plain'):
        # variable documentation
        if len(self.cache.all_vars) == 0:
            return ''

        names, symbols, units = list(), list(), list()
        ivs, properties, info = list(), list(), list()
        units_rest, ivs_rest = list(), list()

        for p in self.cache.all_vars.values():
            names.append(p.name)
            ivs.append(p.v_str if p.v_str else '')
            info.append(p.info if p.info else '')
            units.append(p.unit if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            # collect properties
            all_properties = ['v_str', 'v_setter', 'e_setter', 'v_iter']
            plist = []
            for item in all_properties:
                if (p.__dict__[item] is not None) and (p.__dict__[item] is not False):
                    plist.append(item)
            properties.append(','.join(plist))

        # replace with latex math expressions if export is ``rest``
        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = math_wrap(call_store.x_latex + call_store.y_latex, export=export)
            ivs_rest = math_wrap(call_store.init_latex.values(), export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Initial Value', ivs),
                                  ('Description', info),
                                  ('Unit', units),
                                  ('Properties', properties)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Initial Value', ivs_rest),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ('Properties', properties)])

        return make_doc_table(title='Variables',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _eq_doc(self, max_width=80, export='plain', e_code=None):
        out = ''
        # equation documentation
        if len(self.cache.all_vars) == 0:
            return out
        e2dict = {'f': self.cache.states_and_ext,
                  'g': self.cache.algebs_and_ext,
                  }
        e2full = {'f': 'Differential',
                  'g': 'Algebraic'}

        e2form = {'f': "T x' = f(x, y)",
                  'g': "0 = g(x, y)"}

        if e_code is None:
            e_code = ('f', 'g')
        elif isinstance(e_code, str):
            e_code = (e_code, )

        for e_name in e_code:

            if len(e2dict[e_name]) == 0:
                continue

            names, symbols = list(), list()
            eqs, eqs_rest = list(), list()
            lhs_names, lhs_tex_names = list(), list()
            class_names = list()

            for p in e2dict[e_name].values():
                names.append(p.name)
                class_names.append(p.class_name)
                eqs.append(p.e_str if p.e_str else '')
                if e_name == 'f':
                    lhs_names.append(p.t_const.name if p.t_const else '')
                    lhs_tex_names.append(p.t_const.tex_name if p.t_const else '')

            if export == 'rest':
                call_store = self.system.calls[self.class_name]
                symbols = math_wrap(call_store.x_latex + call_store.y_latex, export=export)
                eqs_rest = math_wrap(call_store.f_latex + call_store.g_latex, export=export)

            plain_dict = OrderedDict([('Name', names),
                                      ('Type', class_names),
                                      (f'RHS of Equation "{e2form[e_name]}"', eqs),
                                      ])

            rest_dict = OrderedDict([('Name', names),
                                     ('Symbol', symbols),
                                     ('Type', class_names),
                                     (f'RHS of Equation "{e2form[e_name]}"', eqs_rest),
                                     ])

            if e_name == 'f':
                plain_dict['T (LHS)'] = lhs_names
                rest_dict['T (LHS)'] = math_wrap(lhs_tex_names, export=export)

            out += make_doc_table(title=f'{e2full[e_name]} Equations',
                                  max_width=max_width,
                                  export=export,
                                  plain_dict=plain_dict,
                                  rest_dict=rest_dict)

        return out

    def _service_doc(self, max_width=80, export='plain'):
        if len(self.services) == 0:
            return ''

        names, symbols = list(), list()
        eqs, eqs_rest, class_names = list(), list(), list()

        for p in self.services.values():
            names.append(p.name)
            class_names.append(p.class_name)
            eqs.append(p.v_str if p.v_str else '')

        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = math_wrap([item.tex_name for item in self.services.values()], export=export)
            eqs_rest = math_wrap(call_store.s_latex, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Equation', eqs),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Equation', eqs_rest),
                                 ('Type', class_names)])

        return make_doc_table(title='Services',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _discrete_doc(self, max_width=80, export='plain'):
        if len(self.discrete) == 0:
            return ''

        names, symbols = list(), list()
        class_names = list()

        for p in self.discrete.values():
            names.append(p.name)
            class_names.append(p.class_name)

        if export == 'rest':
            symbols = math_wrap([item.tex_name for item in self.discrete.values()], export=export)
        plain_dict = OrderedDict([('Name', names),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', class_names)])

        return make_doc_table(title='Discrete',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _block_doc(self, max_width=80, export='plain'):
        """
        Documentation for blocks. To be implemented.
        """
        return ''

    def doc(self, max_width=80, export='plain'):
        """
        Return the model documentation in table-formatted string.

        Parameters
        ----------
        max_width : int
            Maximum table width. Automatically et to 0 if format is ``rest``.
        export : str, ('plain', 'rest')
            Export format. Use fancy table if is ``rest``.

        Returns
        -------
        str
            A string with the documentations.
        """
        out = ''
        if export == 'rest':
            max_width = 0
            model_header = '-' * 80 + '\n'
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
            out += '\n'  # this fixes the indentation for the next line

        # add tables
        out += self._param_doc(max_width=max_width, export=export) + \
            self._var_doc(max_width=max_width, export=export) + \
            self._eq_doc(max_width=max_width, export=export) + \
            self._service_doc(max_width=max_width, export=export) + \
            self._discrete_doc(max_width=max_width, export=export) + \
            self._block_doc(max_width=max_width, export=export) + \
            self.config.doc(max_width=max_width, export=export)

        return out
