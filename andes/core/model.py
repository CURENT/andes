"""
Base class for building ANDES models
"""
import logging
from collections import OrderedDict

from andes.core.config import Config
from andes.core.discrete import Discrete
from andes.core.param import BaseParam, RefParam, IdxParam, DataParam, NumParam, ExtParam, TimerParam
from andes.core.var import BaseVar, Algeb, State, ExtAlgeb, ExtState
from andes.core.block import Block
from andes.core.service import BaseService, ConstService, ExtService, OperationService, RandomService

from andes.utils.func import list_flatten
from andes.utils.tab import Tab

from andes.shared import np, pd, newton_krylov

logger = logging.getLogger(__name__)


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
        self.cache.add_callback('dict_in', lambda: self.as_dict(True))
        self.cache.add_callback('df', self.as_df)
        self.cache.add_callback('df_in', self.as_df_in)

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
        self.s_lambdify = None

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
        and external ones
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
            collate=True,
            pflow=False,
            tds=False,  # if `tds` is False, `dae_t` cannot be used
            series=False,
            nr_iter=False,
            sys_base=False,
            address=False,
            initialized=False,
        )

        self.config = Config(name=self.class_name)
        if config is not None:
            self.config.load(config)

        self.tex_names = OrderedDict((('dae_t', 't_{dae}'),))
        self.input_syms = OrderedDict()
        self.vars_syms = OrderedDict()
        self.iter_syms = OrderedDict()
        self.vars_syms_list = list()
        self.non_vars_syms = OrderedDict()  # input_syms - vars_syms
        self.non_iter_syms = OrderedDict()  # input_syms - iter_syms

        self.vars_print = list()
        self.f_syms, self.g_syms = list(), list()
        self.f_matrix, self.g_matrix, self.s_matrix = list(), list(), list()
        self.f_print, self.g_print, self.s_print = list(), list(), list()
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
        self._input_z = OrderedDict()

    def __setattr__(self, key, value):
        if isinstance(value, (BaseVar, BaseService, Discrete, Block)):
            if not value.owner:
                value.owner = self
            if not value.name:
                value.name = key
            if not value.tex_name:
                value.tex_name = key
            if key in self.__dict__:
                logger.warning(f"{self.class_name}: redefinition of member <{key}>")

        # store the variable declaration order
        if isinstance(value, BaseVar):
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
        elif isinstance(value, ExtParam):
            self.params_ext[key] = value
        elif isinstance(value, Discrete):
            self.discrete[key] = value
        elif isinstance(value, ConstService):
            # only services with `v_str`
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

        super(Model, self).__setattr__(key, value)

    def idx2uid(self, idx):
        if idx is None:
            logger.error("idx2uid cannot search for None idx")
            return None
        if isinstance(idx, (float, int, str, np.int32, np.int64, np.float64)):
            return self.uid[idx]
        elif isinstance(idx, (list, np.ndarray)):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self.uid[i] for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def get(self, src: str, idx, attr: str = 'v'):
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
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_var()
            instance.set_var()

    def l_check_eq(self):
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.check_eq()

    def l_set_eq(self):
        if self.n == 0:
            return
        for instance in self.discrete.values():
            instance.set_eq()

    def s_update(self):
        """
        Evaluate service equations.

        This function is only evaluated at initialization.
        Service values are updated sequentially.
        The ``v`` attribute of services will change once.
        """
        if self.n == 0:
            return

        if self.calls.s_lambdify is not None and len(self.calls.s_lambdify):
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
            if func is not None and callable(func):
                kwargs = self.get_inputs(refresh=True)
                instance.v = func(**kwargs)

        kwargs = self.get_inputs(refresh=True)
        self.s_numeric(**kwargs)

    def generate_initializers(self):
        """
        Generate lambda functions for initial values
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
        self.calls.init_std = lambdify((list(self.iter_syms),
                                        list(self.non_iter_syms)),
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
            self.vars_syms[var] = Symbol(var)
            self.input_syms[var] = Symbol(var)
            if self.__dict__[var].v_iter is not None:
                self.iter_syms[var] = Symbol(var)

        for key in self.config.as_dict():
            self.input_syms[key] = Symbol(key)

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
        logger.debug(f'Generating Jacobians for {self.__class__.__name__}')
        from sympy import SparseMatrix, lambdify, Matrix

        # clear storage
        self.df_syms, self.dg_syms = Matrix([]), Matrix([])

        self.calls._ifx, self.calls._jfx, self.calls._vfx = list(), list(), list()
        self.calls._ify, self.calls._jfy, self.calls._vfy = list(), list(), list()
        self.calls._igx, self.calls._jgx, self.calls._vgx = list(), list(), list()
        self.calls._igy, self.calls._jgy, self.calls._vgy = list(), list(), list()

        self.calls._ifxc, self.calls._jfxc, self.calls._vfxc = list(), list(), list()
        self.calls._ifyc, self.calls._jfyc, self.calls._vfyc = list(), list(), list()
        self.calls._igxc, self.calls._jgxc, self.calls._vgxc = list(), list(), list()
        self.calls._igyc, self.calls._jgyc, self.calls._vgyc = list(), list(), list()

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

                jac_name = f'{eqn.e_code}{var.v_code}'

                self.calls.__dict__[f'_i{jac_name}'].append(e_idx)
                self.calls.__dict__[f'_j{jac_name}'].append(v_idx)
                self.calls.__dict__[f'_v{jac_name}'].append(lambdify(syms_list, e_symbolic))

        # The for loop below is intended to add an epsilon small value to the diagonal of gy matrix.
        # The user should take care of the algebraic equations by using `diag_eps` in Algeb definition

        for var in self.algebs.values():
            if var.diag_eps == 0.0:
                continue
            e_idx = algebs_and_ext_list.index(var.name)
            v_idx = vars_syms_list.index(var.name)
            self.calls.__dict__[f'_igyc'].append(e_idx)
            self.calls.__dict__[f'_jgyc'].append(v_idx)
            self.calls.__dict__[f'_vgyc'].append(var.diag_eps)

    def generate_pretty_print(self):
        """Generate pretty print variables and equations"""
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
            # if `self.n` is 0, skipping the processes below will avoid appending empty lists/arrays and
            # non-empty values, which, as a combination, is not accepted by `cvxopt.spmatrix`
            #
            # If we don't want to check `self.n`, we can check if len(row) == 0 or len(col) == 0 below instead.
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
                        logger.error(f'row {row_name}, row_idx: {row_idx}')
                        logger.error(f'col {col_name}, col_idx: {col_idx}')
                        raise ValueError(f'Model {self.class_name} has non-matching row_idx and col_idx')
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
                        logger.error(f'row_idx: {row}')
                        logger.error(f'col_idx: {col}')
                        raise ValueError(f'Model {self.class_name} has non-matching row_idx and col_idx')
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
        """
        Initialization sequence:
        1. Sequential initialization based on the order of definition
        2. Use Newton-Krylov method for iterative initialization
        3. Custom init
        """
        if self.n == 0:
            return

        # update service values
        self.s_update()

        logger.debug(f'{self.class_name}: calling initialize()')

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

        # call custom variable initializer after lambdified initializers
        kwargs = self.get_inputs(refresh=True)
        self.v_numeric(**kwargs)

        self.flags['initialized'] = True

    def get_init_order(self):
        """
        Get variable initialization order sent to logger.info.

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

        # update equations for algebraic variables supplied with `f_numeric`
        # evaluate numerical function calls
        kwargs = self.get_inputs()

        # call lambdified functions with use self.call
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

        # update equations for algebraic variables supplied with `g_numeric`
        # evaluate numerical function calls
        kwargs = self.get_inputs()

        # call lambdified functions with use self.call
        ret = self.calls.g_lambdify(**kwargs)

        for idx, instance in enumerate(self.cache.algebs_and_ext.values()):
            instance.e += ret[idx][0]

        # numerical calls defined in the model
        self.g_numeric(**kwargs)

        # numerical calls in blocks
        for instance in self.blocks.values():
            instance.g_numeric(**kwargs)

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
            for instance in self.timer_params.values():
                out.append(instance.v)

        return out

    def switch_action(self, dae_t):
        if self.n == 0:
            return

        for timer in self.timer_params.values():
            if timer.callback is not None:
                timer.callback(timer.is_time(dae_t))

        # TODO: consider `Block` with timer

    @property
    def class_name(self):
        """Return class name"""
        return self.__class__.__name__

    def _all_vars(self):
        """An OrderedDict of States, ExtStates, Algebs, ExtAlgebs"""
        return OrderedDict(list(self.states.items()) +
                           list(self.states_ext.items()) +
                           list(self.algebs.items()) +
                           list(self.algebs_ext.items())
                           )

    def _iter_vars(self):
        """Variables to be iteratively initialized"""
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
        """Custom service value functions. Modify ``Service.v`` directly."""
        pass

    def j_numeric(self, **kwargs):
        """
        Custom numeric update functions.

        This function should append indices to `_ifx`, `_jfx`, and `_vfx`.
        It is only called once in `store_sparse_pattern`.
        """
        pass

    def _param_doc(self, max_width=80, export='plain'):
        """
        Export formatted model parameter documentation as a string.

        Returns
        -------
        str
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

            plist = []
            for key, val in p.property.items():
                if val is True:
                    plist.append(key)
            properties.append(','.join(plist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.params.values()]
            symbols = self.math_wrap(symbols, export=export)
            units_rest = [f'*{item.unit}*' if item.unit else '' for item in self.params.values()]
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
        return self._make_doc_table(title='Parameters',
                                    max_width=max_width,
                                    export=export,
                                    plain_dict=plain_dict,
                                    rest_dict=rest_dict)

    @staticmethod
    def _make_doc_table(title, max_width, export, plain_dict, rest_dict):
        data_dict = rest_dict if export == 'rest' else plain_dict
        table = Tab(title=title, max_width=max_width, export=export)
        table.header(list(data_dict.keys()))
        rows = list(map(list, zip(*list(data_dict.values()))))
        table.add_rows(rows, header=False)

        return table.draw()

    @staticmethod
    def math_wrap(tex_str_list, export):
        """
        Warp each string item in a list with latex math environment ``$...$``.

        Parameters
        ----------
        export : str, ('rest', 'plain')
            Export format. Only wrap equations if export format is ``rest``.
        """
        if export != 'rest':
            return list(tex_str_list)

        out = []
        for item in tex_str_list:
            if item is None or item == '':
                out.append('')
            else:
                out.append(rf':math:`{item}`')
        return out

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
            symbols = self.math_wrap(call_store.x_latex + call_store.y_latex, export=export)
            ivs_rest = self.math_wrap(call_store.init_latex.values(), export=export)
            units_rest = [f'*{item.unit}*' if item.unit else '' for item in self.cache.all_vars.values()]

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

        return self._make_doc_table(title='Variables',
                                    max_width=max_width,
                                    export=export,
                                    plain_dict=plain_dict,
                                    rest_dict=rest_dict)

    def _eq_doc(self, max_width=80, export='plain'):
        # equation documentation
        if len(self.cache.all_vars) == 0:
            return ''
        names, symbols = list(), list()
        eqs, eqs_rest, class_names = list(), list(), list()

        for p in self.cache.all_vars.values():
            names.append(p.name)
            class_names.append(p.class_name)
            eqs.append(p.e_str if p.e_str else '')

        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = self.math_wrap(call_store.x_latex + call_store.y_latex, export=export)
            eqs_rest = self.math_wrap(call_store.f_latex + call_store.g_latex, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Equation (x\'=f or g=0)', eqs),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Equation (x\'=f or g=0)', eqs_rest),
                                 ('Type', class_names)])

        return self._make_doc_table(title='Equations',
                                    max_width=max_width,
                                    export=export,
                                    plain_dict=plain_dict,
                                    rest_dict=rest_dict)

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
            symbols = self.math_wrap([item.tex_name for item in self.services.values()], export=export)
            eqs_rest = self.math_wrap(call_store.s_latex, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Equation', eqs),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Equation', eqs_rest),
                                 ('Type', class_names)])

        return self._make_doc_table(title='Services',
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
            symbols = self.math_wrap([item.tex_name for item in self.discrete.values()], export=export)
        plain_dict = OrderedDict([('Name', names),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', class_names)])

        return self._make_doc_table(title='Discrete',
                                    max_width=max_width,
                                    export=export,
                                    plain_dict=plain_dict,
                                    rest_dict=rest_dict)

    def _block_doc(self, max_width=80, export='plain'):
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
