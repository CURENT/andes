"""
Symbolic processor class for ANDES models.
"""

import logging
import os
import numpy as np
from collections import OrderedDict, defaultdict

from andes.utils.paths import get_dot_andes_path

logger = logging.getLogger(__name__)


class SymProcessor:
    """
    A helper class for symbolic processing and code generation.

    Parameters
    ----------
    parent : Model
        The `Model` instance to document

    Attributes
    ----------
    xy : sympy.Matrix
        variables pretty print in the order of State, ExtState, Algeb, ExtAlgeb
    f : sympy.Matrix
        differential equations pretty print
    g : sympy.Matrix
        algebraic equations pretty print
    df : sympy.SparseMatrix
        df /d (xy) pretty print
    dg : sympy.SparseMatrix
        dg /d (xy) pretty print
    inputs_dict : OrderedDict
        All possible symbols in equations, including variables, parameters, discrete flags, and
        config flags. It has the same variables as what ``get_inputs()`` returns.
    vars_dict : OrderedDict
        variable-only symbols, which are useful when getting the Jacobian matrices.
    non_vars_dict : OrderedDict
        symbols in ``input_syms`` but not in ``var_syms``.

    """

    def __init__(self, parent):

        self.parent = parent
        # symbols that are input to lambda functions
        # including parameters, variables, services, configs, and scalars (dae_t, sys_f, sys_mva)
        self.inputs_dict = OrderedDict()
        self.lambdify_func = [dict(), 'numpy']

        self.vars_dict = OrderedDict()
        self.iters_dict = OrderedDict()
        self.non_vars_dict = OrderedDict()  # inputs_dict - vars_dict
        self.non_iters_dict = OrderedDict()  # inputs_dict - iters_dict
        self.vars_list = list()

        self.f_list, self.g_list = list(), list()  # symbolic equations in lists
        self.f_matrix, self.g_matrix, self.s_matrix = list(), list(), list()  # equations in matrices

        # pretty print of variables
        self.xy = list()  # variables in the order of states, algebs
        self.f, self.g, self.s = list(), list(), list()
        self.df, self.dg = None, None

        # get references to the parent attributes
        self.calls = parent.calls
        self.cache = parent.cache
        self.config = parent.config
        self.class_name = parent.class_name
        self.tex_names = OrderedDict()

    def generate_init(self):
        """
        Generate lambda functions for initial values.
        """
        logger.debug(f'- Generating initializers for {self.class_name}')
        from sympy import sympify, lambdify, Matrix
        from sympy.printing import latex

        init_lambda_list = OrderedDict()
        init_latex = OrderedDict()
        init_seq_list = []
        init_g_list = []  # initialization equations in g(x, y) = 0 form

        input_syms_list = list(self.inputs_dict)

        for name, instance in self.cache.all_vars.items():
            if instance.v_str is None and instance.v_iter is None:
                init_latex[name] = ''
            else:
                if instance.v_str is not None:
                    sympified = sympify(instance.v_str, locals=self.inputs_dict)
                    self._check_expr_symbols(sympified)
                    lambdified = lambdify(input_syms_list, sympified, modules=self.lambdify_func)
                    init_lambda_list[name] = lambdified
                    init_latex[name] = latex(sympified.subs(self.tex_names))
                    init_seq_list.append(sympify(f'{instance.v_str} - {name}', locals=self.inputs_dict))

                if instance.v_iter is not None:
                    sympified = sympify(instance.v_iter, locals=self.inputs_dict)
                    self._check_expr_symbols(sympified)
                    init_g_list.append(sympified)
                    init_latex[name] = latex(sympified.subs(self.tex_names))

        self.init_seq = Matrix(init_seq_list)
        self.init_std = Matrix(init_g_list)
        self.init_dstd = Matrix([])
        if len(self.init_std) > 0:
            self.init_dstd = self.init_std.jacobian(list(self.vars_dict.values()))

        self.calls.init = init_lambda_list
        self.calls.init_latex = init_latex
        self.calls.init_std = lambdify((list(self.iters_dict), list(self.non_iters_dict)),
                                       self.init_std,
                                       modules=self.lambdify_func)

    def generate_symbols(self):
        """
        Generate symbols for symbolic equation generations.

        This function should run before other generate equations.

        Attributes
        ----------
        inputs_dict : OrderedDict
            name-symbol pair of all parameters, variables and configs

        vars_dict : OrderedDict
            name-symbol pair of all variables, in the order of (states_and_ext + algebs_and_ext)

        non_vars_dict : OrderedDict
            name-symbol pair of all non-variables, namely, (inputs_dict - vars_dict)

        """

        logger.debug(f'- Generating symbols for {self.class_name}')
        from sympy import Symbol, Matrix

        # clear symbols storage
        self.f_list, self.g_list = list(), list()
        self.f_matrix, self.g_matrix = Matrix([]), Matrix([])

        # process tex_names defined in model
        # -----------------------------------------------------------
        for key in self.parent.tex_names.keys():
            self.tex_names[key] = Symbol(self.parent.tex_names[key])
        for instance in self.parent.discrete.values():
            for name, tex_name in zip(instance.get_names(), instance.get_tex_names()):
                self.tex_names[name] = tex_name
        # -----------------------------------------------------------

        for var in self.cache.all_params_names:
            self.inputs_dict[var] = Symbol(var)

        for var in self.cache.all_vars_names:
            tmp = Symbol(var)
            self.vars_dict[var] = tmp
            self.inputs_dict[var] = tmp
            if self.parent.__dict__[var].v_iter is not None:
                self.iters_dict[var] = tmp

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = Symbol(key)
            self.inputs_dict[key] = tmp
            if key in self.config.tex_names:
                self.tex_names[tmp] = Symbol(self.config.tex_names[key])

        # store tex names for pretty printing replacement later
        for var in self.inputs_dict:
            if var in self.parent.__dict__ and self.parent.__dict__[var].tex_name is not None:
                self.tex_names[Symbol(var)] = Symbol(self.parent.__dict__[var].tex_name)

        self.inputs_dict['dae_t'] = Symbol('dae_t')
        self.inputs_dict['sys_f'] = Symbol('sys_f')
        self.inputs_dict['sys_mva'] = Symbol('sys_mva')

        self.lambdify_func[0]['Indicator'] = lambda x: x
        self.lambdify_func[0]['imag'] = np.imag
        self.lambdify_func[0]['real'] = np.real

        # build ``non_vars_dict`` by removing ``vars_dict`` keys from a copy of ``inputs``
        self.non_vars_dict = OrderedDict(self.inputs_dict)
        self.non_iters_dict = OrderedDict(self.inputs_dict)
        for key in self.vars_dict:
            self.non_vars_dict.pop(key)
        for key in self.iters_dict:
            self.non_iters_dict.pop(key)

        self.vars_list = list(self.vars_dict.values())  # useful for ``.jacobian()``

    def _check_expr_symbols(self, expr):
        """
        Check if expression contains unknown symbols.
        """
        fs = expr.free_symbols
        for item in fs:
            if item not in self.inputs_dict.values():
                raise ValueError(f'{self.class_name} expression "{expr}" contains unknown symbol "{item}"')

        return fs

    def generate_equations(self):
        logger.debug(f'- Generating equations for {self.class_name}')
        from sympy import Matrix, sympify, lambdify, SympifyError

        self.f_list, self.g_list = list(), list()

        self.calls.f = None
        self.calls.g = None
        self.calls.f_args = list()
        self.calls.g_args = list()

        vars_list = [self.cache.states_and_ext, self.cache.algebs_and_ext]
        expr_list = [self.f_list, self.g_list]

        eqn_names = ['f', 'g']
        eqn_args = [self.calls.f_args, self.calls.g_args]

        for vlist, elist, ename, eargs in zip(vars_list, expr_list, eqn_names, eqn_args):
            sym_args = list()
            for name, instance in vlist.items():
                if instance.e_str is None:
                    elist.append(0)
                else:
                    try:
                        expr = sympify(instance.e_str, locals=self.inputs_dict)
                    except SympifyError as e:
                        logger.error('Error parsing equation "%s "for %s.%s',
                                     instance.e_str, instance.owner.class_name, name)
                        raise e
                    except TypeError as e:
                        logger.error('Error parsing equation "%s "for %s.%s',
                                     instance.e_str, instance.owner.class_name, name)
                        raise e

                    free_syms = self._check_expr_symbols(expr)

                    for s in free_syms:
                        if s not in sym_args:
                            sym_args.append(s)
                            eargs.append(str(s))

                    elist.append(expr)
            if len(elist) == 0 or not any(elist):  # `any`, not `all`
                self.calls.__dict__[ename] = None
            else:
                self.calls.__dict__[ename] = lambdify(sym_args, tuple(elist),
                                                      modules=self.lambdify_func)

        # convert to SymPy matrices
        self.f_matrix = Matrix(self.f_list)
        self.g_matrix = Matrix(self.g_list)

    def generate_services(self):
        """
        Generate calls for services, including ``ConstService``, ``VarService`` among others.
        """
        from sympy import Matrix, sympify, lambdify, SympifyError

        # convert service equations
        # Service equations are converted sequentially due to possible dependency
        s_args = OrderedDict()
        s_syms = OrderedDict()
        s_calls = OrderedDict()

        for name, instance in self.parent.services.items():
            if instance.v_str is not None:
                try:
                    expr = sympify(instance.v_str, locals=self.inputs_dict)
                except (SympifyError, TypeError) as e:
                    logger.error(f'Error parsing equation for {instance.owner.class_name}.{name}')
                    raise e
                self._check_expr_symbols(expr)
                s_syms[name] = expr
                s_args[name] = [str(i) for i in expr.free_symbols]
                s_calls[name] = lambdify(s_args[name], s_syms[name], modules=self.lambdify_func)
            else:
                s_syms[name] = 0
                s_args[name] = []
                s_calls[name] = 0

        self.s_matrix = Matrix(list(s_syms.values()))
        self.calls.s = s_calls
        self.calls.s_args = s_args

    def generate_jacobians(self):
        """
        Generate Jacobians and store to corresponding triplets.

        The internal indices of equations and variables are stored, alongside the lambda functions.

        For example, dg/dy is a sparse matrix whose elements are ``(row, col, val)``, where ``row`` and ``col``
        are the internal indices, and ``val`` is the numerical lambda function. They will be stored to

            row -> self.calls._igy
            col -> self.calls._jgy
            val -> self.calls._vgy

        """
        logger.debug(f'- Generating Jacobians for {self.class_name}')

        from sympy import SparseMatrix, lambdify, Matrix

        # clear storage
        self.df_syms, self.dg_syms = Matrix([]), Matrix([])
        self.calls.clear_ijv()

        # NOTE: SymPy does not allow getting the derivative of an empty array
        if len(self.g_matrix) > 0:
            self.dg_syms = self.g_matrix.jacobian(self.vars_list)

        if len(self.f_matrix) > 0:
            self.df_syms = self.f_matrix.jacobian(self.vars_list)

        self.df_sparse = SparseMatrix(self.df_syms)
        self.dg_sparse = SparseMatrix(self.dg_syms)

        vars_syms_list = list(self.vars_dict)
        algebs_and_ext_list = list(self.cache.algebs_and_ext)
        states_and_ext_list = list(self.cache.states_and_ext)

        fg_sparse = [self.df_sparse, self.dg_sparse]
        j_args = defaultdict(list)   # argument list for each jacobian call
        j_calls = defaultdict(list)  # jacobian functions (one for each type)

        for idx, eq_sparse in enumerate(fg_sparse):
            for item in eq_sparse.row_list():
                e_idx, v_idx, e_symbolic = item
                if idx == 0:
                    eq_name = states_and_ext_list[e_idx]
                else:
                    eq_name = algebs_and_ext_list[e_idx]

                var_name = vars_syms_list[v_idx]
                eqn = self.cache.all_vars[eq_name]    # `BaseVar` that corr. to the equation
                var = self.cache.all_vars[var_name]   # `BaseVar` that corr. to the variable
                jname = f'{eqn.e_code}{var.v_code}'

                # jac calls with all arguments and stored individually
                self.calls.append_ijv(jname, e_idx, v_idx, 0)

                free_syms = self._check_expr_symbols(e_symbolic)
                for fs in free_syms:
                    if fs not in j_args[jname]:
                        j_args[jname].append(fs)
                # j_args[jname].extend(free_syms)
                j_calls[jname].append(e_symbolic)

        for jname in j_calls:
            self.calls.j_args[jname] = [str(i) for i in j_args[jname]]
            self.calls.j[jname] = lambdify(j_args[jname], tuple(j_calls[jname]), modules=self.lambdify_func)

        # The for loop below is intended to add an epsilon small value to the diagonal of `gy`.
        # The user should take care of the algebraic equations by using `diag_eps` in `Algeb` definition

        for var in self.parent.cache.vars_int.values():
            if var.diag_eps == 0.0:
                continue

            elif var.diag_eps is True:
                eps = self.parent.system.config.diag_eps

            else:
                eps = var.diag_eps

            if var.e_code == 'g':
                eq_list = algebs_and_ext_list
            else:
                eq_list = states_and_ext_list

            e_idx = eq_list.index(var.name)
            v_idx = vars_syms_list.index(var.name)

            self.calls.append_ijv(f'{var.e_code}{var.v_code}c', e_idx, v_idx, eps)

    def generate_pretty_print(self):
        """
        Generate pretty print variables and equations.
        """
        logger.debug(f"- Generating pretty prints for {self.class_name}")
        from sympy import Matrix
        from sympy.printing import latex

        # equation symbols for pretty printing
        self.f, self.g = Matrix([]), Matrix([])

        self.xy = Matrix(list(self.vars_dict.values())).subs(self.tex_names)

        # get pretty printing equations by substituting symbols
        self.f = self.f_matrix.subs(self.tex_names)
        self.g = self.g_matrix.subs(self.tex_names)
        self.s = self.s_matrix.subs(self.tex_names)

        # store latex strings
        nx = len(self.f)
        ny = len(self.g)
        self.calls.x_latex = [latex(item) for item in self.xy[:nx]]
        self.calls.y_latex = [latex(item) for item in self.xy[nx:nx + ny]]

        self.calls.f_latex = [latex(item) for item in self.f]
        self.calls.g_latex = [latex(item) for item in self.g]
        self.calls.s_latex = [latex(item) for item in self.s]

        self.df = self.df_sparse.subs(self.tex_names)
        self.dg = self.dg_sparse.subs(self.tex_names)

    def generate_pycode(self):
        """
        Create output source code file for generated code. NOT WORKING NOW.
        """
        models_dir = os.path.join(get_dot_andes_path(), 'pycode')
        os.makedirs(models_dir, exist_ok=True)
        file_path = os.path.join(models_dir, f'{self.class_name}.py')

        header = \
            """from numpy import nan, pi, sin, cos, tan, sqrt, exp, select  # NOQA
from numpy import greater_equal, less_equal, greater, less  # NOQA


"""

        with open(file_path, 'w') as f:
            f.write(header)
            f.write(self._rename_func(self.calls.f, 'f_update'))
            f.write(self._rename_func(self.calls.g, 'g_update'))

            for jname in self.calls.j:
                f.write(self._rename_func(self.calls.j[jname], f'{jname}_update'))

    def _rename_func(self, func, func_name):
        """
        Rename the function name and return source code.

        This function does not check for name conflicts.
        Install `yapf` for optional code reformatting (takes extra processing time).
        """
        import inspect

        if func is None:
            return f"# empty {func_name}\n"

        src = inspect.getsource(func)
        src = src.replace("def _lambdifygenerated(", f"def {func_name}(")
        # remove `Indicator`
        src = src.replace("Indicator", "")

        if self.parent.system.config.yapf_pycode:
            try:
                from yapf.yapflib.yapf_api import FormatCode
                src = FormatCode(src, style_config='pep8')[0]  # drop the encoding `None`
            except ImportError:
                logger.warning("`yapf` not installed. Skipped code reformatting.")

        src += '\n'
        return src
