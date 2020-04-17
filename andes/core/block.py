from andes.core.var import Algeb, State
from typing import Optional, Iterable
from andes.core.discrete import AntiWindupLimiter
from andes.core.service import ConstService
from andes.core.triplet import JacTriplet


class DummyValues(object):
    def __init__(self, name):
        self.name = name
        self.tex_name = name


class Block(object):
    r"""
    Base class for control blocks.

    Blocks are meant to be instantiated as Model attributes to provide pre-defined equation sets. Subclasses
    must overload the `__init__` method to take custom inputs.
    Subclasses of Block must overload the `define` method to provide initialization and equation strings.
    Exported variables, services and blocks must be constructed into a dictionary ``self.vars`` at the end of
    the constructor.

    Blocks can be nested. A block can have blocks but itself as attributes and therefore reuse equations. When a
    block has sub-blocks, the outer block must be constructed with a``name``.

    Nested block works in the following way: the parent block modifies the sub-block's ``name`` attribute by
    prepending the parent block's name at the construction phase. The parent block then exports the sub-block
    as a whole. When the parent Model class picks up the block, it will recursively import the variables in the
    block and the sub-blocks correctly. See the example section for details.

    Parameters
    ----------
    name : str, optional
        Block name
    tex_name : str, optional
        Block LaTeX name
    info : str, optional
        Block description.

    Warnings
    --------
    It is suggested to have at most one level of nesting to avoid messy variable names.

    Examples
    --------
    Example for two-level nested blocks. Suppose we have the following hierarchy ::

        SomeModel  instance M
           |
        LeadLag A  exports (x, y)
           |
        Lag B      exports (x, y)

    SomeModel instance M contains an instance of Leadlag block named A, which contains an instance of a Lag block
    named B. Both A and B exports two variables ``x`` and ``y``.

    In the code of Model, the following code is used to instantiate LeadLag ::

        class SomeModel:
            def __init__(...)
                ...
                self.A = LeadLag(name='A',
                                 u=self.foo1,
                                 T1=self.foo2,
                                 T2=self.foo3)

    To use Lag in the LeadLag code, the following lines are found in the constructor of LeadLag ::

        class LeadLag:
            def __init__(name, ...)
                ...
                self.B = Lag(u=self.y, K=self.K, T=self.T)
                self.vars = {..., 'A': self.A}

    The ``__setattr__`` magic of LeadLag takes over the construction and assigns ``B.name`` as ``A_B``,
    given A's name provided at run time. ``self.A`` is exported with the internal name ``A`` at the end.

    Again, the LeadLag instance name (``A`` in this example) MUST be provided in SomeModel's constructor for the
    name prepending to work correctly. If there is more than one level of nesting, other than the leaf-level
    block, all parent blocks' names must be provided at instantiation.

    When A is picked up by ``SomeModel.__setattr__``, B is captured from A's exports. Recursively, B's variables
    are exported, Recall that ``B.name`` is ``A_B``, following the naming rule (parent block's name + variable
    name), B's internal variables become ``A_B_x`` and ``A_B_y``.

    In this way, B's ``define`` needs no modification since the naming rule is the same. For example,
    B's internal y is always ``{self.name}_y``, although B has gotten a new name ``A_B``.

    """

    def __init__(self, name: Optional[str] = None, tex_name: Optional[str] = None, info: Optional[str] = None):
        self.name = name
        self.tex_name = tex_name if tex_name else name
        self.info = info
        self.owner = None
        self.vars: dict = dict()
        self.triplets = JacTriplet()

    def __setattr__(self, key, value):
        # handle sub-blocks by prepending self.name
        if isinstance(value, Block):
            if self.name is None:
                raise ValueError(f"`name` must be specified when constructing {self.class_name} instances.")
            if not value.owner:
                value.__dict__['owner'] = self

            if not value.name:
                value.__dict__['name'] = self.name + '_' + key
            else:
                value.__dict__['name'] = self.name + '_' + value.name

            if not value.tex_name:
                value.__dict__['tex_name'] = self.name + r'\ ' + key
            else:
                value.__dict__['tex_name'] = self.name + r'\ ' + value.tex_name

        self.__dict__[key] = value

    def j_reset(self):
        """
        Helper function to clear the lists holding the numerical Jacobians.

        This function should be only called once at the beginning of ``j_numeric`` in blocks.
        """
        self.triplets.clear_ijv()

    def define(self):
        """
        Function for setting the initialization and equation strings for internal variables. This method must be
        implemented by subclasses.

        The equations should be written with the "final" variable names.
        Let's say the block instance is named `blk` (kept at ``self.name`` of the block), and an internal
        variable `v` is defined.
        The internal variable will be captured as ``blk_v`` by the parent model. Therefore, all equations should
        use ``{self.name}_v`` to represent variable ``v``, where ``{self.name}`` is the name of the block at
        run time.

        On the other hand, the names of externally provided parameters or variables are obtained by
        directly accessing the ``name`` attribute. For example, if ``self.T`` is a parameter provided through
        the block constructor, ``{self.T.name}`` should be used in the equation.

        Examples
        --------
        An internal variable ``v`` has a trivial equation ``T = v``, where T is a parameter provided to the block
        constructor.

        In the model, one has ::

            class SomeModel():
                def __init__(...)
                    self.input = Algeb()
                    self.T = Param()

                    self.blk = ExampleBlock(u=self.input, T=self.T)

        In the ExampleBlock function, the internal variable is defined in the constructor as ::

            class ExampleBlock():
                def __init__(...):
                    self.v = Algeb()
                    self.vars = {'v', self.v}

        In the ``define``, the equation is provided as ::

            def define(self):
                self.v.v_str = '{self.T.name}'
                self.v.e_str = '{self.T.name} - {self.name}_v'

        In the parent model, ``v`` from the block will be captured as ``blk_v``, and the equation will
        evaluate into ::

            self.blk_v.v_str = 'T'
            self.blk_v.e_str = 'T - blk_v'

        See Also
        --------
        PIController.define : Equations for the PI Controller block

        """
        raise NotImplementedError(f'define() method not implemented in {self.class_name}')

    def export(self):
        """
        Method for exporting instances defined in this class in a dictionary. This method calls the ``define``
        method first and returns ``self.vars``.

        Returns
        -------
        dict
            Keys are the (last section of the) variable name, and the values are the attribute instance.
        """
        self.define()
        return self.vars

    def g_numeric(self, **kwargs):
        """
        Function call to update algebraic equation values.

        This function should modify the ``e`` value of block ``Algeb`` and ``ExtAlgeb`` in place.
        """
        pass

    def f_numeric(self, **kwargs):
        """
        Function call to update differential equation values.

        This function should modify the ``e`` value of block ``State`` and ``ExtState`` in place.
        """
        pass

    def j_numeric(self):
        """
        This function stores the constant and variable jacobian information in corresponding lists.

        Constant jacobians are stored by indices and values in, for example, `ifxc`, `jfxc` and `vfxc`.
        Value scalars or arrays are stored in `vfxc`.

        Variable jacobians are stored by indices and functions. The function shall return the value of the
        corresponding jacobian elements.
        """
        pass

    @property
    def class_name(self):
        """Return the class name."""
        return self.__class__.__name__

    @staticmethod
    def enforce_tex_name(fields):
        """Enforce tex_name is not None"""
        if not isinstance(fields, Iterable):
            fields = [fields]

        for field in fields:
            if field.tex_name is None:
                raise NameError(f'tex_name for <{field.name}> cannot be None')


class PIController(Block):
    """
    Proportional Integral Controller with the reference from an external variable

    Parameters
    ----------
    u : BaseVar
        The input variable instance
    ref : Union[BaseVar, BaseParam]
        The reference instance
    kp : BaseParam
        The proportional gain parameter instance
    ki : [type]
        The integral gain parameter instance

    """
    def __init__(self, u, ref, kp, ki, name=None, info=None):
        super(PIController, self).__init__(name=name, info=info)

        self.u = u
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller")
        self.y = Algeb(info="Output value")

        self.vars = {'xi': self.xi, 'y': self.y}

    def define(self):
        r"""
        Define equations for the PI Controller.

        Notes
        -----
        One state variable ``xi`` and one algebraic variable ``y`` are added.

        Equations implemented are

        .. math ::
            \dot{x_i} = k_i * (ref - var)

            y = x_i + k_i * (ref - var)
        """

        self.xi.e_str = f'ki * ({self.ref.name} - {self.u.name})'
        self.y.e_str = f'kp * ({self.ref.name} - {self.u.name}) + {self.name}_xi'


class PIControllerNumeric(Block):
    """A PI Controller implemented with numerical function calls"""

    def __init__(self, u, ref, kp, ki, name=None, info=None):
        super().__init__(name=name, info=info)

        self.u = u
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller")
        self.y = Algeb(info="Output value")

        self.vars = {'xi': self.xi, 'y': self.y}

    def g_numeric(self, **kwargs):
        self.y.e = self.kp.v * (self.ref.v - self.u.v) + self.xi.v

    def f_numeric(self, **kwargs):
        self.xi.e = self.ki.v * (self.ref.v - self.u.v)

    def j_numeric(self):
        self.j_reset()

        # TODO: test the following code
        self.triplets.append_ijv('fyc', self.xi.id, self.u.id, -self.ki.v)
        self.triplets.append_ijv('gyc', self.y.id, self.u.id, -self.kp.v)
        self.triplets.append_ijv('gxc', self.y.id, self.xi.id, 1)

    def define(self):
        """Skip the symbolic definition"""
        pass


class Washout(Block):
    r"""
    Washout filter (high pass) block ::

                sK
         u -> ------- -> y
              1 + sT

    """

    def __init__(self, u, T, K, info=None, name=None):
        super().__init__(name=name, info=info)
        if isinstance(T, (int, float)):
            self.T = DummyValues(T)
        else:
            self.T = T

        if isinstance(K, (int, float)):
            self.K = DummyValues(K)
        else:
            self.K = K

        self.enforce_tex_name((self.K, self.T))

        self.KT = ConstService(info='Constant K/T',
                               tex_name=f'({self.K.tex_name}/{self.T.tex_name})',
                               v_str=f'{self.K.name} / {self.T.name}')

        self.u = u
        self.x = State(info='State in washout filter', tex_name="x'", t_const=self.T)
        self.y = Algeb(info='Output of washout filter', tex_name=r'y')
        self.vars = {'KT': self.KT, 'x': self.x, 'y': self.y}

    def define(self):
        r"""
        Notes
        -----
        Equations and initial values:

        .. math ::
            \dot{x'} = (u - x) / T \\
            y = u - x \\
            x'_0 = u \\
            y_0 = 0

        """
        self.x.v_str = f'{self.u.name}'
        self.y.v_str = f'0'

        self.x.e_str = f'({self.u.name} - {self.name}_x)'
        self.y.e_str = f'{self.name}_KT * ({self.u.name} - {self.name}_x) - {self.name}_y'


class Lag(Block):
    r"""
    Lag (low pass) transfer function block ::

                K
        u -> ------ -> y
             1 + sT

    Exports one state variable `x` as the output.

    Parameters
    ----------
    K
        Gain
    T
        Time constant
    u
        Input variable

    """
    def __init__(self, u, T, K, name=None, info='Lag transfer function'):
        super().__init__(name=name, info=info)
        self.u = u
        self.T = T
        if isinstance(K, (int, float)):
            self.K = DummyValues(K)
        else:
            self.K = K

        self.enforce_tex_name((self.K, self.T))
        self.x = State(info='State in lag transfer function', tex_name="x'",
                       t_const=self.T)

        self.vars = {'x': self.x}

    def define(self):
        r"""

        Notes
        -----
        Equation and initial value

        .. math ::

            \dot{x'} = (u - x) / T

            x'_0 = u

        """
        self.x.v_str = f'{self.u.name} * {self.K.name}'
        self.x.e_str = f'({self.K.name} * {self.u.name} - {self.name}_x)'


class LagAntiWindup(Block):
    r"""
    Lag (low pass) transfer function block with anti-windup limiter ::

                K
        u -> ------ -> y
             1 + sT

    Exports one state variable `x` as the output.

    Parameters
    ----------
    K
        Gain
    T
        Time constant
    u
        Input variable

    """
    def __init__(self, u, T, K, lower, upper, name=None,
                 info='Lag transfer function with non-windup limiter'):
        super().__init__(name=name, info=info)
        if isinstance(K, (int, float)):
            self.K = DummyValues(K)
        else:
            self.K = K
        self.u = u
        self.T = T
        self.lower = lower
        self.upper = upper

        self.enforce_tex_name((self.T, self.K))

        self.x = State(info='State in lag transfer function', tex_name="x'",
                       t_const=self.T)
        self.lim = AntiWindupLimiter(u=self.x, lower=self.lower, upper=self.upper, tex_name='lim')

        self.vars = {'x': self.x, 'lim': self.lim}

    def define(self):
        r"""

        Notes
        -----
        Equation and initial value

        .. math ::

            \dot{x'} = (u - x) / T

            x'_0 = u

        """
        self.x.v_str = f'{self.u.name} * {self.K.name}'
        self.x.e_str = f'{self.name}_lim_zi * ({self.K.name} * {self.u.name} - {self.name}_x)'


class LeadLag(Block):
    r"""
    Lead-Lag transfer function block in series implementation ::

             1 + sT1
        u -> ------- -> y
             1 + sT2

    Exports two variables: state x and output y.

    Parameters
    ----------
    T1
        Time constant 1
    T2
        Time constant 2

    Notes
    -----
    Future implementation will allow T2 to be zero with a safe division. Any division by zero will become zero.

    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    """
    def __init__(self, u, T1, T2, name=None, info='Lead-lag transfer function', safe_div=True):
        super().__init__(name=name, info=info)
        self.T1 = T1
        self.T2 = T2
        self.u = u
        self.safe_div = safe_div  # TODO: implement me

        self.enforce_tex_name((self.T1, self.T2))

        self.x = State(info='State in lead-lag transfer function', tex_name="x'", t_const=self.T2)
        self.y = Algeb(info='Output of lead-lag transfer function', tex_name=r'y', diag_eps=1e-6)
        self.vars = {'x': self.x, 'y': self.y}

    def define(self):
        r"""

        Notes
        -----

        Implemented equations and initial values

        .. math ::

            T_2 \dot{x'} = (u - x') \\
            T_2 * y = T_1 * (u - x') + T_2 * x' \\
            x'_0 = y_0 = u

        """
        self.x.v_str = f'{self.u.name}'
        self.y.v_str = f'{self.u.name}'

        self.x.e_str = f'({self.u.name} - {self.name}_x)'
        self.y.e_str = f'{self.T1.name} * ({self.u.name} - {self.name}_x) + ' \
                       f'{self.name}_x * {self.T2.name} - ' \
                       f'{self.name}_y * {self.T2.name}'


class LeadLagLimit(Block):
    r"""
    Lead-Lag transfer function block with hard limiter (series implementation) ::

                       ___upper___
                      /
                  1 + sT1
           u ->   -------  -> y
                  1 + sT2
        __lower____/

    Exports four variables: state `x`, output before hard limiter `ynl`, output `y`, and limiter `lim`,

    """
    def __init__(self, u, T1, T2, lower, upper, name=None, info='Lead-lag transfer function'):
        super().__init__(name=name, info=info)
        self.T1 = T1
        self.T2 = T2
        self.u = u
        self.lower = lower
        self.upper = upper
        self.enforce_tex_name((self.T1, self.T2))

        self.x = State(info='State in lead-lag transfer function', tex_name="x'", t_const=self.T2)
        self.ynl = Algeb(info='Output of lead-lag transfer function before limiter', tex_name=r'y_{nl}')
        self.y = Algeb(info='Output of lead-lag transfer function after limiter', tex_name=r'y',
                       diag_eps=1e-6)
        self.lim = AntiWindupLimiter(u=self.ynl, lower=self.lower, upper=self.upper)

        self.vars = {'x': self.x, 'ynl': self.ynl, 'y': self.y, 'lim': self.lim}

    def define(self):
        r"""

        Notes
        -----

        Implemented equations and initial values

        .. math ::

            T_2 \dot{x'} = (u - x') \\
            T_2 y = T_1 * (u - x') + T_2 * x' \\
            x'_0 = y_0 = u

        """
        self.x.v_str = f'{self.u.name}'
        self.ynl.v_str = f'{self.u.name}'
        self.y.v_str = f'{self.u.name}'

        self.x.e_str = f'({self.u.name} - {self.name}_x)'
        self.ynl.e_str = f'{self.T1.name} * ({self.u.name} - {self.name}_x) + ' \
                         f'{self.name}_x * {self.T2.name} - ' \
                         f'{self.name}_ynl * {self.T2.name}'

        self.y.e_str = f'{self.name}_ynl * {self.name}_lim_zi + ' \
                       f'{self.lower.name} * {self.name}_lim_zl + ' \
                       f'{self.upper.name} * {self.name}_lim_zu - ' \
                       f'{self.name}_y'


class Piecewise(Block):
    """
    Piecewise block.

    This block takes a list of N points, [x0, x1, ...x_{n-1}] and a list of N+1 functions [fun0, ..., fun_n].
    Inputs in each range (xk, x_{k+1}] applies its corresponding function `fun_k`. The last range (x_{n-1},
    +inf) applies the last function `fun_n`.

    Parameters
    ----------
    points : list
        A list of piecewise points. Need to be provided in the constructor function.
    funs : list
        A list of strings for the piecewise functions. Need to be provided in the overloaded `define` function.
    """
    def __init__(self, u, points: list, funs: list, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = u
        self.points = points
        self.funs = funs

        self.y = Algeb(info='Output of piecewise function', tex_name='y')
        self.vars = {'y': self.y}

    def define(self):
        """
        Build the equation string for the piecewise equations.

        ``self.funs`` needs to be provided with the function strings corresponding to each range.
        """
        args = []
        i = 0
        for i in range(len(self.points)):
            args.append(f'({self.funs[i]}, {self.u.name} <= {self.points[i]})')
        args.append(f'({self.funs[i+1]}, True)')

        args_comma = ', '.join(args)
        pw_fun = f'Piecewise({args_comma})'

        self.y.v_str = pw_fun
        self.y.e_str = f'{pw_fun} - {self.name}_y'
