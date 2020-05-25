from typing import Optional, Iterable, Union, List, Tuple

from andes.core.var import Algeb, State
from andes.core.discrete import AntiWindup, LessThan, Selector, HardLimiter
from andes.core.common import JacTriplet
from andes.core.common import ModelFlags, dummify
from collections import OrderedDict
import numpy as np


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
    It is a good practice to avoid more than one level of nesting, to avoid multi-underscore variable names.

    Examples
    --------
    Example for two-level nested blocks. Suppose we have the following hierarchy ::

        SomeModel  instance M
           |
        LeadLag A  exports (x, y)
           |
        Lag B      exports (x, y)

    SomeModel instance M contains an instance of LeadLag block named A, which contains an instance of a Lag block
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

    The ``__setattr__`` magic of LeadLag takes over the construction and assigns ``A_B`` to `B.name`,
    given A's name provided at run time. `self.A` is exported with the internal name ``A`` at the end.

    Again, the LeadLag instance name (`A` in this example) MUST be provided in `SomeModel`'s constructor for the
    name prepending to work correctly. If there is more than one level of nesting, other than the leaf-level
    block, all parent blocks' names must be provided at instantiation.

    When A is picked up by `SomeModel.__setattr__`, B is captured from A's exports. Recursively, B's variables
    are exported, Recall that `B.name` is now ``A_B``, following the naming rule (parent block's name + variable
    name), B's internal variables become ``A_B_x`` and ``A_B_y``.

    In this way, B's ``define()`` needs no modification since the naming rule is the same. For example,
    B's internal y is always ``{self.name}_y``, although B has gotten a new name ``A_B``.

    """

    def __init__(self, name: Optional[str] = None, tex_name: Optional[str] = None, info: Optional[str] = None):
        self.name = name
        self.tex_name = tex_name if tex_name else name
        self.info = info
        self.owner = None
        self.vars = OrderedDict()
        self.triplets = JacTriplet()
        self.flags = ModelFlags()  # f_num, g_num and j_num can be set

    def __setattr__(self, key, value):
        # handle sub-blocks by prepending self.name
        if isinstance(value, Block):
            if self.name is None:
                raise ValueError(f"Must specify `name` for {self.class_name} any instance.")
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
        """
        Enforce tex_name is not None
        """
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
            \dot{x_i} &= k_i * (ref - var) \\
            y &= x_i + k_i * (ref - var)
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
        self.flags.update({'f_num': True, 'g_num': True, 'j_num': True})

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


class Gain(Block):
    r"""
    Gain block. ::

             ┌───┐
        u -> │ K │ -> y
             └───┘

    Exports an algebraic output `y`.
    """

    def __init__(self, u, K, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.K = dummify(K)
        self.enforce_tex_name((self.K,))

        self.y = Algeb(info='Gain output', tex_name='y')
        self.vars = {'y': self.y}

    def define(self):
        r"""
        Implemented equation and the initial condition are

        .. math ::
            y = K u \\
            y^{(0)} = K u^{(0)}

        """
        self.y.v_str = f'{self.K.name} * {self.u.name}'
        self.y.e_str = f'{self.K.name} * {self.u.name} - {self.name}_y'


class Integrator(Block):
    r"""
    Integrator block. ::

             ┌──────┐
        u -> │ K/sT │ -> y
             └──────┘

    Exports a differential variable `y`.
    The initial output is specified by `y0` and default to zero.
    """

    def __init__(self, u, T, K, y0, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.K = dummify(K)
        self.T = dummify(T)
        self.y0 = dummify(y0)
        self.enforce_tex_name((self.K, self.T))

        self.y = State(info='Integrator output', tex_name='y', t_const=self.T)
        self.vars = {'y': self.y}

    def define(self):
        r"""
        Implemented equation and the initial condition are

        .. math ::
            \dot{y} = K u \\
            y^{(0)} = 0

        """
        self.y.v_str = f'{self.y0.name}'
        self.y.e_str = f'{self.K.name} * ({self.u.name})'


class IntegratorAntiWindup(Block):
    r"""
    Integrator block with anti-windup limiter. ::

                   upper
                  /¯¯¯¯¯
             ┌──────┐
        u -> │ K/sT │ -> y
             └──────┘
           _____/
           lower

    Exports a differential variable `y` and an AntiWindup `lim`.
    The initial output must be specified through `y0`.
    """

    def __init__(self, u, T, K, y0, lower, upper, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T = dummify(T)
        self.K = dummify(K)
        self.y0 = dummify(y0)
        self.lower = dummify(lower)
        self.upper = dummify(upper)
        self.enforce_tex_name((self.K, self.T))

        self.y = State(info='AW Integrator output', tex_name='y', t_const=self.T)

        self.lim = AntiWindup(u=self.y, lower=self.lower, upper=self.upper, tex_name='lim',
                              info='Limiter in integrator',
                              )

        self.vars = {'y': self.y, 'lim': self.lim}

    def define(self):
        r"""
        Implemented equation and the initial condition are

        .. math ::
            \dot{y} = K u \\
            y^{(0)} = 0

        """
        self.y.v_str = f'{self.y0.name}'
        self.y.e_str = f'{self.K.name} * ({self.u.name})'


class Washout(Block):
    r"""
    Washout filter (high pass) block. ::

             ┌────────┐
             │   sK   │
        u -> │ ────── │ -> y
             │ 1 + sT │
             └────────┘

    Exports state `x` (symbol `x'`) and output algebraic variable `y`.
    """

    def __init__(self, u, T, K, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T = dummify(T)
        self.K = dummify(K)
        self.enforce_tex_name((self.K, self.T))

        self.x = State(info='State in washout filter', tex_name="x'", t_const=self.T)
        self.y = Algeb(info='Output of washout filter', tex_name=r'y', diag_eps=1e-6)
        self.vars.update({'x': self.x, 'y': self.y})

    def define(self):
        r"""
        Notes
        -----
        Equations and initial values:

        .. math ::
            T \dot{x'} &= (u - x') \\
            T y &= K (u - x') \\
            x'^{(0)} &= u \\
            y^{(0)} &= 0

        """
        self.x.v_str = f'{self.u.name}'
        self.y.v_str = '0'

        self.x.e_str = f'({self.u.name} - {self.name}_x)'
        self.y.e_str = f'{self.K.name} * ({self.u.name} - {self.name}_x) - {self.T.name} * {self.name}_y'


class WashoutOrLag(Washout):
    """
    Washout with the capability to convert to Lag when K = 0.

    Can be enabled with `zero_out`. Need to provide `name` to construct.

    Exports state `x` (symbol `x'`), output algebraic variable `y`, and a LessThan block `LT`.

    Parameters
    ----------
    zero_out : bool, optional
        If True, ``sT`` will become 1, and the washout will become a low-pass filter.
        If False, functions as a regular Washout.
    """

    def __init__(self, u, T, K, name=None, zero_out=True, tex_name=None, info=None):
        super().__init__(u, T, K, name=name, tex_name=tex_name, info=info)
        self.zero_out = zero_out
        self.LT = LessThan(K,
                           dummify(0),
                           equal=True,
                           enable=zero_out,
                           tex_name='LT',
                           cache=True,
                           z0=1, z1=0)

        self.vars.update({'LT': self.LT})

    def define(self):
        r"""
        Notes
        -----
        Equations and initial values:

        .. math ::
            T \dot{x'} &= (u - x') \\
            T y = z_0 K (u - x') + z_1 T x \\
            x'^{(0)} &= u \\
            y^{(0)} &= 0

        where ``z_0`` is a flag array for the greater-than-zero elements, and ``z_1`` is that for the
        less-than or equal-to zero elements.
        """

        super().define()

        self.y.v_str = f'{self.name}_LT_z0 * 0 + {self.name}_LT_z1 * {self.name}_x'

        self.y.e_str = f'{self.name}_LT_z0 * {self.K.name} * ({self.u.name} - {self.name}_x) + ' \
                       f'{self.name}_LT_z1 * {self.T.name} * {self.name}_x - ' \
                       f'{self.T.name} * {self.name}_y'


class Lag(Block):
    r"""
    Lag (low pass filter) transfer function. ::

             ┌────────┐
             │    K   │
        u -> │ ────── │ -> y
             │ 1 + sT │
             └────────┘

    Exports one state variable `y` as the output.

    Parameters
    ----------
    K
        Gain
    T
        Time constant
    u
        Input variable

    """

    def __init__(self, u, T, K, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T = dummify(T)
        self.K = dummify(K)

        self.enforce_tex_name((self.K, self.T))
        self.y = State(info='State in lag transfer function', tex_name="y",
                       t_const=self.T)

        self.vars = {'y': self.y}

    def define(self):
        r"""

        Notes
        -----
        Equations and initial values are

        .. math ::

            T \dot{y} &= (Ku - y) \\
            y^{(0)} &= K u

        """
        self.y.v_str = f'{self.u.name} * {self.K.name}'
        self.y.e_str = f'({self.K.name} * {self.u.name} - {self.name}_y)'


class LagAntiWindup(Block):
    r"""
    Lag (low pass filter) transfer function block with an anti-windup limiter. ::

                     upper
                   /¯¯¯¯¯¯
             ┌────────┐
             │    K   │
        u -> │ ────── │ -> y
             │ 1 + sT │
             └────────┘
           ______/
           lower

    Exports one state variable `y` as the output and one AntiWindup instance `lim`.

    Parameters
    ----------
    K
        Gain
    T
        Time constant
    u
        Input variable

    """

    def __init__(self, u, T, K, lower, upper,
                 name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T = dummify(T)
        self.K = dummify(K)

        self.lower = lower
        self.upper = upper

        self.enforce_tex_name((self.T, self.K))

        self.y = State(info='State in lag TF', tex_name="y",
                       t_const=self.T)
        self.lim = AntiWindup(u=self.y, lower=self.lower, upper=self.upper, tex_name='lim',
                              info='Limiter in Lag')

        self.vars = {'y': self.y, 'lim': self.lim}

    def define(self):
        r"""

        Notes
        -----
        Equations and initial values are

        .. math ::

            T \dot{y} &= (Ku - y) \\
            y^{(0)} &= K u

        """
        self.y.v_str = f'{self.u.name} * {self.K.name}'
        self.y.e_str = f'{self.K.name} * {self.u.name} - {self.name}_y'


class Lag2ndOrd(Block):
    r"""
    Second order lag transfer function (low-pass filter) ::

             ┌──────────────────┐
             │         K        │
        u -> │ ──────────────── │ -> y
             │ 1 + sT1 + s^2 T2 │
             └──────────────────┘

    Exports one two state variables (`x`, `y`), where `y` is the output.

    Parameters
    ----------
    u
        Input
    K
        Gain
    T1
        First order time constant
    T2
        Second order time constant
    """

    def __init__(self, u, K, T1, T2, name=None, tex_name=None, info=None):
        super(Lag2ndOrd, self).__init__(name=name, tex_name=tex_name, info=info)

        self.u = dummify(u)
        self.K = dummify(K)
        self.T1 = dummify(T1)
        self.T2 = dummify(T2)

        self.enforce_tex_name((self.K, self.T1, self.T2))

        self.x = State(info='State in 2nd order LPF', tex_name="x'", t_const=self.T2)
        self.y = State(info='Output of 2nd order LPF', tex_name='y')

        self.vars = {'x': self.x, 'y': self.y}

    def define(self):
        r"""

        Notes
        -------
        Implemented equations and initial values are

        .. math ::

            T_2 \dot{x} &= Ku - y - T_1 x \\
            \dot{y} &= x \\
            x^{(0)} &= 0 \\
            y^{(0)} &= K u
        """

        self.x.v_str = 0
        self.x.e_str = f'{self.u.name} * {self.K.name} - ' \
                       f'{self.name}_y - ' \
                       f'{self.T1.name} * {self.name}_x'

        self.y.v_str = f'{self.u.name} * {self.K.name}'
        self.y.e_str = f'{self.name}_x'


class LeadLag(Block):
    r"""
    Lead-Lag transfer function block in series implementation ::

             ┌───────────┐
             │   1 + sT1 │
        u -> │ K ─────── │ -> y
             │   1 + sT2 │
             └───────────┘

    Exports two variables: internal state `x` and output algebraic variable `y`.

    Notes
    -----
    To allow zeroing out lead-lag as a pure gain, set ``zero_out`` to `True`.

    Parameters
    ----------
    T1 : BaseParam
        Time constant 1
    T2 : BaseParam
        Time constant 2
    zero_out : bool
        True to allow zeroing out lead-lag as a pass through (when T1=T2=0)
    """

    def __init__(self, u, T1, T2, K=1, zero_out=True, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T1 = dummify(T1)
        self.T2 = dummify(T2)
        self.K = dummify(K)
        self.zero_out = zero_out

        self.enforce_tex_name((self.T1, self.T2))

        self.x = State(info='State in lead-lag', tex_name="x'", t_const=self.T2)
        self.y = Algeb(info='Output of lead-lag', tex_name=r'y', diag_eps=1e-6)
        self.vars = {'x': self.x, 'y': self.y}

        if self.zero_out is True:
            self.LT1 = LessThan(T1, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.LT2 = LessThan(T2, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.x.discrete = (self.LT1, self.LT2)
            self.vars['LT1'] = self.LT1
            self.vars['LT2'] = self.LT2

    def define(self):
        r"""

        Notes
        -----

        Implemented equations and initial values

        .. math ::

            T_2 \dot{x'} &= (u - x') \\
            T_2  y &= K T_1  (u - x') + K T_2  x' + E_2 \, , \text{where} \\
            E_2 = &
            \left\{\begin{matrix}
            (y - K x') &\text{ if } T_1 = T_2 = 0 \& zero\_out=True \\
            0& \text{ otherwise }
            \end{matrix}\right. \\
            x'^{(0)} & = u\\
            y^{(0)} & = Ku\\


        """
        self.x.v_str = f'{self.u.name}'
        self.y.v_str = f'{self.u.name}'

        self.x.e_str = f'({self.u.name} - {self.name}_x)'
        self.y.e_str = f'{self.K.name} * {self.T1.name} * ({self.u.name} - {self.name}_x) + ' \
                       f'{self.K.name} * {self.name}_x * {self.T2.name} - ' \
                       f'{self.name}_y * {self.T2.name}'

        # when T1=T2=0, use equation `0 = y - Kx`
        if self.zero_out is True:
            self.y.e_str += f'+ {self.name}_LT1_z1 * {self.name}_LT2_z1 * ' \
                            f'({self.name}_y - {self.K.name} * {self.name}_x)'


class LeadLag2ndOrd(Block):
    r"""
    Second-order lead-lag transfer function block ::

             ┌──────────────────┐
             │ 1 + sT3 + s^2 T4 │
        u -> │ ──────────────── │ -> y
             │ 1 + sT1 + s^2 T2 │
             └──────────────────┘

    Exports two internal states (`x1` and `x2`) and output algebraic variable `y`.

    # TODO: instead of implementing `zero_out` using `LessThan` and an additional
    term, consider correcting all parameters to 1 if all are 0.

    """

    def __init__(self, u, T1, T2, T3, T4, zero_out=False, name=None, tex_name=None, info=None):
        super(LeadLag2ndOrd, self).__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T1 = dummify(T1)
        self.T2 = dummify(T2)
        self.T3 = dummify(T3)
        self.T4 = dummify(T4)
        self.zero_out = zero_out
        self.enforce_tex_name((self.T1, self.T2, self.T3, self.T4))

        self.x1 = State(info='State #1 in 2nd order lead-lag', tex_name="x'", t_const=self.T2)
        self.x2 = State(info='State #2 in 2nd order lead-lag', tex_name="x''")
        self.y = Algeb(info='Output of 2nd order lead-lag', tex_name='y', diag_eps=1e-6)

        self.vars = {'x1': self.x1, 'x2': self.x2, 'y': self.y}

        if self.zero_out is True:
            self.LT1 = LessThan(T1, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.LT2 = LessThan(T2, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.LT3 = LessThan(T4, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.LT4 = LessThan(T4, dummify(0), equal=True, enable=zero_out, tex_name='LT',
                                cache=True, z0=1, z1=0)
            self.x2.discrete = (self.LT1, self.LT2, self.LT3, self.LT4)
            self.vars['LT1'] = self.LT1
            self.vars['LT2'] = self.LT2
            self.vars['LT3'] = self.LT3
            self.vars['LT4'] = self.LT4

    def define(self):
        r"""
        Notes
        -----
        Implemented equations and initial values are

        .. math ::
            T_2 \dot{x}_1 &= u - x_2 - T_1 x_1 \\
            \dot{x}_2 &= x_1 \\
            T_2 y &= T_2 x_2 + T_2 T_3 x_1 + T_4 (u - x_2 - T_1 x_1) + E_2 \, , \text{ where} \\
            E_2 = &
            \left\{\begin{matrix}
            (y - x_2) &\text{ if } T_1 = T_2 = T_3 = T_4 = 0 \& zero\_out=True \\
            0& \text{ otherwise }
            \end{matrix}\right. \\
            x_1^{(0)} &= 0 \\
            x_2^{(0)} &= y^{(0)} = u

        """
        self.x1.e_str = f'{self.u.name} - {self.name}_x2 - {self.T1.name} * {self.name}_x1'
        self.x2.e_str = f'{self.name}_x1'
        self.y.e_str = f'{self.T2.name} * {self.name}_x2 + ' \
                       f'{self.T2.name} * {self.T3.name} * {self.name}_x1 + ' \
                       f'{self.T4.name} * ({self.u.name} - {self.name}_x2 - {self.T1.name} * {self.name}_x1) - ' \
                       f'{self.T2.name} * {self.name}_y'

        self.x1.v_str = 0
        self.x2.v_str = f'{self.u.name}'
        self.y.v_str = f'{self.u.name}'

        # when T1=T2=0, use equation `0 = y - Kx`
        if self.zero_out is True:
            self.y.e_str += f'+ {self.name}_LT1_z1*{self.name}_LT2_z1*{self.name}_LT3_z1*{self.name}_LT4_z1 * ' \
                            f'({self.name}_y - {self.name}_x2)'


class LeadLagLimit(Block):
    r"""
    Lead-Lag transfer function block with hard limiter (series implementation) ::

             ┌─────────┐          upper
             │ 1 + sT1 │         /¯¯¯¯¯
        u -> │ ─────── │ -> ynl / -> y
             │ 1 + sT2 │  _____/
             └─────────┘  lower

    Exports four variables: state `x`, output before hard limiter `ynl`, output `y`, and AntiWindup `lim`.

    """

    def __init__(self, u, T1, T2, lower, upper,
                 name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.T1 = dummify(T1)
        self.T2 = dummify(T2)
        self.lower = lower
        self.upper = upper
        self.enforce_tex_name((self.T1, self.T2))

        self.x = State(info='State in lead-lag TF', tex_name="x'", t_const=self.T2)
        self.ynl = Algeb(info='Output of lead-lag TF before limiter', tex_name=r'y_{nl}')
        self.y = Algeb(info='Output of lead-lag TF after limiter', tex_name=r'y',
                       diag_eps=1e-6)
        self.lim = AntiWindup(u=self.ynl, lower=self.lower, upper=self.upper)

        self.vars = {'x': self.x, 'ynl': self.ynl, 'y': self.y, 'lim': self.lim}

    def define(self):
        r"""

        Notes
        -----

        Implemented control block equations (without limiter) and initial values

        .. math ::

            T_2 \dot{x'} &= (u - x') \\
            T_2 y &= T_1  (u - x') + T_2  x' \\
            x'^{(0)} &= y^{(0)} = u

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


class HVGate(Block):
    """
    High Value Gate. Outputs the maximum of two inputs. ::

              ┌─────────┐
        u1 -> │ HV Gate │
              │         │ ->  y
        u2 -> │  (MAX)  │
              └─────────┘

    """

    def __init__(self, u1, u2, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u1 = dummify(u1)
        self.u2 = dummify(u2)
        self.enforce_tex_name((u1, u2))

        self.sl = Selector(self.u1, self.u2, fun=np.maximum.reduce,
                           info='HVGate Selector',
                           )

        self.y = Algeb(info='HVGate output', tex_name='y', discrete=self.sl)
        self.vars = {'y': self.y, 'sl': self.sl}

    def define(self):
        """
        Implemented equations and initial conditions

        .. math ::

            0 = s_0^{sl} u_1 + s_1^{sl} u_2 - y
            y_0 = maximum(u_1, u_2)

        Notes
        -----
        In the implementation, one should not use ::

            self.y.v_str = f'maximum({self.u1.name}, {self.u2.name})',

        because SymPy processes this equation to `{self.u1.name}`.
        Not sure if this is a bug or intended.

        """
        self.y.v_str = f'{self.name}_sl_s0*{self.u1.name} + {self.name}_sl_s1*{self.u2.name}'
        self.y.e_str = f'{self.name}_sl_s0*{self.u1.name} + {self.name}_sl_s1*{self.u2.name} - ' \
                       f'{self.name}_y'


class LVGate(Block):
    """
    Low Value Gate. Outputs the minimum of the two inputs. ::

              ┌─────────┐
        u1 -> │ LV Gate |
              │         | ->  y
        u2 -> │  (MIN)  |
              └─────────┘

    """

    def __init__(self, u1, u2, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u1 = dummify(u1)
        self.u2 = dummify(u2)
        self.enforce_tex_name((u1, u2))

        self.y = Algeb(info='LVGate output', tex_name='y')
        self.sl = Selector(self.u1, self.u2, fun=np.minimum.reduce,
                           info='LVGate Selector',
                           )

        self.vars = {'y': self.y, 'sl': self.sl}

    def define(self):
        """
        Implemented equations and initial conditions

        .. math ::

            0 = s_0^{sl} u_1 + s_1^{sl} u_2 - y
            y_0 = minimum(u_1, u_2)

        Notes
        -----
        Same problem as `HVGate` as `minimum` does not sympify correctly.

        """
        self.y.v_str = f'{self.name}_sl_s0*{self.u1.name} + {self.name}_sl_s1*{self.u2.name}'
        self.y.e_str = f'{self.name}_sl_s0*{self.u1.name} + {self.name}_sl_s1*{self.u2.name} - ' \
                       f'{self.name}_y'


class GainLimiter(Block):
    """
    Gain followed by a limiter.

    Exports the limited output `y`, unlimited output `x`, and HardLimiter `lim`. ::

             ┌─────┐         upper
             │     │        /¯¯¯¯¯
        u -> │  K  │ -> x  / -> y
             │     │ _____/
             └─────┘ lower

    Parameters
    ----------
    u : str, BaseVar
        Input variable, or an equation string for constructing an anonymous variable

    """

    def __init__(self, u, K, upper, lower, no_upper=False, no_lower=False,
                 name=None, tex_name=None, info=None):
        Block.__init__(self, name=name, tex_name=tex_name, info=info)
        self.u = dummify(u)
        self.K = dummify(K)
        self.upper = upper
        self.lower = lower

        if (no_upper and no_lower) is True:
            raise ValueError("no_upper or no_lower cannot both be True")

        self.no_lower = no_lower
        self.no_upper = no_upper

        self.x = Algeb(info='Gain output before limiter', tex_name='x')
        self.y = Algeb(info='Gain output after limiter', tex_name='y')

        self.lim = HardLimiter(u=self.x, lower=self.lower, upper=self.upper,
                               no_upper=no_upper, no_lower=no_lower,
                               tex_name='lim')

        self.vars = {'lim': self.lim, 'x': self.x, 'y': self.y}

    def define(self):
        """
        TODO: write docstring
        """
        self.x.v_str = f'{self.K.name} * ({self.u.name})'
        self.x.e_str = f'{self.K.name} * ({self.u.name}) - {self.name}_x'

        self.y.e_str = f'{self.name}_x * {self.name}_lim_zi'
        self.y.v_str = f'{self.name}_x * {self.name}_lim_zi'

        if not self.no_upper:
            self.y.e_str += f' + {self.name}_lim_zu*{self.upper.name}'
            self.y.v_str += f' + {self.name}_lim_zu*{self.upper.name}'
        if not self.no_lower:
            self.y.e_str += f' + {self.name}_lim_zl*{self.lower.name}'
            self.y.v_str += f' + {self.name}_lim_zl*{self.lower.name}'

        self.y.e_str += f' - {self.name}_y'


class Piecewise(Block):
    """
    Piecewise block. Outputs an algebraic variable `y`.

    This block takes a list of N points, [x0, x1, ...x_{n-1}] to define N+1 ranges,
    namely (-inf, x0), (x0, x1), ..., (x_{n-1}, +inf).
    and a list of N+1 functions [fun0, ..., fun_n].

    Inputs that fall within each range applies the corresponding function.
    The first range (-inf, x0) applies `fun_0`, and
    the last range (x_{n-1}, +inf) applies the last function `fun_n`.

    Parameters
    ----------
    points : list, tuple
        A list of piecewise points. Need to be provided in the constructor function.
    funs : list, tuple
        A list of strings for the piecewise functions. Need to be provided in the overloaded `define` function.
    """

    def __init__(self, u, points: Union[List, Tuple], funs: Union[List, Tuple],
                 name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = u
        self.points = points
        self.funs = funs

        self.y = Algeb(info='Output of piecewise', tex_name='y')
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
        args.append(f'({self.funs[i + 1]}, True)')

        args_comma = ', '.join(args)
        pw_fun = f'Piecewise({args_comma})'

        self.y.v_str = pw_fun
        self.y.e_str = f'{pw_fun} - {self.name}_y'
