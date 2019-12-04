from andes.core.var import Algeb, State
from typing import Optional
from andes.core.discrete import HardLimiter


class Block(object):
    """
    Base class for control blocks.

    Blocks are meant to be instantiated as Model attributes
    to provide pre-defined equation sets. All subclasses must
    provide `get_name` method and `export_vars` method. Subclasses
    must overload the `__init__` method to take custom inputs.

    Warnings
    --------
    This class is subject to changes soon.

    Parameters
    ----------
    info : str, optional
        Block description.
    """

    def __init__(self, name: Optional[str] = None, info: Optional[str] = None):
        self.name = name
        self.tex_name = None
        self.owner = None
        self.info = info
        self.vars = {}

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

    def j_reset(self):
        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()

        self.ifxc, self.jfxc, self.vfxc = list(), list(), list()
        self.ifyc, self.jfyc, self.vfyc = list(), list(), list()
        self.igxc, self.jgxc, self.vgxc = list(), list(), list()
        self.igyc, self.jgyc, self.vgyc = list(), list(), list()

    def get_name(self):
        """
        Method for getting the name of the block.

        Notes
        -----
        This method is currently unused

        Returns
        -------
        list
            The `name` attribute in a list
        """
        return [self.name]

    def set_eq(self):
        pass

    def export_vars(self):
        """
        Method for exporting algebraic and state variables
        created in this block as a dictionary.

        Subclasses must implement the meta-equations in the
        derived `export_vars` method. Before returning `vars`,
        the subclass should first update `vars`.


        See Also
        --------
        PIController.export_vars

        Returns
        -------
        dict
            Key being the variable name and the value being
            the variable instance.
        """
        self.set_eq()
        return self.vars

    def g_numeric(self, **kwargs):
        """
        Function to customize function calls
        """
        pass

    def f_numeric(self, **kwargs):
        """
        Function to customize differential function calls
        """
        pass

    def c_numeric(self, **kwargs):
        pass

    def j_numeric(self):
        """
        This function stores the constant and variable jacobian information.

        Constant jacobians are stored by indices and values in `ifxc`, `jfxc`
        and `vfxc`. Note that it is the values that gets stored in `vfxc`.
        Variable jacobians are stored by indices and functions. The function
        shall return the value of the corresponding jacobian elements.

        """
        pass


class SampleAndHolder(Block):
    """
    Sample and hold block

    Warnings
    --------
    Not implemented yet.
    """
    pass


class PIController(Block):
    """
    Proportional Integral Controller with the reference from an external variable

    Parameters
    ----------
    var : VarBase
        The input variable instance
    ref : Union[VarBase, ParamBase]
        The reference instance
    kp : ParamBase
        The proportional gain parameter instance
    ki : [type]
        The integral gain parameter instance

    # TODO: what if a PI controller has a limiter? How can it be exported?
    """
    def __init__(self, u, ref, kp, ki, name=None, info=None):
        super(PIController, self).__init__(name=name, info=info)

        self.u = u
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller")
        self.y = Algeb(info="Integration value of PI controller")

        self.vars = {'xi': self.xi, 'y': self.y}

    def set_eq(self):
        r"""
        Define meta equations and export variables of the PI Controller.

        Notes
        -----
        One state variable ``xi`` and one algebraic variable ``y`` are added.

        Equations implemented are

        .. math ::
            \dot{x_i} = k_i * (ref - var)

            y = x_i + k_i * (ref - var)

        Warnings
        --------
        Only one level of nesting is allowed.
        Namely, PIController cannot have a sub-block.

        Returns
        -------
        dict
            A dictionary with the keys being the names of the two variables,
            ``xi`` and ``y``, and the values being the corresponding
            instances.
        """

        self.xi.e_str = f'ki * ({self.ref.name} - {self.u.name})'
        self.y.e_str = f'kp * ({self.ref.name} - {self.u.name}) + {self.name}_xi'


class PIControllerNumeric(Block):

    def __init__(self, u, ref, kp, ki, name=None, info=None):
        super().__init__(name=name, info=info)

        self.u = u
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller")
        self.y = Algeb(info="Integration value of PI controller")

        self.vars = {'xi': self.xi, 'y': self.y}

    def g_numeric(self, **kwargs):
        self.y.e = self.kp.v * (self.ref.v - self.u.v) + self.xi.v

    def f_numeric(self, **kwargs):
        self.xi.e = self.ki.v * (self.ref.v - self.u.v)

    def store_jacobian(self):
        self.j_reset()

        self.ifyc.append(self.xi.a)
        self.jfyc.append(self.u.a)
        self.vfyc.append(-self.ki.v)

        self.igyc.append(self.y.a)
        self.jgyc.append(self.u.v)
        self.vgyc.append(-self.kp.v)

        self.igxc.append(self.y.a)
        self.jgxc.append(self.xi.a)
        self.vgxc.append(1)


class Washout(Block):
    """
    Washout filter (high pass) block

      sT1
    -------
    1 + sT2

    Equations:
    x' dot = (u - x) / T
    y = u - x

    Initial Values
    x0 = u
    y0 = 0
    """

    def __init__(self, u, T, info=None, name=None):
        super().__init__(name=name, info=info)
        self.T = T
        self.u = u

        self.x = State(info='State in washout filter', tex_name="x'")
        self.y = Algeb(info='Output of washout filter', tex_name=r'y')
        self.vars = {'x': self.x, 'y': self.y}

    def set_eq(self):
        self.x.v_init = f'{self.u.name}'
        self.y.v_init = f'0'

        self.x.e_str = f'({self.u.name} - {self.name}_x) / {self.T.name}'
        self.y.e_str = f'({self.u.name} - {self.name}_x) - {self.name}_y'


class Lag(Block):
    """
    Lag (low pass) transfer function block

       K
    ------
    1 + sT

    Equations:
    x' dot = (u - x) / T

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
    def __init__(self, u, K, T, info='Lag transfer function', name=None):
        super().__init__(name=name, info=info)

        self.K = K
        self.T = T
        self.u = u
        self.x = State(info='State in lag transfer function', tex_name="x'")

        self.vars = {'x': self.x}

    def set_eq(self):
        self.x.v_init = f'{self.u.name}'
        self.x.e_str = f'({self.K.name} * {self.u.name} - {self.name}_x) / {self.T.name}'


class LeadLag(Block):
    """
    Lead-Lag transfer function block in series implementation

    1 + sT1
    -------
    1 + sT2

    Exports two variables: state x and output y.

    Equations:
    x' dot = (u - x') / T2
    y = T1/T2 * (u - x') + x'
    """
    def __init__(self, u, T1, T2, info='Lead-lag transfer function', name=None):
        super().__init__(name=name, info=info)
        self.T1 = T1
        self.T2 = T2
        self.u = u

        self.x = State(info='State in lead-lag transfer function', tex_name="x'")
        self.y = Algeb(info='Output of lead-lag transfer function', tex_name=r'y')
        self.vars = {'x': self.x, 'y': self.y}

    def set_eq(self):
        self.x.v_init = f'{self.u.name}'
        self.y.v_init = f'{self.u.name}'

        self.x.e_str = f'({self.u.name} - {self.name}_x) / {self.T2.name}'
        self.y.e_str = f'{self.T1.name} / {self.T2.name} * ({self.u.name} - {self.name}_x) + \
                         {self.name}_x - \
                         {self.name}_y'


class LeadLagLimit(Block):
    """
    Lead-Lag transfer function block with hard limiter (series implementation)
                   ___upper___
                  /
              1 + sT1
       u ->   -------  -> y
              1 + sT2
    __lower____/

    Exports four variables: state `x`, output before hard limiter `ynl`, output `y`, and limiter `lim`,

    Equations:
    x' dot = (u - x') / T2
    y = T1/T2 * (u - x') + x'
    """
    def __init__(self, u, T1, T2, lower, upper, info='Lead-lag transfer function', name=None):
        super().__init__(name=name, info=info)
        self.T1 = T1
        self.T2 = T2
        self.u = u
        self.lower = lower
        self.upper = upper

        self.x = State(info='State in lead-lag transfer function', tex_name="x'")
        self.ynl = Algeb(info='Output of lead-lag transfer function before limiter', tex_name=r'y_{nl}')
        self.y = Algeb(info='Output of lead-lag transfer function after limiter', tex_name=r'y')
        self.lim = HardLimiter(u=self.y, lower=self.lower, upper=self.upper)

        self.vars = {'x': self.x, 'ynl': self.ynl, 'y': self.y, 'lim': self.lim}

    def set_eq(self):
        self.x.v_init = f'{self.u.name}'
        self.ynl.v_init = f'{self.u.name}'
        self.y.v_init = f'{self.u.name}'

        self.x.e_str = f'({self.u.name} - {self.name}_x) / {self.T2.name}'
        self.ynl.e_str = f'{self.T1.name} / {self.T2.name} * ({self.u.name} - {self.name}_x) + ' \
                         f'{self.name}_x - ' \
                         f'{self.name}_ynl'

        self.y.e_str = f'{self.name}_ynl * {self.name}_lim_zi + ' \
                       f'{self.lower.name} * {self.name}_lim_zl + ' \
                       f'{self.upper.name} * {self.name}_lim_zu - ' \
                       f'{self.name}_y'
