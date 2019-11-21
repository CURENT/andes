from andes.core.var import Algeb, State
from typing import Optional


class Block(object):
    """
    Base class for control blocks.

    Blocks are meant to be instantiated as Model attributes
    to provide pre-defined equation sets. All subclasses must
    provide `get_name` method and `export_vars` method. Subclasses
    must overload the `__init__` method to take custom inputs.


    Warnings
    --------
    This class may be significantly modified soon.

    Parameters
    ----------
    info : str, optional
        Block description.
    """

    def __init__(self, info: Optional[str] = None, *args, **kwargs):
        self.name = None
        self.owner = None
        self.info = info
        self.vars = {}

        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()

        self.ifxc, self.jfxc, self.vfxc = list(), list(), list()
        self.ifyc, self.jfyc, self.vfyc = list(), list(), list()
        self.igxc, self.jgxc, self.vgxc = list(), list(), list()
        self.igyc, self.jgyc, self.vgyc = list(), list(), list()

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

    """
    def __init__(self, var, ref, kp, ki, **kwargs):
        super(PIController, self).__init__(**kwargs)

        self.var = var
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller", block=True)
        self.y = Algeb(info="Integration value of PI controller", block=True)

    def export_vars(self):
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

        self.xi.e_str = f'ki * ({self.ref.name} - {self.var.name})'
        self.y.e_str = f'kp * ({self.ref.name} - {self.var.name}) + {self.name}_xi'

        self.vars = {self.name + '_xi': self.xi, self.name + '_y': self.y}
        return self.vars


# TODO: what if a PI controller has a limiter? How can it be exported?

class PIControllerNumeric(Block):

    def __init__(self, var, ref, kp, ki, **kwargs):
        super().__init__(**kwargs)

        self.var = var
        self.ref = ref
        self.kp = kp
        self.ki = ki

        self.xi = State(info="Integration value of PI controller", block=True)
        self.y = Algeb(info="Integration value of PI controller", block=True)

    def export_vars(self):
        self.vars = {self.name + '_xi': self.xi, self.name + '_y': self.y}
        return self.vars

    def g_numeric(self, **kwargs):
        self.y.e = self.kp.v * (self.ref.v - self.var.v) + self.xi.v

    def f_numeric(self, **kwargs):
        self.xi.e = self.ki.v * (self.ref.v - self.var.v)

    def store_jacobian(self):
        self.j_reset()

        self.ifyc.append(self.xi.a)
        self.jfyc.append(self.var.a)
        self.vfyc.append(-self.ki.v)

        self.igyc.append(self.y.a)
        self.jgyc.append(self.var.v)
        self.vgyc.append(-self.kp.v)

        self.igxc.append(self.y.a)
        self.jgxc.append(self.xi.a)
        self.vgxc.append(1)


class ArrayReduce(Block):
    """
    This block takes a 2D `ExtVar` (with its `_v` as a list of arrays),
    call the callback for each array in the list, and return an array of scalars.
    """
    def __init__(self, ext_var):
        pass
