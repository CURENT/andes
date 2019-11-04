from andes.core.var import Algeb, State
from typing import Optional


class Block(object):
    """
    Base class for control blocks.

    Blocks are meant to be instantiated as Model attributes
    to provide pre-defined equation sets. All subclasses must
    provide `get_name` method and `export_vars` method. Subclasses
    must overload the `__init__` method to take custom inputs.

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

        self.xi = State(info="integration value of PI controller", block=True)
        self.y = Algeb(info="integration value of PI controller", block=True)

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
        # NOTE:

        self.xi.e_symbolic = f'ki * ({self.ref.name} - {self.var.name})'
        self.y.e_symbolic = f'kp * ({self.ref.name} - {self.var.name}) + {self.name}_xi'

        self.vars = {self.name + '_xi': self.xi, self.name + '_y': self.y}
        return self.vars
