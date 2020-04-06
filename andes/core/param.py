from typing import Optional, Union, Callable

import math
import logging
from andes.models.group import GroupBase
from andes.shared import np
logger = logging.getLogger(__name__)


class BaseParam(object):
    """
    The base parameter class.

    This class provides the basic data structure and interfaces for all types of parameters. Parameters are from
    input files and in general constant once initialized.

    Subclasses should overload the ``n()`` method for the total count of elements in the value
    array.

    Parameters
    ----------
    default : str or float, optional
        The default value of this parameter if None is provided
    name : str, optional
        Parameter name. If not provided, it will be automatically set to the attribute name defined in the
        owner model.
    tex_name : str, optional
        LaTeX-formatted parameter name. If not provided, `tex_name` will be assigned the same as `name`.
    info : str, optional
        Descriptive information of parameter
    mandatory : bool
        True if this parameter is mandatory
    export : bool
        True if the parameter will be exported when dumping data into files. True for most parameters.
        False for ``RefParam``.

    Attributes
    ----------
    v : list
        A list holding all the values. The ``BaseParam`` class does not convert the ``v`` attribute into NumPy
        arrays.
    property : dict
        A dict containing the truth values of the model properties.
    """
    def __init__(self,
                 default: Optional[Union[float, str, int]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 mandatory: bool = False,
                 export: bool = True):
        self.name = name
        self.default = default
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.unit = unit
        self.owner = None
        self.export = export

        self.v = []
        self.property = dict(mandatory=mandatory)

    def add(self, value=None):
        """
        Add a new parameter value (from a new device of the owner model) to the ``v`` list.

        Parameters
        ----------
        value : str or float, optional
            Parameter value of the new element. If None, the default will be used.

        Notes
        -----
        If the value is ``math.nan``, it will set to ``None``.
        """

        if isinstance(value, float) and math.isnan(value):
            value = None

        # check for mandatory
        if value is None:
            if self.get_property('mandatory'):
                raise ValueError(f'Mandatory parameter {self.name} for {self.owner.class_name} missing')
            else:
                value = self.default

        if isinstance(self.v, list):
            self.v.append(value)
        else:
            np.append(self.v, value)

    def get_property(self, property_name: str):
        """
        Check the boolean value of the given property. If the property does not exist in the dictionary,
        ``False`` will be returned.

        Parameters
        ----------
        property_name : str
            Property name

        Returns
        -------
        The truth value of the property.
        """
        if property_name not in self.property:
            return False
        return self.property[property_name]

    def get_names(self):
        """
        Return ``self.name`` in a list.

        This is a helper function to provide the same API as blocks or discrete components.

        Returns
        -------
        list
            A list only containing the name of the parameter
        """
        return [self.name]

    @property
    def class_name(self):
        """Return the class name."""
        return self.__class__.__name__

    @property
    def n(self):
        """Return the count of elements in the value array."""
        return len(self.v) if self.v else 0


class DataParam(BaseParam):
    """
    An alias of the ``BaseParam`` class.

    This class is used for string parameters or non-computational numerical parameters.
    This class does not provide a ``to_array`` method.
    All input values will be stored in ``v`` as a list.

    See Also
    --------
    BaseParam : Base parameter class
    """
    pass


class IdxParam(BaseParam):
    """
    An alias of ``BaseParam`` with an additional storage of the owner model name

    This class is intended for storing ``idx`` into other models. It can be used in the future for data
    consistency check.

    Examples
    --------
    A PQ model connected to Bus model will have the following code ::

        class PQModel(...):
            def __init__(...):
                ...
                self.bus = IdxParam(model='Bus')

    """
    def __init__(self,
                 default: Optional[Union[float, str, int]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 mandatory: bool = False,
                 export: bool = True,
                 model: Optional[str] = None):
        super().__init__(default=default, name=name, tex_name=tex_name, info=info, unit=unit, mandatory=mandatory,
                         export=export)
        self.model = model  # must be a `Model` name for building RefParam - Not checked yet


class NumParam(BaseParam):
    """
    A computational numerical parameter.

    Parameters defined using this class will have their ``v`` field converted to a NumPy.ndarray after adding.
    The original input values will be copied to `vin`, and the system-base per-unit conversion coefficients
    (through multiplication) will be stored in `pu_coeff`.

    Parameters
    ----------
    default : str or float, optional
        The default value of this parameter if no value is provided
    name : str, optional
        Name of this parameter. If not provided, `name` will be set
        to the attribute name of the owner model.
    tex_name : str, optional
        LaTeX-formatted parameter name. If not provided, `tex_name`
        will be assigned the same as `name`.
    info : str, optional
        A description of this parameter
    mandatory : bool
        True if this parameter is mandatory
    unit : str, optional
        Unit of the parameter

    Other Parameters
    ----------------
    non_zero : bool
        True if this parameter must be non-zero
    mandatory : bool
        True if this parameter must not be None
    power : bool
        True if this parameter is a power per-unit quantity
        under the device base
    ipower : bool
        True if this parameter is an inverse-power per-unit
        quantity under the device base
    voltage : bool
        True if the parameter is a voltage pu quantity
        under the device base
    current : bool
        True if the parameter is a current pu quantity
        under the device base
    z : bool
        True if the parameter is an AC impedance pu quantity
        under the device base
    y : bool
        True if the parameter is an AC admittance pu quantity
        under the device base
    r : bool
        True if the parameter is a DC resistance pu quantity
        under the device base
    g : bool
        True if the parameter is a DC conductance pu quantity
        under the device base
    dc_current : bool
        True if the parameter is a DC current pu quantity under
        device base
    dc_voltage : bool
        True if the parameter is a DC voltage pu quantity under
        device base
    """

    def __init__(self,
                 default: Optional[Union[float, str, Callable]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 non_zero: bool = False,
                 mandatory: bool = False,
                 power: bool = False,
                 ipower: bool = False,
                 voltage: bool = False,
                 current: bool = False,
                 z: bool = False,
                 y: bool = False,
                 r: bool = False,
                 g: bool = False,
                 dc_voltage: bool = False,
                 dc_current: bool = False,
                 export: bool = True,
                 ):
        super(NumParam, self).__init__(default=default, name=name, tex_name=tex_name, info=info,
                                       unit=unit, export=export)

        self.property = dict(non_zero=non_zero,
                             mandatory=mandatory,
                             power=power,
                             ipower=ipower,
                             voltage=voltage,
                             current=current,
                             z=z,
                             y=y,
                             r=r,
                             g=g,
                             dc_current=dc_current,
                             dc_voltage=dc_voltage)

        self.pu_coeff = np.ndarray([])
        self.vin = None  # values from input

    def add(self, value=None):
        """
        Add a value to the parameter value list.

        In addition to ``BaseParam.add``, this method checks for non-zero property and reset to default if is zero.

        See Also
        --------
        BaseParam.add : add method of BaseParam

        """

        # check for math.nan, usually imported from pandas
        if isinstance(value, float) and math.isnan(value):
            value = None
        elif isinstance(value, str):
            value = float(value)

        # check for mandatory
        if value is None:
            if self.get_property('mandatory'):
                raise ValueError(f'Mandatory parameter {self.name} missing')
            else:
                value = self.default

        # check for non-zero
        if value == 0.0 and self.get_property('non_zero'):
            logger.debug(f'Parameter {self.name} of {self.owner.class_name} must be non-zero')
            value = self.default

        super(NumParam, self).add(value)

    def to_array(self):
        """
        Convert ``v`` to np.ndarray after adding elements.
        Store a copy if the input in `vin`.
        Set ``pu_coeff`` to all ones.

        The conversion enables array-based calculation.

        Warnings
        --------
        After this call, `add` will not be allowed, because data will not be copied over to ``vin``.
        """

        # data quality check
        # ----------------------------------------
        self.v = np.array(self.v, dtype=float)

        # NOTE: temporarily disabled due to nested parameters
        # if np.sum(np.isnan(self.v)) > 0:
        #     raise ValueError(f'Param <{self.name} contains NaN.')

        self.v[self.v == np.inf] = 1e8
        self.v[self.v == -np.inf] = -1e8
        # ----------------------------------------

        self.vin = np.array(self.v, dtype=float)
        self.pu_coeff = np.ones_like(self.v)

    def set_pu_coeff(self, coeff):
        """
        Store p.u. conversion coefficient into ``self.pu_coeff`` and calculate the system-base per unit with
        ``self.v = self.vin * self.pu_coeff``.

        This function must be called after ``self.to_array``.

        Parameters
        ----------
        coeff : np.ndarray
            An array with the pu conversion coefficients
        """
        self.pu_coeff = coeff
        self.v[:] = self.vin * self.pu_coeff

    def restore(self):
        """
        Restore parameter to the original input by copying ``self.vin`` to ``self.v``.

        `pu_coeff` will not be overwritten.
        """
        self.v[:] = self.vin


class TimerParam(NumParam):
    """
    A parameter whose values are event occurrence times during the simulation.

    The constructor takes an additional Callable ``self.callback`` for the action of the event.
    ``TimerParam`` has a default value of -1, meaning deactivated.

    Examples
    --------
    A connectivity status toggler class ``Toggler`` takes a parameter ``t`` for the toggle time.
    Inside ``Toggler.__init__``, one would have ::

        self.t = TimerParam()

    The ``Toggler`` class also needs to define a method for togging the connectivity status ::

        def _u_switch(self, is_time: np.ndarray):
            action = False
            for i in range(self.n):
                if is_time[i] and (self.u.v[i] == 1):
                    instance = self.system.__dict__[self.model.v[i]]
                    # get the original status and flip the value
                    u0 = instance.get(src='u', attr='v', idx=self.dev.v[i])
                    instance.set(src='u',
                                 attr='v',
                                 idx=self.dev.v[i],
                                 value=1-u0)
                    action = True
            return action

    Finally, in ``Toggler.__init__``, assign the function as the callback for ``self.t`` ::

        self.t.callback = self._u_switch

    """
    def __init__(self,
                 callback: Optional[Callable] = None,
                 default: Optional[Union[float, str, Callable]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 non_zero: bool = False,
                 mandatory: bool = False,
                 export: bool = True):
        super(TimerParam, self).__init__(default=default, name=name, tex_name=tex_name, info=info, unit=unit,
                                         mandatory=mandatory, non_zero=non_zero, export=export)
        self.default = -1  # default to -1 to deactivate
        self.callback = callback  # provide a callback function that takes an array of booleans

    def is_time(self, dae_t):
        """
        Element-wise check if the DAE time is the same as the parameter value. The current implementation uses
        ``np.isclose``

        Parameters
        ----------
        dae_t : float
            Current simulation time

        Returns
        -------
        np.ndarray
            The array containing the truth value of if the DAE time is close to the parameter value.

        See Also
        --------
        numpy.isclose : See NumPy.isclose for the warning on absolute tolerance
        """
        return np.isclose(dae_t, self.v)


class ExtParam(NumParam):
    """
    A parameter whose values are retrieved from an external model or group.

    Parameters
    ----------
    model : str
        Name of the model or group providing the original parameter
    src : str
        The source parameter name
    indexer : BaseParam
        A parameter defined in the model defining this ExtParam instance. ``indexer.v`` should contain indices into
        ``model.src.v``. If is None, the source parameter values will be fully copied. If ``model`` is a group
        name, the indexer cannot be None.

    Attributes
    ----------
    parent_model : Model
        The parent model providing the original parameter.
    """
    def __init__(self,
                 model: str,
                 src: str,
                 indexer=None,
                 **kwargs):
        super(ExtParam, self).__init__(**kwargs)
        self.model = model
        self.src = src
        self.indexer = indexer
        self.parent_model = None   # parent model instance

    def link_external(self, ext_model):
        """
        Update parameter values provided by external models. This needs to be called before pu conversion.

        TODO: Check if the pu conversion is correct or not.

        Parameters
        ----------
        ext_model : Model, Group
            Instance of the parent model or group, provided by the System calling this method.

        """
        self.parent_model = ext_model

        if isinstance(ext_model, GroupBase):

            # TODO: the three lines below is a bit inefficient - 3x same loops
            self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr='v')
            try:
                self.vin = ext_model.get(src=self.src, idx=self.indexer.v, attr='vin')
                self.pu_coeff = ext_model.get(src=self.src, idx=self.indexer.v, attr='vin')
            except KeyError:  # idx param without vin
                pass
            except TypeError:  # vin or pu_coeff is None
                pass

            # TODO: copy properties from models in the group

        else:
            parent_instance = ext_model.__dict__[self.src]
            self.property = dict(parent_instance.property)

            if self.indexer is None:
                # if `idx` is None, retrieve all the values
                uid = np.arange(ext_model.n)
            else:
                if len(self.indexer.v) == 0:
                    return
                else:
                    uid = ext_model.idx2uid(self.indexer.v)

            # pull in values
            self.v = parent_instance.v[uid]
            try:
                self.vin = parent_instance.vin[uid]
                self.pu_coeff = parent_instance.pu_coeff[uid]
            except KeyError:
                pass


class RefParam(BaseParam):
    """
    A special type of reference collector parameter.

    ``RefParam`` is used for collecting device indices of other models referencing the parent model of the
    ``RefParam``. The ``v`` field will be a list of lists, each containing the ``idx`` of other models
    referencing each device of the parent model.

    RefParam can be passed as indexer for params and vars, or shape for ``ReducerService`` and
    ``RepeaterService``. See examples for illustration.

    Examples
    --------
    A Bus device has an ``IdxParam`` of ``area``, storing the ``idx`` of area to which the bus device belongs.
    In ``Bus.__init__``, one has ::

        self.area = IdxParam(model='Area')

    Assume Bus has the following data ::

        idx    area  Vn
        1      1     110
        2      2     220
        3      1     345
        4      1     500

    The Area model wants to collect the indices of Bus devices which points to the corresponding Area device.
    In ``Area.__init__``, one defines ::

        self.Bus = RefParam()

    where the member attribute name ``Bus`` needs to match exactly model name that ``Area`` wants to collect
    ``idx`` for.
    Similarly, one can define ``self.ACTopology = RefParam()`` to collect devices in the ``ACTopology`` group
    that references Area.

    The collection of ``idx`` happens in ``System._collect_ref_param``. It has to be noted that the specific
    ``Area`` entry must exist to collect model idx-es referencing it. For example, if ``Area`` has the
    following data ::

        idx
        1

    Then, only Bus 1, 3, and 4 will be collected into ``self.Bus.v``, namely, ``self.Bus.v == [ [1, 3, 4] ]``.

    If ``Area`` has data ::

        idx
        1
        2

    Then, ``self.Bus.v`` will end up with ``[ [1, 3, 4], [2] ]``.

    See Also
    --------
    andes.core.service.ReducerService : A more complete example using RefParam to build the COI model

    """
    def __init__(self, **kwargs):
        super(RefParam, self).__init__(**kwargs)
        self.export = False
