#  [ANDES] (C)2015-2021 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

#  File name: param.py
#  Last modified: 8/16/20, 7:27 PM

from typing import Optional, Union, Callable, List, Tuple, Type, Iterable

import math
import logging

from andes.shared import np
logger = logging.getLogger(__name__)


class BaseParam:
    """
    The base parameter class.

    This class provides the basic data structure and interfaces for all types of parameters. Parameters are from
    input files and in general constant once initialized.

    Subclasses should overload the `n()` method for the total count of elements in the value
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
        False for ``BackRef``.

    Attributes
    ----------
    v : list
        A list holding all the values. The ``BaseParam`` class does not convert the ``v`` attribute into NumPy
        arrays.
    property : dict
        A dict containing the truth values of the model properties.

    Warnings
    --------
    The most distinct feature of BaseParam, DataParam and IdxParam is that
    values are stored in a list without conversion to array.
    BaseParam, DataParam or IdxParam are **not allowed** in equations.
    """

    def __init__(self,
                 default: Optional[Union[float, str, int]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 mandatory: bool = False,
                 export: bool = True,
                 iconvert: Optional[Callable] = None,
                 oconvert: Optional[Callable] = None,
                 ):
        self.name = name
        self.default = default
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.unit = unit
        self.owner = None
        self.export = export

        self.v = []
        self.property = dict(mandatory=mandatory)
        self.iconvert = iconvert
        self.oconvert = oconvert
        self.vtype = float
        self.eltype = list

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
        value = self._sanitize(value)

        if isinstance(self.v, list):
            self.v.append(value)
        else:
            self.v = np.append(self.v, value)

    def set(self, pos, attr, value):
        """
        Set attributes of the BaseParam class to new values at the given positions.

        Parameters
        ----------
        pos : int, list of integers
            Positions in arrays where the values should be set
        attr : 'v', 'vin'
            Name of the attribute to be set
        value : str, float or list of above
            New values
        """
        if not isinstance(pos, Iterable):
            pos = [pos]

        if isinstance(self.__dict__[attr], list):
            for ii, p in enumerate(pos):
                val = self._sanitize(value[ii])
                self.__dict__[attr][p] = val
        else:
            new_vals = np.zeros_like(value)
            for ii, p in enumerate(pos):
                new_vals[ii] = self._sanitize(value[ii])
            self.__dict__[attr][:] = new_vals

    def set_all(self, attr, value):
        """
        Set attributes of the BaseParam class to new values for all positions.

        Parameters
        ----------
        attr : 'v', 'vin'
            Name of the attribute to be set
        value : list of str, float or int
            New values
        """
        if len(value) != self.n:
            raise ValueError("Value length does not match parameter numbers")

        if isinstance(self.__dict__[attr], list):
            for ii, val in enumerate(value):
                val = self._sanitize(val)
                self.__dict__[attr][ii] = val
        else:
            new_vals = np.zeros_like(value)
            for ii, val in enumerate(value):
                new_vals[ii] = self._sanitize(val)
            self.__dict__[attr][:] = new_vals

    def _sanitize(self, value):
        """
        Helper function for sanitizing parameter value.
        """

        if isinstance(value, float) and math.isnan(value):
            value = None

        # check for mandatory
        if value is None:
            if self.get_property('mandatory'):
                raise ValueError(f'Mandatory parameter {self.name} for {self.owner.class_name} missing')
            else:
                value = self.default

        return value

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
        return len(self.v)

    def __repr__(self):
        span = ''
        if 1 <= self.n <= 20:
            span = f', v={self.v}'
            if hasattr(self, 'vin') and (self.vin is not None):
                span += f', vin={self.vin}'

        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}{span}'


class DataParam(BaseParam):
    """
    An alias of the `BaseParam` class.

    This class is used for string parameters or non-computational numerical parameters.
    This class does not provide a `to_array` method.
    All input values will be stored in `v` as a list.

    See Also
    --------
    andes.core.param.BaseParam : Base parameter class

    """
    pass


class IdxParam(BaseParam):
    """
    An alias of `BaseParam` with an additional storage of the owner model name

    This class is intended for storing `idx` into other models.
    It can be used in the future for data consistency check.

    Examples
    --------
    A PQ model connected to Bus model will have the following code ::

        class PQModel(...):
            def __init__(...):
                ...
                self.bus = IdxParam(model='Bus')

    Notes
    -----
    This will be useful when, for example, one connects two TGs to one SynGen.
    """
    def __init__(self,
                 default: Optional[Union[float, str, int]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 mandatory: bool = False,
                 unique: bool = False,
                 export: bool = True,
                 model: Optional[str] = None,
                 iconvert: Optional[Callable] = None,
                 oconvert: Optional[Callable] = None,
                 ):
        super().__init__(default=default, name=name, tex_name=tex_name,
                         info=info, unit=unit, mandatory=mandatory,
                         export=export, iconvert=iconvert, oconvert=oconvert,
                         )
        self.property['unique'] = unique
        self.model = model  # must be a `Model` name for building BackRef - Not checked yet

    def add(self, value=None):
        if self.get_property('unique'):
            if value in self.v:
                logger.error('Your input data may be inconsistent.')
                raise IndexError(f'Unique parameter {self.owner.class_name}.{self.name} '
                                 f'contains duplicate value <{value}>.')

        super().add(value)


class NumParam(BaseParam):
    """
    A computational numerical parameter.

    Parameters defined using this class will have their `v`
    field converted to a NumPy array after adding.

    The original input values will be copied to `vin`,
    and the system-base per-unit conversion coefficients
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
    vrange : list, tuple, optional
        Typical value range
    vtype : type, optional
        Type of the ``v`` field. The default is ``float``.

    Other Parameters
    ----------------
    Sn : str
        Name of the parameter for the device base power.
    Vn : str
        Name of the parameter for the device base voltage.
    non_zero : bool
        True if this parameter must be non-zero. `non_zero`
        can be combined with `non_positive` or `non_negative`.
    non_positive : bool
        True if this parameter must be non-positive.
    non_negative : bool
        True if this parameter must be non-negative.
    mandatory : bool
        True if this parameter must not be None.
    power : bool
        True if this parameter is a power per-unit quantity
        under the device base.
    iconvert : callable
        Callable to convert input data from excel or others
        to the internal ``v`` field.
    oconvert : callable
        Callable to convert input data from internal type
        to a serializable type.
    ipower : bool
        True if this parameter is an inverse-power per-unit
        quantity under the device base.
    voltage : bool
        True if the parameter is a voltage pu quantity
        under the device base.
    current : bool
        True if the parameter is a current pu quantity
        under the device base.
    z : bool
        True if the parameter is an AC impedance pu quantity
        under the device base.
    y : bool
        True if the parameter is an AC admittance pu quantity
        under the device base.
    r : bool
        True if the parameter is a DC resistance pu quantity
        under the device base.
    g : bool
        True if the parameter is a DC conductance pu quantity
        under the device base.
    dc_current : bool
        True if the parameter is a DC current pu quantity under
        device base.
    dc_voltage : bool
        True if the parameter is a DC voltage pu quantity under
        device base.

    """

    def __init__(self,
                 default: Optional[Union[float, str, Callable]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 vrange: Optional[Union[List, Tuple]] = None,
                 vtype: Optional[Type] = float,
                 iconvert: Optional[Callable] = None,
                 oconvert: Optional[Callable] = None,
                 non_zero: bool = False,
                 non_positive: bool = False,
                 non_negative: bool = False,
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
                                       unit=unit, export=export, iconvert=iconvert, oconvert=oconvert,
                                       )

        self.property = dict(non_zero=non_zero,
                             non_positive=non_positive,
                             non_negative=non_negative,
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
        self.vrange = vrange
        self.vtype = vtype

    def add(self, value=None):
        """
        Add a value to the parameter value list.

        In addition to ``BaseParam.add``, this method checks for non-zero property and reset to default if is zero.

        See Also
        --------
        BaseParam.add : the add method of BaseParam

        """

        if hasattr(self, 'iconvert') and callable(self.iconvert):
            value = self.iconvert(value)

        # check for math.nan, usually imported from pandas
        if isinstance(value, float) and math.isnan(value):
            value = None

        # check for mandatory
        if value is None:
            if self.get_property('mandatory'):
                raise ValueError(f'Mandatory parameter {self.name} missing')
            else:
                value = self.default

        if isinstance(value, float):
            # check for non-zero
            if value == 0.0 and self.get_property('non_zero'):
                logger.warning('Non-zero parameter %s.%s corrected to %s',
                               self.owner.class_name, self.name, self.default)
                value = self.default

            # check for non-positive
            if value > 0.0 and self.get_property('non_positive'):
                logger.warning('Non-Positive parameter %s.%s corrected to %s',
                               self.owner.class_name, self.name, self.default)
                value = self.default

            # check for non-negative
            if value < 0.0 and self.get_property('non_negative'):
                logger.warning('Non-negative parameter %s.%s corrected to %s',
                               self.owner.class_name, self.name, self.default)
                value = self.default

        super(NumParam, self).add(value)

    def to_array(self):
        """
        Converts field ``v`` to the NumPy array type.
        to enable array-based calculation.

        Must be called after adding all elements.
        Store a copy of original input values to field ``vin``.
        Set ``pu_coeff`` to all ones.

        Warnings
        --------
        After this call, `add` will not be allowed to avoid
        unexpected issues.
        """

        self.v = np.array(self.v, dtype=self.vtype)

        # data quality check
        # ----------------------------------------
        # NOTE: temporarily disabled due to nested parameters
        # if np.sum(np.isnan(self.v)) > 0:
        #     raise ValueError(f'Param <{self.name} contains NaN.')

        if self.v.dtype != object:
            self.v[self.v == np.inf] = 1e8
            self.v[self.v == -np.inf] = -1e8
        # ----------------------------------------

        self.vin = np.array(self.v, dtype=self.vtype)

        self.pu_coeff = np.ones_like(self.v, dtype=float)

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
        if self.pu_coeff.ndim == 1:
            self.pu_coeff[:] = coeff

        elif self.pu_coeff.ndim == 2:
            for ii in range(len(self.pu_coeff)):
                self.pu_coeff[ii] = coeff[ii]
        else:
            raise NotImplementedError("Parameters with ndim > 2 not understood.")

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

    The constructor takes an additional Callable `self.callback` for the action of the event.
    `TimerParam` has a default value of -1, meaning deactivated.

    Examples
    --------
    A connectivity status toggler class `Toggler` takes a parameter `t` for the toggle time.
    Inside ``Toggler.__init__``, one would have ::

        self.t = TimerParam()

    The `Toggler` class also needs to define a method for togging the connectivity status ::

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

    Finally, in ``Toggler.__init__``, assign the function as the callback for `self.t` ::

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
        `np.equal`.

        Parameters
        ----------
        dae_t : float
            Current simulation time

        Returns
        -------
        np.ndarray
            The array containing the truth value of if the DAE time is close to the parameter value.

        Notes
        -----
        The previous implementation with `np.isclose` with default `rtol=1e-5` mistakes
        the immediate pre- and post-event time as in-event when simulation time is greater than 10.
        """

        return np.equal(dae_t, self.v)


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
        A parameter defined in the model defining this ExtParam instance. `indexer.v` should contain indices into
        `model.src.v`. If is None, the source parameter values will be fully copied. If `model` is a group
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
                 vtype=float,
                 allow_none=False,
                 default=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.src = src
        self.indexer = indexer
        self.vtype = vtype
        self.parent_model = None   # parent model instance
        self.allow_none = allow_none
        self.default = default

    def add(self, value=None):
        """
        ExtParam has an empty `add` method.
        """
        pass

    def restore(self):
        """
        ExtParam has an empty `restore` method
        """
        pass

    def link_external(self, ext_model):
        """
        Update parameter values provided by external models. This needs to be called before pu conversion.

        Parameters
        ----------
        ext_model : Model, Group
            Instance of the parent model or group, provided by the System calling this method.

        """
        self.parent_model = ext_model

        if hasattr(ext_model, "_idx2model"):
            # copy properties from models in the group

            # TODO: the three `get` calls below is a bit inefficient - same loops for three times
            try:
                self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr='v',
                                       allow_none=self.allow_none, default=self.default)
            except IndexError:
                pass

            try:
                self.vin = ext_model.get(src=self.src, idx=self.indexer.v, attr='vin',
                                         allow_none=self.allow_none, default=self.default)
                self.pu_coeff = ext_model.get(src=self.src, idx=self.indexer.v, attr='vin',
                                              allow_none=self.allow_none, default=self.default)
            except KeyError:  # idx param without vin
                pass
            except TypeError:  # vin or pu_coeff is None
                pass

        else:
            if self.allow_none:
                raise NotImplementedError(f"{self.name}: allow_none not implemented for Model")

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
            if isinstance(parent_instance.v, np.ndarray):
                self.v = parent_instance.v[uid]
            else:
                self.v = [parent_instance.v[i] for i in uid]
            try:
                self.vin = parent_instance.vin[uid]
                self.pu_coeff = parent_instance.pu_coeff[uid]
            except KeyError:
                pass
            except AttributeError:
                pass

    def to_array(self):
        """
        Convert to array when d_type is not str
        """
        if self.vtype == str:
            return
        NumParam.to_array(self)
