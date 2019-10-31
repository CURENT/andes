import sys
from typing import Optional, Union, Callable

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ParamBase(object):
    """
    Basic single data class
    """
    def __init__(self, default=None, name=None, tex_name=None, descr=None, mandatory=False):
        self.name = name
        self.default = default
        self.tex_name = tex_name if tex_name else name
        self.descr = descr
        self.owner = None

        self.n = 0
        self.v = []
        self.property = dict(mandatory=mandatory)

    def add(self, value=None):
        """
        Add a new value (from a new element) to this parameter list

        Parameters
        ----------
        value
            Parameter value of the new element

        Returns
        -------
        None
        """

        # check for mandatory
        if value is None:
            value = self.default

        self.v.append(value)
        self.n += 1


class DataParam(ParamBase):
    pass


class NumParam(ParamBase):
    """
    Parameter class
    """

    def __init__(self,
                 default: Optional[Union[float, str, Callable]] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 descr: Optional[str] = None,
                 unit: Optional[str] = None,
                 nonzero: bool = False,
                 mandatory: bool = False,
                 power: bool = False,
                 voltage: bool = False,
                 current: bool = False,
                 z: bool = False,
                 y: bool = False,
                 r: bool = False,
                 g: bool = False,
                 dcvoltage: bool = False,
                 dccurrent: bool = False,
                 toarray: bool = True,
                 ):
        super(NumParam, self).__init__(default=default, name=name, tex_name=tex_name, descr=descr)
        self.unit = unit

        self.property = dict(nonzero=nonzero,
                             mandatory=mandatory,
                             power=power,
                             voltage=voltage,
                             toarray=toarray,
                             current=current,
                             z=z,
                             y=y,
                             r=r,
                             g=g,
                             dccurrent=dccurrent,
                             dcvoltage=dcvoltage)

        self.pu_coeff = None
        self.vin = None  # values from input

    def get_property(self, property_name):
        """
        Check the boolean value of the given property

        Parameters
        ----------
        property_name
            Property name

        Returns
        -------
        The truth value of the property
        """
        return self.property[property_name]

    def add(self, value=None):
        """
        Add a new value (from a new element) to this parameter list

        Parameters
        ----------
        value
            Parameter value of the new element

        Returns
        -------
        None
        """

        # check for mandatory
        if value is None:
            if self.get_property('mandatory'):
                logger.error(f'Mandatory parameter {self.name} missing')
                sys.exit(1)
            else:
                value = self.default

        # check for non-zero
        if value == 0 and self.get_property('nonzero'):
            logger.warning(f'Parameter {self.name} must be non-zero')
            value = self.default

        super(NumParam, self).add(value)

    def to_array(self):
        """
        Convert to np array for speed up

        Returns:

        """
        self.v = np.array(self.v)

    def get_name(self):
        return [self.name]


class ExtParam(NumParam):
    """
    External parameter, specified by other modules
    """
    def __init__(self, model: str, src: str, indexer=None, **kwargs):
        super(ExtParam, self).__init__(**kwargs)
        self.model = model
        self.src = src
        self.indexer = indexer

        self.parent_model = None   # parent model instance
        self.parent_instance = None
        self.uid = None

    def set_external(self, ext_model):
        """
        Update parameter values provided by external models
        Returns
        -------
        """
        self.parent_model = ext_model
        self.parent_instance = ext_model.__dict__[self.src]
        self.property = dict(self.parent_instance.property)

        if self.indexer is None:
            # if `indexer` is None, retrieve all the values
            self.uid = np.arange(ext_model.n)
        else:
            n_indexer = len(self.indexer.v)
            if n_indexer == 0:
                return
            else:
                self.uid = ext_model.idx2uid(self.indexer.v)

        # pull in values
        self.v = self.parent_instance.v[self.uid]
        self.vin = self.parent_instance.vin[self.uid]
        self.pu_coeff = self.parent_instance.pu_coeff[self.uid]
        self.n = len(self.v)
