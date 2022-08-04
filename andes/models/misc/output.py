"""
Module for specifying output variables as part of the data file.
"""

import logging

import numpy as np
from andes.core.model import ModelData, Model
from andes.core.param import DataParam

logger = logging.getLogger(__name__)


class OutputData(ModelData):
    """
    Data for outputs.
    """

    def __init__(self):
        ModelData.__init__(self, three_params=False)

        self.model = DataParam(info='Name of the model', mandatory=True)
        self.varname = DataParam(info='Variable name', )
        self.dev = DataParam(info='Device name', )


class Output(OutputData, Model):
    """
    Model for specifying output models and/or variables.
    """

    def __init__(self, system, config):
        OutputData.__init__(self)
        Model.__init__(self, system, config)

        self.group = 'OutputSelect'
        self.xidx = []
        self.yidx = []

    def in1d(self, addr, v_code):
        """
        Helper function for finding boolean flags to indicate intersections.

        Parameters
        ----------
        idx : array-like
            indices to find
        v_code : str
            variable code in 'x' and 'y'
        """

        if v_code == 'x':
            return np.in1d(self.xidx, addr)
        if v_code == 'y':
            return np.in1d(self.yidx, addr)

        raise NotImplementedError("v_code <%s> not recognized" % v_code)

    def to_output_addr(self, item, check=False):
        """
        Convert DAE-based variable address to relative output addresses.

        Parameters
        ----------
        check : bool, optional, False by default
            If True, check if the address fully or partially exists.

        Returns
        -------
        np.ndarray
            An array containing the indices into the output matrix
        """

        addr = item.a
        v_code = item.v_code

        bool_intersect = self.in1d(addr, v_code)
        output_addr = np.where(bool_intersect)[0]

        if check is True:
            if len(output_addr) == 0:
                logger.info("<%s.%s> not found in <Output>, skipped.",
                            item.owner.class_name, item.name)

            if len(output_addr) != len(item.a):
                logger.info("<%s.%s> is partially stored by <Output>. Showing all saved data.",
                            item.owner.class_name, item.name)

        return output_addr
