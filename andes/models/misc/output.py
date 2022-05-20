"""
Module for specifying output variables as part of the data file.
"""

from andes.core.model import ModelData, Model
from andes.core.param import DataParam


class OutputData(ModelData):
    """
    Data for outputs.
    """

    def __init__(self):
        ModelData.__init__(self, three_params=False)

        self.model = DataParam(info='Name of the model', mandatory=True)
        self.varname = DataParam(info='Variable name', )


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
