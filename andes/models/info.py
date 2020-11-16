"""
Model for storing information, such as system summary.

Not used in DAE calculations.
"""

from andes.core.model import ModelData, Model
from andes.core.param import DataParam


class Summary(ModelData, Model):
    """
    Class for storing system summary
    """
    def __init__(self, system, config):
        ModelData.__init__(self, three_params=False)

        self.field = DataParam(info='field name')
        self.comment = DataParam(info='information, comment, or anything')

        Model.__init__(self, system, config)
        self.group = 'Information'
