from .base import ModelBase


class SeriesBase(ModelBase):
    """Base class for AC series devices"""
    def __init__(self, system, name):
        super(SeriesBase, self).__init__(system, name)
