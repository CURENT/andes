from .base import ModelBase


class Zone(ModelBase):
    """Zone class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Topology'
        self._name = 'Zone'
        self._inst_meta()

    def setup(self):
        # TODO: account for >1 area/region/zone
        super().setup()


class Area(Zone):
    """Area class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'Area'
        self._inst_meta()

    def setup(self):
        # TODO: account for >1 area/region/zone
        super().setup()


class Region(Zone):
    """Region class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'Region'
        self._params.extend(['Ptol', 'slack'])
        self._descr.update({'Ptol': 'Total transfer capacity',
                            'slack': 'slack bus idx',
                            })
        self._data.update({'Ptol': None,
                           'slack': None,
                           })
        self._powers.extend(['Ptol'])
        self._inst_meta()

    def setup(self):
        super().setup()
