from .base import ModelBase


class Zone(ModelBase):
    """Zone class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Topology'
        self._name = 'Zone'
        self._inst_meta()

        self.tieline = dict()
        self.buses = dict()

    def setup(self):
        # TODO: account for >1 area/region/zone
        super().setup()
        var = self._name.lower()
        for idx, int_idx in self.system.Bus.int.items():
            var_code = self.system.Bus.__dict__[var][int_idx]
            if var_code not in self.buses.keys():
                self.buses[var_code] = list()
            self.buses[var_code].append(idx)

        for idx, int_idx in self.system.Line.int.items():
            bus1 = self.system.Line.bus1[int_idx]
            bus2 = self.system.Line.bus2[int_idx]
            code1 = self.system.Bus.__dict__[var][self.system.Bus.int[bus1]]
            code2 = self.system.Bus.__dict__[var][self.system.Bus.int[bus2]]
            if code1 != code2:
                if code1 not in self.tieline.keys():
                    self.tieline[code1] = list()
                if code2 not in self.tieline.keys():
                    self.tieline[code2] = list()

                self.tieline[code1].append(idx)
                self.tieline[code2].append(idx)


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
