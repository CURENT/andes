import logging

logger = logging.getLogger(__name__)

SHOW_PF_CALL = False
SHOW_INT_CALL = False

all_calls = [
    'gcall', 'gycall', 'fcall', 'fxcall', 'init0', 'pflow', 'windup', 'jac0',
    'init1', 'shunt', 'series', 'flows', 'connection', 'times', 'stagen',
    'dyngen', 'gmcall', 'fmcall', 'dcseries', 'opf', 'obj'
]


class Call(object):
    """ Equation call mamager class for andes routines"""

    def __init__(self, system):
        self.system = system
        self.ndevice = 0
        self.devices = []

        for item in all_calls:
            self.__dict__[item] = []

    def setup(self):
        """
        setup the call list after case file is parsed and jit models are loaded
        """
        self.devices = self.system.devman.devices
        self.ndevice = len(self.devices)

        self.build_vec()

    def build_vec(self):
        """build call validity vector for each device"""
        for item in all_calls:
            self.__dict__[item] = []

        for dev in self.devices:
            for item in all_calls:
                if self.system.__dict__[dev].n == 0:
                    val = False
                else:
                    val = self.system.__dict__[dev].calls.get(item, False)
                self.__dict__[item].append(val)

    def fdpf(self):
        system = self.system
        system.dae.init_g()
        for device, pflow, gcall in zip(self.devices, self.pflow):
            if pflow and gcall:
                system.__dict__[device].gcall(system.dae)
        system.dae.reset_small()

    def pfload(self):
        system = self.system
        system.dae.init_g()
        for device, gcall, pflow, shunt, stagen in zip(self.devices, self.gcall, self.pflow, self.shunt,
                                                       self.stagen):
            if gcall and pflow and shunt and not stagen:
                system.__dict__[device].gcall(system.dae)
        system.dae.reset_small_g()

    def pfgen(self):
        system = self.system
        system.dae.init_g()
        for device, gcall, pflow, shunt, series, stagen in zip(self.devices, self.gcall, self.pflow, self.shunt,
                                                               self.series, self.stagen):
            if gcall and pflow and (shunt or series) and not stagen:
                system.__dict__[device].gcall(system.dae)
        system.dae.reset_small_g()

    def bus_injection(self):
        system = self.system
        for device, series in zip(self.devices, self.series):
            if series:
                system.__dict__[device].gcall(system.dae)
        system.dae.reset_small_g()
        self.gisland()

    def gisland(self):
        self.system.Bus.gisland(self.system.dae)

    def gyisland(self):
        self.system.Bus.gyisland(self.system.dae)

    def seriesflow(self):
        system = self.system
        for device, pflow, series in zip(self.devices, self.pflow, self.series):
            if pflow and series:
                system.__dict__[device].seriesflow(system.dae)

    def int_fg(self):
        system = self.system
        system.dae.init_fg(resetz=False)
        for device, gcall in zip(self.devices, self.gcall):
            if gcall:
                system.__dict__[device].gcall(system.dae)
        system.dae.reset_small_g()
        self.gisland()

        for device, fcall in zip(self.devices, self.fcall):
            if fcall:
                system.__dict__[device].fcall(system.dae)
        system.dae.reset_small_f()

    def int_fxgy(self):
        system = self.system
        # rebuilt constant elements in jacobian if needed
        if system.dae.factorize:
            system.dae.init_jac0()
            for device, jac0 in zip(self.devices, self.jac0):
                if jac0:
                    system.__dict__[device].jac0(system.dae)
            system.dae.temp_to_spmatrix('jac0')

        system.dae.setup_FxGy()

        self.gyisland()
        for device, gycall in zip(self.devices, self.gycall):
            if gycall:
                system.__dict__[device].gycall(system.dae)

        for device, fxcall in zip(self.devices, self.fxcall):
            if fxcall:
                system.__dict__[device].fxcall(system.dae)

        system.dae.temp_to_spmatrix('jac')

    def int(self):
        self.int_fg()
        self.int_fxgy()
