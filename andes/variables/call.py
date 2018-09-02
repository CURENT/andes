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
        call_strings = [
            'gcalls',
            'fcalls',
            'gycalls',
            'fxcalls',
            'jac0s',
        ]

        self.gisland = 'system.Bus.gisland(system.dae)\n'
        self.gyisland = 'system.Bus.gyisland(system.dae)\n'

        for item in all_calls + call_strings:
            self.__dict__[item] = []

    def setup(self):
        """
        setup the call list after case file is parsed and jit models are loaded
        """
        self.devices = self.system.devman.devices
        self.ndevice = len(self.devices)

        self.gcalls = [''] * self.ndevice
        self.fcalls = [''] * self.ndevice
        self.gycalls = [''] * self.ndevice
        self.fxcalls = [''] * self.ndevice
        self.jac0s = [''] * self.ndevice

        self.build_vec()
        self.build_strings()
        self._compile_newton()
        self._compile_fdpf()
        self._compile_pfload()
        self._compile_pfgen()
        self._compile_seriesflow()
        self._compile_int()
        self._compile_int_f()
        self._compile_int_g()
        self._compile_bus_injection()

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

    def build_strings(self):
        """build call string for each device"""
        for idx, dev in enumerate(self.devices):
            header = 'system.' + dev
            self.gcalls[idx] = header + '.gcall(system.dae)\n'
            self.fcalls[idx] = header + '.fcall(system.dae)\n'
            self.gycalls[idx] = header + '.gycall(system.dae)\n'
            self.fxcalls[idx] = header + '.fxcall(system.dae)\n'
            self.jac0s[idx] = header + '.jac0(system.dae)\n'

    def _compile_newton(self):
        """Newton power flow execution
                1. evaluate g and f;
                1.1. handle islanded buses by Bus.gisland()
                2. factorize when needed;
                3. evaluate Gy and Fx.
                3.1. take care of islanded buses by Bus.gyisland()
        """
        string = '"""\n'

        # evaluate algebraic equations g and differential equations f
        string += 'system.dae.init_fg()\n'
        for pflow, gcall, call in zip(self.pflow, self.gcall, self.gcalls):
            if pflow and gcall:
                string += call
        string += 'system.dae.reset_small_g()\n'
        string += '\n'
        for pflow, fcall, call in zip(self.pflow, self.fcall, self.fcalls):
            if pflow and fcall:
                string += call
        string += 'system.dae.reset_small_f()\n'

        # handle islanded buses in algebraic equations
        string += self.gisland
        string += '\n'

        # rebuild constant Jacobian elements if factorization needed
        string += 'if system.dae.factorize:\n'
        string += '    system.dae.init_jac0()\n'
        for pflow, jac0, call in zip(self.pflow, self.jac0, self.jac0s):
            if pflow and jac0:
                string += '    ' + call
        string += '    system.dae.temp_to_spmatrix(\'jac0\')\n'

        # evaluate Jacobians Gy and Fx
        string += 'system.dae.setup_FxGy()\n'
        for pflow, gycall, call in zip(self.pflow, self.gycall, self.gycalls):
            if pflow and gycall:
                string += call
        for pflow, fxcall, call in zip(self.pflow, self.fxcall, self.fxcalls):
            if pflow and fxcall:
                string += call

        # handle islanded buses in the Jacobian
        string += self.gyisland
        string += 'system.dae.temp_to_spmatrix(\'jac\')\n'

        string += '"""'
        if SHOW_PF_CALL:
            logger.debug(string)
        self.newton = compile(eval(string), '', 'exec')

    def _compile_fdpf(self):
        """Fast Decoupled Power Flow execution: Implement g(y)
        """
        string = '"""\n'
        string += 'system.dae.init_g()\n'
        for pflow, gcall, call in zip(self.pflow, self.gcall, self.gcalls):
            if pflow and gcall:
                string += call
        string += 'system.dae.reset_small_g()\n'
        string += '\n'
        string += '"""'
        self.fdpf = compile(eval(string), '', 'exec')

    def _compile_pfload(self):
        """Post power flow computation for load
                  S_gen  + S_line + [S_shunt  - S_load] = 0
        """
        string = '"""\n'
        string += 'system.dae.init_g()\n'
        for gcall, pflow, shunt, stagen, call in zip(
                self.gcall, self.pflow, self.shunt, self.stagen, self.gcalls):
            if gcall and pflow and shunt and not stagen:
                string += call
        string += '\n'
        string += 'system.dae.reset_small_g()\n'
        string += '"""'
        self.pfload = compile(eval(string), '', 'exec')

    def _compile_pfgen(self):
        """Post power flow computation for PV and SW"""
        string = '"""\n'
        string += 'system.dae.init_g()\n'
        for gcall, pflow, shunt, series, stagen, call in zip(
                self.gcall, self.pflow, self.shunt, self.series, self.stagen,
                self.gcalls):
            if gcall and pflow and (shunt or series) and not stagen:
                string += call
        string += '\n'
        string += 'system.dae.reset_small_g()\n'
        string += '"""'
        self.pfgen = compile(eval(string), '', 'exec')

    def _compile_bus_injection(self):
        """Impose injections on buses"""
        string = '"""\n'
        for device, series in zip(self.devices, self.series):
            if series:
                string += 'system.' + device + '.gcall(system.dae)\n'
        string += '\n'
        string += 'system.dae.reset_small_g()\n'
        string += self.gisland
        string += '"""'
        self.bus_injection = compile(eval(string), '', 'exec')

    def _compile_seriesflow(self):
        """Post power flow computation of series device flow"""
        string = '"""\n'
        for device, pflow, series in zip(self.devices, self.pflow,
                                         self.series):
            if pflow and series:
                string += 'system.' + device + '.seriesflow(system.dae)\n'
        string += '\n'
        string += '"""'
        self.seriesflow = compile(eval(string), '', 'exec')

    def _compile_int(self):
        """Time Domain Simulation routine execution"""
        string = '"""\n'

        # evaluate the algebraic equations g
        string += 'system.dae.init_fg(resetz=False)\n'
        for gcall, call in zip(self.gcall, self.gcalls):
            if gcall:
                string += call
        string += '\n'
        string += 'system.dae.reset_small_g()\n'

        # handle islands
        string += self.gisland

        # evaluate differential equations f
        for fcall, call in zip(self.fcall, self.fcalls):
            if fcall:
                string += call
        string += 'system.dae.reset_small_f()\n'
        string += '\n'

        fg_string = string + '"""'
        self.int_fg = compile(eval(fg_string), '', 'exec')

        # rebuild constant Jacobian elements if needed
        string += 'if system.dae.factorize:\n'
        string += '    system.dae.init_jac0()\n'
        for jac0, call in zip(self.jac0, self.jac0s):
            if jac0:
                string += '    ' + call
        string += '    system.dae.temp_to_spmatrix(\'jac0\')\n'

        # evaluate Jacobians Gy and Fx
        string += 'system.dae.setup_FxGy()\n'
        for gycall, call in zip(self.gycall, self.gycalls):
            if gycall:
                string += call
        string += '\n'
        for fxcall, call in zip(self.fxcall, self.fxcalls):
            if fxcall:
                string += call
        string += self.gyisland
        string += 'system.dae.temp_to_spmatrix(\'jac\')\n'

        string += '"""'
        if SHOW_INT_CALL:
            logger.debug(string)

        self.int = compile(eval(string), '', 'exec')

    def _compile_int_f(self):
        """Time Domain Simulation - update differential equations"""
        string = '"""\n'
        string += 'system.dae.init_f()\n'

        # evaluate differential equations f
        for fcall, call in zip(self.fcall, self.fcalls):
            if fcall:
                string += call
        string += 'system.dae.reset_small_f()\n'
        string += '"""'
        self.int_f = compile(eval(string), '', 'exec')

    def _compile_int_g(self):
        """Time Domain Simulation - update algebraic equations and Jacobian"""
        string = '"""\n'

        # evaluate the algebraic equations g
        string += 'system.dae.init_g()\n'
        for gcall, call in zip(self.gcall, self.gcalls):
            if gcall:
                string += call
        string += '\n'
        string += 'system.dae.reset_small_g()\n'

        # handle islands
        string += self.gisland

        # rebuild constant Jacobian elements if needed
        string += 'if system.dae.factorize:\n'
        string += '    system.dae.init_jac0()\n'
        for jac0, call in zip(self.jac0, self.jac0s):
            if jac0:
                string += '    ' + call
        string += '    system.dae.temp_to_spmatrix(\'jac0\')\n'

        # evaluate Jacobians Gy
        string += 'system.dae.setup_Gy()\n'
        for gycall, call in zip(self.gycall, self.gycalls):
            if gycall:
                string += call
        string += '\n'
        string += self.gyisland
        string += 'system.dae.temp_to_spmatrix(\'jac\')\n'

        string += '"""'
        self.int_g = compile(eval(string), '', 'exec')
