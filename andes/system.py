import importlib
from operator import itemgetter
from logging import DEBUG, INFO, WARNING, CRITICAL, ERROR
from .variables import FileMan, DevMan, DAE, VarName, VarOut, Call, Report
from .settings import Settings, SPF, TDS, CPF, SSSA
from .utils import Logger
from .models import non_jits, jits, JIT


class PowerSystem(object):
    """everything in a power system class including models, settings,
     file and call managers"""
    def __init__(self, case='', pid=0, verbose=INFO, no_output=False, log=None, dump=None, output=None,
                 addfile=None, settings=None, input_format=None, output_format=None, gis=None, **kwargs):
        """
        Initialize an empty power system object with defaults
        Args:
            case: case file name
            pid: process idx
            verbose: logging verbose level
            no_output: disable all output
            log: log file name
            dump: simulation result dump name
            addfile: additional file used by some formats
            settings: specified setting file name
            input_format: specified input case file format
            output_format: specified dump case file format
            output: specified output case file name
            gis: JML formatted GIS file name
            **kwargs: all other kwargs

        Returns: None
        """
        self.pid = pid
        self.Files = FileMan(case, input_format, addfile, settings, no_output,
                             log, dump, output_format, output, gis, **kwargs)
        self.Settings = Settings()
        self.SPF = SPF()
        self.CPF = CPF()
        self.TDS = TDS()
        self.SSSA = SSSA()
        if settings:
            self.load_settings(self.Files)
        self.Settings.verbose = verbose
        self.Log = Logger(self)

        self.DevMan = DevMan(self)
        self.Call = Call(self)
        self.DAE = DAE(self)
        self.VarName = VarName(self)
        self.VarOut = VarOut(self)
        self.Report = Report(self)

        self.inst_models()

    def setup(self):
        """set up everything after receiving the inputs"""
        self.Call.setup()
        self.dev_setup()
        self.xy_addr0()
        self.DAE.setup()

    def inst_models(self):
        """instantiate non-JIT models and import JIT models"""
        # non-JIT models
        for file, pair in non_jits.items():
            for cls, name in pair.items():
                try:
                    themodel = importlib.import_module('andes.models.' + file)
                    theclass = getattr(themodel, cls)
                    self.__dict__[name] = theclass(self, name)
                    self.DevMan.register_device(name)
                except ImportError:
                    self.Log.error('Error adding non-JIT model <{:s}.{:s}>.'.format(file, cls))

        # import JIT models
        for file, pair in jits.items():
            for cls, name in pair.items():
                self.__dict__[name] = JIT(self, file, cls, name)
                # do not register device. register after JIT loading

    def dev_setup(self):
        """set up models after data input"""
        for device in self.DevMan.devices:
            self.__dict__[device].setup()

    def xy_addr0(self):
        """assign x y indicies for power flow"""
        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._varname()

    def xy_addr1(self):
        """assign x y indices after power flow"""
        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if not pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._varname()

    def init_pf(self):
        """run models.init0() for power flow"""
        self.DAE.init_xy()
        for device, pflow, init0 in zip(self.DevMan.devices, self.Call.pflow, self.Call.init0):
            if pflow and init0:
                self.__dict__[device].init0(self.DAE)

    def base(self):
        """per-unitize model parameters"""
        for item in self.DevMan.devices:
            self.__dict__[item].base()

    def td_init(self):
        """run models.init1() time domain simulation"""

        # Assign indices for post-powerflow device variables
        self.VarName.resize()
        self.xy_addr1()

        # Reshape DAE to retain power flow solutions
        self.DAE.init1()

        # Initialize post-powerflow device variables
        for device, init1 in zip(self.DevMan.devices, self.Call.init1):
            if init1:
                self.__dict__[device].init1(self.DAE)

    def rmgen(self, idx):
        """remove static generators if dynamic ones exist"""
        stagens = []
        for device, stagen in zip(self.DevMan.devices, self.Call.stagen):
            if stagen:
                stagens.append(device)
        for gen in idx:
            for stagen in stagens:
                if gen in self.__dict__[stagen].int.keys():
                    self.__dict__[stagen].disable_gen(gen)

    def load_settings(self, Files):
        """load settings from file"""
        self.Log.debug('Loaded specified settings file.')
        raise NotImplemented

    def check_islands(self):
        """check connectivity for the ac system"""
        if not hasattr(self, 'Line'):
            self.Log.error('<Line> device not found.')
            return
        self.Line.connectivity(self.Bus)




