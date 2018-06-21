"""
ANDES, a power system simulation tool for research.

Copyright 2015-2017 Hantao Cui

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import importlib
from operator import itemgetter
from logging import DEBUG, INFO, WARNING, CRITICAL, ERROR
from .variables import FileMan, DevMan, DAE, VarName, VarOut, Call, Report
from .settings import Settings, SPF, TDS, CPF, SSSA
from .utils import Logger
from .models import non_jits, jits, JIT
from .consts import *


class PowerSystem(object):
    """everything in a power system class including models, settings,
     file and call managers"""
    def __init__(self, case='', pid=-1, verbose=INFO, no_output=False, log=None, dump_raw=None, output=None, dynfile=None,
                 addfile=None, settings=None, input_format=None, output_format=None, gis=None, dime=None, tf=None,
                 **kwargs):
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
        self.Files = FileMan(case, input_format, addfile, settings, no_output, dynfile,
                             log, dump_raw, output_format, output, gis, **kwargs)
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

        if tf:
            self.TDS.tf = tf

        self.inst_models()

    def setup(self):
        """set up everything after receiving the inputs"""
        self.DevMan.sort_device()
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
            if self.__dict__[device].n:
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

        # Assign variable names for bus injections and line flows if enabled
        self.VarName.resize_for_flows()
        self.VarName.bus_line_names()

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
        raise NotImplementedError

    def check_islands(self):
        """check connectivity for the ac system"""
        if not hasattr(self, 'Line'):
            self.Log.error('<Line> device not found.')
            return
        self.Line.connectivity(self.Bus)

    def get_busdata(self, dec=5):
        """get ac bus data from solved power flow"""
        if not self.SPF.solved:
            self.Log.error('Power flow not solved when getting bus data.')
            return tuple([False] * 7)
        idx = self.Bus.idx
        names = self.Bus.name
        Vm = [round(self.DAE.y[x], dec) for x in self.Bus.v]
        if self.SPF.usedegree:
            Va = [round(self.DAE.y[x] * rad2deg, dec) for x in self.Bus.a]
        else:
            Va = [round(self.DAE.y[x], dec) for x in self.Bus.a]

        Pg = [round(self.Bus.Pg[x], dec) for x in range(self.Bus.n)]
        Qg = [round(self.Bus.Qg[x], dec) for x in range(self.Bus.n)]
        Pl = [round(self.Bus.Pl[x], dec) for x in range(self.Bus.n)]
        Ql = [round(self.Bus.Ql[x], dec) for x in range(self.Bus.n)]
        return (list(x) for x in zip(*sorted(zip(idx, names, Vm, Va, Pg, Qg, Pl, Ql), key=itemgetter(0))))

    def get_nodedata(self, dec=5):
        """get dc node data from solved power flow"""
        if not self.Node.n:
            return
        if not self.SPF.solved:
            self.Log.error('Power flow not solved when getting bus data.')
            return tuple([False] * 7)
        idx = self.Node.idx
        names = self.Node.name
        V = [round(self.DAE.y[x], dec) for x in self.Node.v]
        return (list(x) for x in zip(*sorted(zip(idx, names, V), key=itemgetter(0))))

    def get_linedata(self, dec=5):
        """get line data from solved power flow"""
        if not self.SPF.solved:
            self.Log.error('Power flow not solved when getting line data.')
            return tuple([False] * 7)
        idx = self.Line.idx
        fr = self.Line.bus1
        to = self.Line.bus2
        Pfr = [round(self.Line.S1[x].real, dec) for x in range(self.Line.n)]
        Qfr = [round(self.Line.S1[x].imag, dec) for x in range(self.Line.n)]
        Pto = [round(self.Line.S2[x].real, dec) for x in range(self.Line.n)]
        Qto = [round(self.Line.S2[x].imag, dec) for x in range(self.Line.n)]
        Ploss = [i + j for i, j in zip(Pfr, Pto)]
        Qloss = [i + j for i, j in zip(Qfr, Qto)]
        return (list(x) for x in zip(*sorted(zip(idx, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss), key=itemgetter(0))))

