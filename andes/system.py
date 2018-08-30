# ANDES, a power system simulation tool for research.
#
# Copyright 2015-2017 Hantao Cui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Power system class
"""

import importlib
from operator import itemgetter
from logging import INFO
from .variables import FileMan, DevMan, DAE, VarName, VarOut, Call, Report
from .config import Config, Pflow, TDS, CPF, SSSA
# from .utils import Logger, elapsed
from .utils import elapsed
import logging

from .models import non_jits, jits, JIT
from .consts import rad2deg

try:
    from .utils.streaming import Streaming
    STREAMING = True
except ImportError:
    STREAMING = False

from .routines.pflow import PowerFlow


class PowerSystem(object):
    """
    everything in a power system class including models, settings,
     file and call managers
    """

    def __init__(self,
                 case='',
                 pid=-1,
                 verbose=INFO,
                 no_output=False,
                 log=None,
                 dump_raw=None,
                 output=None,
                 dynfile=None,
                 addfile=None,
                 settings=None,
                 input_format=None,
                 output_format=None,
                 gis=None,
                 dime=None,
                 tf=None,
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
        self.Files = FileMan(case, input_format, addfile, settings, no_output,
                             dynfile, log, dump_raw, output_format, output,
                             gis, **kwargs)

        self.config = Config()
        self.SPF = Pflow()
        self.CPF = CPF()
        self.TDS = TDS()
        self.SSSA = SSSA()

        if settings:
            self.load_settings(self.Files)
        self.config.verbose = verbose
        self.log = logging.getLogger(__name__)

        self.DevMan = DevMan(self)
        self.Call = Call(self)
        self.DAE = DAE(self)
        self.VarName = VarName(self)
        self.VarOut = VarOut(self)
        self.Report = Report(self)

        self.groups = []
        self.status = {
            'pf_solved': False,
            'sys_base': False,
        }

        if dime:
            self.config.dime_enable = True
            self.config.dime_server = dime
        if tf:
            self.TDS.tf = tf

        if not STREAMING:
            self.Streaming = None
            self.config.dime_enable = False
        else:
            self.Streaming = Streaming(self)

        self.model_import()

        # import routines
        self.powerflow = PowerFlow(self)

    def setup(self):
        """
        set up everything after receiving the inputs

        :return: reference of self
        """
        self.DevMan.sort_device()
        self.Call.setup()
        self.model_setup()
        self.xy_addr0()
        self.DAE.setup()
        self.to_sysbase()

        return self

    def to_sysbase(self):
        """
        Convert model parameters to system base

        :return: None
        """

        if self.config.base:
            for item in self.DevMan.devices:
                self.__dict__[item].data_to_sys_base()

    def group_add(self, name='Ungrouped'):
        """
        Add group ``name`` to the system

        :param name: group name
        :return: None
        """
        if not hasattr(self, name):
            self.__dict__[name] = Group(self, name)

    def model_import(self):
        """
        Import and instantiate non-JIT models, and import JIT models

        :return: None
        """
        # non-JIT models
        for file, pair in non_jits.items():
            for cls, name in pair.items():
                themodel = importlib.import_module('andes.models.' + file)
                theclass = getattr(themodel, cls)
                self.__dict__[name] = theclass(self, name)

                group = self.__dict__[name]._group
                self.group_add(group)
                self.__dict__[group].register_model(name)

                self.DevMan.register_device(name)

        # import JIT models
        for file, pair in jits.items():
            for cls, name in pair.items():
                self.__dict__[name] = JIT(self, file, cls, name)

    def model_setup(self):
        """
        Run model ``setup()`` for models present.

        Called by ``PowerSystem.setup()`` after adding model elements

        :return: None
        """
        for device in self.DevMan.devices:
            if self.__dict__[device].n:
                try:
                    self.__dict__[device].setup()
                except Exception as e:
                    raise e

    def xy_addr0(self):
        """
        Assign indicies and variable names for variables used in power flow

        :return: None
        """
        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.VarName.resize()

        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if pflow:
                self.__dict__[device]._varname()

    def xy_addr1(self):
        """
        Assign indices and variable names for variables after power flow
        """
        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if not pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.VarName.resize()

        for device, pflow in zip(self.DevMan.devices, self.Call.pflow):
            if not pflow:
                self.__dict__[device]._varname()

    def pf_init(self):
        """
        Set power flow initial values by running ``init0()``
        """
        t, s = elapsed()

        self.DAE.init_xy()

        for device, pflow, init0 in zip(self.DevMan.devices, self.Call.pflow,
                                        self.Call.init0):
            if pflow and init0:
                self.__dict__[device].init0(self.DAE)

        # check for islands
        self.check_islands(show_info=True)

        t, s = elapsed(t)
        self.log.info('Power flow initialized in {:s}.\n'.format(s))

        return self

    def td_init(self):
        """
        Set time domain simulation initial values by ``init1()``

        :return: success flag
        """
        if self.powerflow.solved is False:
            return False

        t, s = elapsed()

        # Assign indices for post-powerflow device variables
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

        t, s = elapsed(t)

        if self.DAE.n:
            self.log.info('Dynamic models initialized in {:s}.'.format(s))
        else:
            self.log.info('No dynamic model loaded.')

        return self

    def rmgen(self, idx):
        """
        remove static generators if dynamic ones exist

        :return: None
        """
        stagens = []
        for device, stagen in zip(self.DevMan.devices, self.Call.stagen):
            if stagen:
                stagens.append(device)
        for gen in idx:
            for stagen in stagens:
                if gen in self.__dict__[stagen].uid.keys():
                    self.__dict__[stagen].disable_gen(gen)

    def check_event(self, sim_time):
        """
        Check for event occurrance for``Event`` group models at ``sim_time``

        :param sim_time: current simulation time
        :return: a list of models who report (an) event(s) at ``sim_time``
        """
        ret = []
        for model in self.__dict__['Event'].all_models:
            if self.__dict__[model].is_time(sim_time):
                ret.append(model)

        if self.Breaker.is_time(sim_time):
            ret.append('Breaker')

        return ret

    def get_event_times(self):
        """
        Return event times of Fault, Breaker and other timed events

        :return: a sorted list of event times
        """
        times = []

        times.extend(self.Breaker.get_times())

        for model in self.__dict__['Event'].all_models:
            times.extend(self.__dict__[model].get_times())

        if times:
            times = sorted(list(set(times)))

        return times

    def load_settings(self, Files):
        """
        load settings from file

        :return: None
        """
        self.log.debug('Loaded specified settings file.')
        raise NotImplementedError

    def check_islands(self, show_info=False):
        """
        Check connectivity for the ac system

        :return: None
        """
        if not hasattr(self, 'Line'):
            self.log.error('<Line> device not found.')
            return
        self.Line.connectivity(self.Bus)

        if show_info is True:

            if len(self.Bus.islanded_buses) == 0 and len(
                    self.Bus.island_sets) == 0:
                self.log.info('System is interconnected.')
            else:
                self.log.info(
                    'System contains {:d} islands and {:d} islanded buses.'.
                    format(
                        len(self.Bus.island_sets),
                        len(self.Bus.islanded_buses)))

            nosw_island = []  # no slack bus island
            msw_island = []  # multiple slack bus island
            for idx, island in enumerate(self.Bus.island_sets):
                nosw = 1
                for item in self.SW.bus:
                    if self.Bus.uid[item] in island:
                        nosw -= 1
                if nosw == 1:
                    nosw_island.append(idx)
                elif nosw < 0:
                    msw_island.append(idx)

            if nosw_island:
                self.log.warning(
                    'Slack bus is not defined for {:g} island(s).'.format(
                        len(nosw_island)))
            if msw_island:
                self.log.warning(
                    'Multiple slack buses are defined for {:g} island(s).'.
                    format(len(nosw_island)))
            else:
                self.log.debug(
                    'Each island has a slack bus correctly defined.'.format(
                        nosw_island))

    def get_busdata(self, dec=5):
        """
        get ac bus data from solved power flow
        """
        if self.powerflow.solved is False:
            self.log.error('Power flow not solved when getting bus data.')
            return tuple([False] * 8)
        idx = self.Bus.idx
        names = self.Bus.name
        Vm = [self.DAE.y[x] for x in self.Bus.v]
        if self.SPF.usedegree:
            Va = [self.DAE.y[x] * rad2deg for x in self.Bus.a]
        else:
            Va = [self.DAE.y[x] for x in self.Bus.a]

        Pg = [self.Bus.Pg[x] for x in range(self.Bus.n)]
        Qg = [self.Bus.Qg[x] for x in range(self.Bus.n)]
        Pl = [self.Bus.Pl[x] for x in range(self.Bus.n)]
        Ql = [self.Bus.Ql[x] for x in range(self.Bus.n)]
        return (list(x) for x in zip(*sorted(
            zip(idx, names, Vm, Va, Pg, Qg, Pl, Ql), key=itemgetter(0))))

    def get_nodedata(self, dec=5):
        """
        get dc node data from solved power flow
        """
        if not self.Node.n:
            return
        if not self.powerflow.solved:
            self.log.error('Power flow not solved when getting bus data.')
            return tuple([False] * 7)
        idx = self.Node.idx
        names = self.Node.name
        V = [self.DAE.y[x] for x in self.Node.v]
        return (list(x)
                for x in zip(*sorted(zip(idx, names, V), key=itemgetter(0))))

    def get_linedata(self, dec=5):
        """get line data from solved power flow"""
        if not self.powerflow.solved:
            self.log.error('Power flow not solved when getting line data.')
            return tuple([False] * 7)
        idx = self.Line.idx
        fr = self.Line.bus1
        to = self.Line.bus2
        Pfr = [self.Line.S1[x].real for x in range(self.Line.n)]
        Qfr = [self.Line.S1[x].imag for x in range(self.Line.n)]
        Pto = [self.Line.S2[x].real for x in range(self.Line.n)]
        Qto = [self.Line.S2[x].imag for x in range(self.Line.n)]
        Ploss = [i + j for i, j in zip(Pfr, Pto)]
        Qloss = [i + j for i, j in zip(Qfr, Qto)]
        return (list(x) for x in zip(*sorted(
            zip(idx, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss),
            key=itemgetter(0))))


class GroupMeta(type):
    def __new__(cls, name, base, attr_dict):
        return super(GroupMeta, cls).__new__(cls, name, base, attr_dict)


class Group(metaclass=GroupMeta):
    """
    Group class for registering models and elements.

    Also handles reading and setting attributes.
    """

    def __init__(self, system, name):
        self.system = system
        self.name = name
        self.all_models = []
        self._idx_model = {}

    def register_model(self, model):
        """
        Register ``model`` to this group

        :param model: model name
        :return: None
        """

        assert isinstance(model, str)
        if model not in self.all_models:
            self.all_models.append(model)

    def register_element(self, model, idx):
        """
        Register element with index ``idx`` to ``model``

        :param model: model name
        :param idx: element idx
        :return: final element idx
        """

        if idx is None:
            idx = model + '_' + str(len(self._idx_model))

        assert idx not in self._idx_model.values()

        self._idx_model[idx] = model
        return idx

    def get_field(self, field, idx):
        """
        Return the field ``field`` of elements ``idx`` in the group

        :param field: field name
        :param idx: element idx
        :return: values of the requested field
        """
        ret = []
        scalar = False

        if isinstance(idx, (int, float, str)):
            scalar = True
            idx = [idx]

        models = [self._idx_model[i] for i in idx]

        for i, m in zip(idx, models):
            ret.append(self.system.__dict__[m].get_field(field, idx=i))

        if scalar is True:
            return ret[0]
        else:
            return ret

    def set_field(self, field, idx, value):
        """
        Set the field ``field`` of elements ``idx`` to ``value``.

        This function does not if the field is valid for all models.

        :param field: field name
        :param idx: element idx
        :param value: value of fields to set
        :return: None
        """

        if isinstance(idx, (int, float, str)):
            idx = [idx]
        if isinstance(value, (int, float)):
            value = [value]

        models = [self._idx_model[i] for i in idx]

        for i, m, v in zip(idx, models, value):
            assert hasattr(self.system.__dict__[m], field)

            uid = self.system.__dict__[m].get_uid(idx)
            self.system.__dict__[m].__dict__[field][uid] = v
