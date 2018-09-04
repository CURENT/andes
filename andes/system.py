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

import configparser
import importlib
import logging
import os
from operator import itemgetter

from . import routines
from .config import System
from .consts import pi
from .consts import rad2deg
from .models import non_jits, jits, JIT
from .utils import get_config_load_path
from .variables import FileMan, DevMan, DAE, VarName, VarOut, Call, Report

logger = logging.getLogger(__name__)


class PowerSystem(object):
    """
    A power system class to hold models, routines, DAE numerical values,
    file manager, call manager, variable names and valuable values.
    """

    def __init__(self,
                 case='',
                 pid=-1,
                 no_output=False,
                 dump_raw=None,
                 output=None,
                 dynfile=None,
                 addfile=None,
                 config=None,
                 input_format=None,
                 output_format=None,
                 gis=None,
                 tf=None,
                 **kwargs):
        """
        PowerSystem constructor

        Parameters
        ----------
        case : str, optional
            Path to the case file

        pid : idx, optional
            Process index

        no_output : bool, optional
            ``True`` to disable all updates
        dump_raw : None or str, optional
            Path to the file to dump the raw parameters in the dm format

        output : None or str, optional
            Output case file name, NOT implemented yet

        dynfile : None or str, optional
            Path to the dynamic file for some formats, for example, ``dyr``
            for PSS/E

        addfile : None or str, optional
            Path to the additional dynamic file in the dm format

        config : None or str, optional
            Path to the andes.conf file

        input_format : None or str, optional
            Suggested input file format

        output_format : None or str, optional
            Requested dump file format, NOT implemented yet

        gis : None or str, optional
            Path to the GIS file

        tf : None or float, optional
            Time-domain simulation end time

        kwargs : dict, optional
            Other keyword args
        """
        # set internal flags based on the arguments
        self.pid = pid
        self.files = FileMan(case,
                             input_format=input_format,
                             addfile=addfile,
                             config=config,
                             no_output=no_output,
                             dynfile=dynfile,
                             dump_raw=dump_raw,
                             output_format=output_format,
                             output=output,
                             **kwargs)

        self.config = System()
        self.routine_import()

        self.load_config(get_config_load_path(self.files.config))

        self.devman = DevMan(self)
        self.call = Call(self)
        self.dae = DAE(self)
        self.varname = VarName(self)
        self.varout = VarOut(self)
        self.report = Report(self)

        if tf:
            self.tds.config.tf = tf

        self.model_import()

    @property
    def freq(self):
        """System base frequency"""
        return self.config.freq

    @freq.setter
    def freq(self, freq):
        if freq <= 0:
            self.config.freq = 1
        else:
            self.config.freq = freq

    @property
    def wb(self):
        """System base radial frequency"""
        return 2 * pi * self.config.freq

    @property
    def mva(self):
        """System base power in mega voltage-ampere"""
        return self.config.mva

    @mva.setter
    def mva(self, mva):
        self.config.mva = mva

    def setup(self):
        """
        Set up the power system object by executing the following workflow:

         * Sort the loaded models to meet the initialization sequence
         * Create call strings for routines
         * Call the ``setup`` function of the loaded models
         * Assign addresses for the loaded models
         * Call ``dae.setup`` to assign memory for the numerical dae structure
         * Convert model parameters to the system base

        Returns
        -------
        PowerSystem
            The instance of the PowerSystem
        """

        self.devman.sort_device()
        self.call.setup()
        self.model_setup()
        self.xy_addr0()
        self.dae.setup()
        self.to_sysbase()

        return self

    def to_sysbase(self):
        """
        Convert model parameters to system base. This function calls the
        ``data_to_sys_base`` function of the loaded models.

        Returns
        -------
        None
        """
        if self.config.base:
            for item in self.devman.devices:
                self.__dict__[item].data_to_sys_base()

    def group_add(self, name='Ungrouped'):
        """
        Dynamically add a group instance to the system if not exist.

        Parameters
        ----------
        name : str, optional ('Ungrouped' as default)
            Name of the group

        Returns
        -------
        None
        """
        if not hasattr(self, name):
            self.__dict__[name] = Group(self, name)

    def model_import(self):
        """
        Import and instantiate the non-JIT models and the JIT models.

        Models defined in ``jits`` and ``non_jits`` in ``models/__init__.py``
        will be imported and instantiated accordingly.

        Returns
        -------
        None
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

                self.devman.register_device(name)

        # import JIT models
        for file, pair in jits.items():
            for cls, name in pair.items():
                self.__dict__[name] = JIT(self, file, cls, name)

    def routine_import(self):
        """
        Dynamically import routines as defined in ``routines/__init__.py``.

        The command-line argument ``--routine`` is defined in ``__cli__`` in
        each routine file. A routine instance will be stored in the system
        instance with the name being all lower case.

        For example, a routine for power flow study should be defined in
        ``routines/pflow.py`` where ``__cli__ = 'pflow'``. The class name
        for the routine should be ``Pflow``. The routine instance will be
        saved to ``PowerSystem.pflow``.

        Returns
        -------
        None
        """
        for r in routines.__all__:
            file = importlib.import_module('.' + r.lower(), 'andes.routines')
            self.__dict__[r.lower()] = getattr(file, r)(self)

    def model_setup(self):
        """
        Call the ``setup`` function of the loaded models. This function is
        to be called after parsing all the data files during the system set up.

        Returns
        -------
        None
        """
        for device in self.devman.devices:
            if self.__dict__[device].n:
                try:
                    self.__dict__[device].setup()
                except Exception as e:
                    raise e

    def xy_addr0(self):
        """
        Assign indicies and variable names for variables used in power flow

        For each loaded model with the ``pflow`` flag as ``True``, the following
        functions are called sequentially:

         * ``_addr()``
         * ``_intf_network()``
         * ``_intf_ctrl()``

        After resizing the ``varname`` instance, variable names from models
        are stored by calling ``_varname()``

        Returns
        -------
        None
        """
        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.varname.resize()

        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if pflow:
                self.__dict__[device]._varname()

    def xy_addr1(self):
        """
        Assign indices and variable names for variables after power flow.
        This function is for loaded models that do not have the ``pflow``
        flag as ``True``.
        """
        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if not pflow:
                self.__dict__[device]._addr()
                self.__dict__[device]._intf_network()
                self.__dict__[device]._intf_ctrl()

        self.varname.resize()

        for device, pflow in zip(self.devman.devices, self.call.pflow):
            if not pflow:
                self.__dict__[device]._varname()

    def rmgen(self, idx):
        """
        Remove the static generators if their dynamic models exist

        Parameters
        ----------
        idx : list
            A list of static generator idx
        Returns
        -------
        None
        """
        stagens = []
        for device, stagen in zip(self.devman.devices, self.call.stagen):
            if stagen:
                stagens.append(device)
        for gen in idx:
            for stagen in stagens:
                if gen in self.__dict__[stagen].uid.keys():
                    self.__dict__[stagen].disable_gen(gen)

    def check_event(self, sim_time):
        """
        Check for event occurrance for``Event`` group models at ``sim_time``

        Parameters
        ----------
        sim_time : float
            The current simulation time

        Returns
        -------
        list
            A list of model names who report (an) event(s) at ``sim_time``
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

        Returns
        -------
        list
            A sorted list of event times
        """
        times = []

        times.extend(self.Breaker.get_times())

        for model in self.__dict__['Event'].all_models:
            times.extend(self.__dict__[model].get_times())

        if times:
            times = sorted(list(set(times)))

        return times

    def load_config(self, conf_path):
        """
        Load config from an ``andes.conf`` file.

        This function creates a ``configparser.ConfigParser`` object to read
        the specified conf file and calls the ``load_config`` function of the
        config instances of the system and the routines.

        Parameters
        ----------
        conf_path : None or str
            Path to the Andes config file. If ``None``, the function body
            will not run.

        Returns
        -------
        None
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)

        self.config.load_config(conf)
        for r in routines.__all__:
            self.__dict__[r.lower()].config.load_config(conf)

        logger.debug('Loaded config file from {}.'.format(conf_path))

    def dump_config(self, file_path):
        """
        Dump system and routine configurations to an rc-formatted file.

        Parameters
        ----------
        file_path : str
            path to the configuration file. The user will be prompted if the
            file already exists.

        Returns
        -------
        None
        """
        if os.path.isfile(file_path):
            logger.debug('File {} alreay exist. Overwrite? [y/N]'.format(file_path))
            choice = input('File {} alreay exist. Overwrite? [y/N]'.format(file_path)).lower()
            if len(choice) == 0 or choice[0] != 'y':
                logger.info('File not overwritten.')
                return

        conf = self.config.dump_conf()
        for r in routines.__all__:
            conf = self.__dict__[r.lower()].config.dump_conf(conf)

        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config written to {}'.format(file_path))

    def check_islands(self, show_info=False):
        """
        Check the connectivity for the ac system

        Parameters
        ----------
        show_info : bool
            Show information when the system has islands. To be used when
            initializing power flow.

        Returns
        -------
        None
        """
        if not hasattr(self, 'Line'):
            logger.error('<Line> device not found.')
            return
        self.Line.connectivity(self.Bus)

        if show_info is True:

            if len(self.Bus.islanded_buses) == 0 and len(
                    self.Bus.island_sets) == 0:
                logger.debug('System is interconnected.')
            else:
                logger.info(
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
                logger.warning(
                    'Slack bus is not defined for {:g} island(s).'.format(
                        len(nosw_island)))
            if msw_island:
                logger.warning(
                    'Multiple slack buses are defined for {:g} island(s).'.
                    format(len(nosw_island)))
            else:
                logger.debug(
                    'Each island has a slack bus correctly defined.'.format(
                        nosw_island))

    def get_busdata(self, dec=5):
        """
        get ac bus data from solved power flow
        """
        if self.pflow.solved is False:
            logger.error('Power flow not solved when getting bus data.')
            return tuple([False] * 8)
        idx = self.Bus.idx
        names = self.Bus.name
        Vm = [self.dae.y[x] for x in self.Bus.v]
        if self.pflow.config.usedegree:
            Va = [self.dae.y[x] * rad2deg for x in self.Bus.a]
        else:
            Va = [self.dae.y[x] for x in self.Bus.a]

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
        if not self.pflow.solved:
            logger.error('Power flow not solved when getting bus data.')
            return tuple([False] * 7)
        idx = self.Node.idx
        names = self.Node.name
        V = [self.dae.y[x] for x in self.Node.v]
        return (list(x)
                for x in zip(*sorted(zip(idx, names, V), key=itemgetter(0))))

    def get_linedata(self, dec=5):
        """get line data from solved power flow"""
        if not self.pflow.solved:
            logger.error('Power flow not solved when getting line data.')
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
