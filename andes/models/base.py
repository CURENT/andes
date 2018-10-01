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
Base class for building ANDES models
"""

import sys
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # NOQA

from cvxopt import matrix, spmatrix  # NOQA
from cvxopt import mul, div

from ..utils.math import agtb, altb, index, zeros  # NOQA
from ..utils.tab import Tab

# pandas import slows down the program startup by 0.3 seconds
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


class ModelBase(object):
    """base class for power system device models"""

    def __init__(self, system, name):
        """meta-data to be overloaded by subclasses"""

        # metadata list and dictionaries
        self._param_attr_lists = ('_params', '_zeros', '_mandatory', '_powers',
                                  '_voltages', '_currents', '_z', '_y', '_r',
                                  '_g', '_dccurrents', '_dcvoltages', '_times')

        self._param_attr_dicts = ('_data', '_units', '_descr')

        self.system = system
        # self.n = 0    # device count
        self.u = []  # device status
        self.idx = []  # external index list
        self.uid = {}  # mapping from `self.idx` to unique positional indices
        self.name = []  # element name list

        # identifications
        self._name = name
        self._group = None
        self._category = None

        # interfaces
        self._ac = {}  # ac bus variables
        self._dc = {}  # dc node variables
        self._ctrl = {}  # controller interfaces

        # variables
        self._states = []
        self._algebs = []
        self._algebs_intf = []

        self._states_descr = {}
        self._algebs_descr = {}

        # variable names
        self._unamex = []
        self._unamey = []
        self._fnamex = []
        self._fnamey = []

        # parameters to be converted to matrix
        self._params = ['u', 'Sn', 'Vn']

        # default parameter data
        self._data = {
            'u': 1,
            'Sn': 100.0,
            'Vn': 110.0,
        }

        # units of parameters
        self._units = {
            'u': 'bool',
            'Sn': 'MVA',
            'Vn': 'kV',
        }

        # variable descriptions
        self._descr = {
            'u': 'Connection status',
            'Sn': 'Power rating',
            'Vn': 'Ac Voltage rating',
        }
        # non-zero parameters
        self._zeros = ['Sn', 'Vn']

        # mandatory variables
        self._mandatory = []

        # service/temporary variables
        self._service = []
        self._service_ty = []

        self._store = {}
        self._snapshot = {}

        # equation storage
        self._equations = []

        # parameters to be per-unitized
        self._powers = []  # powers, inertia and damping
        self._voltages = []  # ac voltages
        self._currents = []  # ac currents
        self._z = []  # ac impedance
        self._y = []  # ac admittance

        self._dccurrents = []  # dc currents
        self._dcvoltages = []  # dc voltages
        self._r = []  # dc resistance
        self._g = []  # dc susceptance

        self._times = []  # time constants

        self._event_times = []  # event occurrance times

        # property functions this device has

        self.calls = dict(
            pflow=False,
            addr1=False,
            init0=False,
            init1=False,
            jac0=False,
            windup=False,
            gcall=False,
            fcall=False,
            gycall=False,
            fxcall=False,
            series=False,
            shunt=False,
            flows=False,
            dcseries=False,
        )

        self._flags = {
            'sysbase': False,
            'allocate': False,
            'address': False,
        }

        self._config = {
            'address_group_by': 'element',
            'is_series': False,
        }

        # pandas.DataFrame
        self.param_dict = {}
        self.param_df = {}

        self.define()

    def define(self):
        """
        Hook function where derived models define parameters, variables,
        service constants, and equations

        :return:
        """
        # raise NotImplemented('Subclasses must overwrite this method')
        pass

    def _init(self):
        """
        Convert model metadata to class attributes.

        This function is called automatically after ``define()`` in new
        versions.

        :return: None
        """
        assert self._name
        assert self._group

        # self.n = 0
        self.u = []
        self.name = []
        self.idx = []
        self.uid = {}

        if not self._unamey:
            self._unamey = self._algebs
        else:
            assert len(self._unamey) == len(self._algebs)

        if not self._unamex:
            self._unamex = self._states
        else:
            assert len(self._unamex) == len(self._states)

        for item in self._data.keys():
            self.__dict__[item] = []

        for bus in self._ac.keys():
            for var in self._ac[bus]:
                self.__dict__[var] = []
        for node in self._dc.keys():
            for var in self._dc[node]:
                self.__dict__[var] = []

        for var in self._states + self._algebs + self._service:
            self.__dict__[var] = []

        self._flags['sysbase'] = False
        self._flags['allocate'] = False
        self._flags['address'] = False

    def param_define(self,
                     param,
                     default,
                     unit='',
                     descr='',
                     tomatrix=True,
                     nonzero=False,
                     mandatory=False,
                     power=False,
                     voltage=False,
                     current=False,
                     z=False,
                     y=False,
                     r=False,
                     g=False,
                     dccurrent=False,
                     dcvoltage=False,
                     time=False,
                     event_time=False,
                     **kwargs):
        """
        Define a parameter in the model

        :param tomatrix: convert this parameter list to matrix
        :param param: parameter name
        :param default: parameter default value
        :param unit: parameter unit
        :param descr: description
        :param nonzero: is non-zero
        :param mandatory: is mandatory
        :param power: is a power value in the `self.Sn` base
        :param voltage: is a voltage value in the `self.Vn` base
        :param current: is a current value in the device base
        :param z: is an impedance value in the device base
        :param y: is an admittance value in the device base
        :param r: is a dc resistance value in the device base
        :param g: is a dc conductance value in the device base
        :param dccurrent: is a dc current value in the device base
        :param dcvoltage: is a dc votlage value in the device base
        :param time: is a time value in the device base
        :param event_time: is a variable for timed event

        :type param: str
        :type tomatrix: bool
        :type default: str, float
        :type unit: str
        :type descr: str
        :type nonzero: bool
        :type mandatory: bool
        :type power: bool
        :type voltage: bool
        :type current: bool
        :type z: bool
        :type y: bool
        :type r: bool
        :type g: bool
        :type dccurrent: bool
        :type dcvoltage: bool
        :type time: bool
        :type event_time: bool
        """
        assert param not in self._data
        assert param not in self._algebs
        assert param not in self._states
        assert param not in self._service

        self._data.update({param: default})
        if unit:
            self._units.update({param: unit})
        if descr:
            self._descr.update({param: descr})
        if tomatrix:
            self._params.append(param)
        if nonzero:
            self._zeros.append(param)
        if mandatory:
            self._mandatory.append(param)
        if power:
            self._powers.append(param)
        if voltage:
            self._voltages.append(param)
        if current:
            self._currents.append(param)
        if z:
            self._z.append(param)
        if y:
            self._y.append(param)
        if r:
            self._r.append(param)
        if g:
            self._g.append(param)
        if dccurrent:
            self._dccurrents.append(param)
        if dcvoltage:
            self._dcvoltages.append(param)
        if time:
            self._times.append(param)
        if event_time:
            self._event_times.append(param)

    def var_define(self, variable, ty, fname, descr='', uname=''):
        """
        Define a variable in the model

        :param fname: LaTex formatted variable name string
        :param uname: unformatted variable name string, `variable` as default
        :param variable: variable name
        :param ty: type code in ``('x', 'y')``
        :param descr: variable description

        :type variable: str
        :type ty: str
        :type descr: str
        :return:
        """
        assert ty in ('x', 'y')
        if not uname:
            uname = variable

        if ty == 'x':
            self._states.append(variable)
            self._fnamex.append(fname)
            self._unamex.append(uname)
            if descr:
                self._states_descr.update({variable: descr})
        elif ty == 'y':
            self._algebs.append(variable)
            self._fnamey.append(fname)
            self._unamey.append(uname)
            if descr:
                self._algebs_descr.update({variable: descr})

    def service_define(self, service, ty):
        """
        Add a service variable of type ``ty`` to this model

        :param str service: variable name
        :param type ty: variable type
        :return: None
        """

        assert service not in self._data
        assert service not in self._algebs + self._states

        self._service.append(service)
        self._service_ty.append(ty)

    def get_uid(self, idx):
        """
        Return the `uid` of the elements with the given `idx`

        :param list, matrix idx: external indices
        :type idx: list, matrix
        :return: a matrix of uid
        """
        assert idx is not None

        if isinstance(idx, (int, float, str)):
            return self.uid[idx]

        ret = []
        for i in idx:
            tmp = self.uid.get(i, None)
            assert tmp is not None, (
                'Model <{}> does not have element <{}>'.format(self._name, i))
            ret.append(self.uid[i])

        return ret

    def get_idx(self, uid):
        """
        Return the ``idx`` of the elements whose internal indices are ``uid``

        :param uid: uid of elements to query
        :return: idx of the elements
        """
        return [self.idx[i] for i in uid]

    def get_field(self, field, idx=None, astype=None):
        """
        Return `self.field` for the elements labeled by `idx`

        :param astype: type cast of the return value
        :param field: field name of this model
        :param idx: element indices, will be the whole list if not specified
        :return: field values
        """
        assert astype in (None, list, matrix)
        if idx is None:
            idx = self.idx

        # ===================disable warning ==============================
        # if field in self._service:
        # logger.warning(
        #     'Reading service variable <{model}.{field}> could be unsafe.'
        #         .format(field=field, model=self._name)
        # )
        # =================================================================

        uid = self.get_uid(idx)
        if not astype:
            astype = type(self.__dict__[field])

        ret = matrix(self.__dict__[field])[uid]

        if isinstance(idx, (float, int, str)):
            return ret
        else:
            return astype(ret)

    def _alloc(self):
        """
        Allocate empty memory for dae variable indices.
        Called in device setup phase.

        :return: None
        """
        nzeros = [0] * self.n
        for var in self._states:
            self.__dict__[var] = nzeros[:]
        for var in self._algebs:
            self.__dict__[var] = nzeros[:]

    def data_to_dict(self, sysbase=False):
        """
        Return the loaded model parameters as one dictionary.

        Each key of the dictionary is a parameter name, and the value is a
        list of all the parameter values.

        :param sysbase: use system base quantities
        :type sysbase: bool
        """
        assert isinstance(sysbase, bool)

        ret = {}

        for key in self.data_keys:
            if sysbase and (key in self._store):
                val = self._store[key]
            else:
                val = self.__dict__[key]

            ret[key] = val

        return ret

    @property
    def n(self):
        """
        Return the count of elements

        Returns
        -------
        int:
            count of elements
        """

        return len(self.idx)

    @property
    def data_keys(self):
        """
        Returns a list of all parameters plus ``name`` and ``idx``

        Returns
        -------
        list of parameters
        """

        keys = ['idx', 'name', 'u'] + sorted(self._data.keys())
        return set(keys)

    def data_to_list(self, sysbase=False):
        """
        Return the loaded model data as a list of dictionaries.

        Each dictionary contains the full parameters of an element.
        :param sysbase: use system base quantities
        :type sysbase: bool
        """
        ret = list()

        # for each element
        for i in range(self.n):
            # read the parameter values and put in the temp dict ``e``
            e = {}
            for key in self.data_keys:
                if sysbase and (key in self._store):
                    val = self._store[key][i]
                else:
                    val = self.__dict__[key][i]
                e[key] = val

            ret.append(e)

        return ret

    def data_from_list(self, data):
        """
        Populate model parameters from a list of parameter dictionaries

        Parameters
        ----------
        data: list
            List of parameter dictionaries

        Returns
        -------
        None
        """
        for item in data:
            self.elem_add(**item)

    def data_from_dict(self, data):
        """
        Populate model parameters from a dictionary of parameters

        Parameters
        ----------
        data: list
            List of parameter dictionaries

        Returns
        -------
        None
        """

        nvars = []
        for key, val in data.items():
            self.__dict__[key].extend(val)

            # assure the same parameter matrix size
            if len(nvars) > 1 and len(val) != nvars[-1]:
                raise IndexError(
                    'Model <{}> parameter <{}> must have the same length'.
                    format(self._name, key))
            nvars.append(len(val))

        # assign idx-uid mapping
        for i, idx in zip(range(self.n), self.idx):
            self.uid[idx] = i

    def data_to_df(self, sysbase=False):
        """
        Return a pandas.DataFrame of device parameters.
        :param sysbase: save per unit values in system base
        """

        p_dict_comp = self.data_to_dict(sysbase=sysbase)
        self.param_df = pd.DataFrame(data=p_dict_comp)

        return self.param_df

    def var_to_df(self):
        """
        Return the current var_to_df of variables

        :return: pandas.DataFrame
        """
        ret = {}

        if self._flags['address'] is False:
            return pd.DataFrame.from_dict(ret)

        ret.update({'name': self.name})
        ret.update({'idx': self.idx})

        for x in self._states:
            idx = self.__dict__[x]
            ret.update({x: self.system.dae.x[idx]})
        for y in self._algebs:
            idx = self.__dict__[y]
            ret.update({y: self.system.dae.y[idx]})

        return pd.DataFrame.from_dict(ret)

    def param_remove(self, param: 'str') -> None:
        """
        Remove a param from this model

        :param param: name of the parameter to be removed
        :type param: str
        """
        for attr in self._param_attr_dicts:
            if param in self.__dict__[attr]:
                self.__dict__[attr].pop(param)

        for attr in self._param_attr_lists:
            if param in self.__dict__[attr]:
                self.__dict__[attr].remove(param)

    def param_alter(self,
                    param,
                    default=None,
                    unit=None,
                    descr=None,
                    tomatrix=None,
                    nonzero=None,
                    mandatory=None,
                    power=None,
                    voltage=None,
                    current=None,
                    z=None,
                    y=None,
                    r=None,
                    g=None,
                    dccurrent=None,
                    dcvoltage=None,
                    time=None,
                    **kwargs):
        """
        Set attribute of an existing parameter.
        To be used to alter an attribute inherited from parent models.

        See .. self.param_define for argument descriptions.
        """
        assert param in self._data, \
            'parameter <{}> does not exist in {}'.format(param, self._name)

        def alter_attr(p, attr, value):
            """Set self.__dict__[attr] for param based on value
            """
            if value is None:
                return
            elif (value is True) and (p not in self.__dict__[attr]):
                self.__dict__[attr].append(p)
            elif (value is False) and (p in self.__dict__[attr]):
                self.__dict__[attr].remove(p)
            else:
                self.log('No need to alter {} for {}'.format(attr, p))

        if default is not None:
            self._data.update({param: default})
        if unit is not None:
            self._units.update({param: unit})
        if descr is not None:
            self._descr.update({param: descr})

        alter_attr(param, '_params', tomatrix)
        alter_attr(param, '_zeros', nonzero)
        alter_attr(param, '_mandatory', mandatory)
        alter_attr(param, '_powers', power)
        alter_attr(param, '_voltages', voltage)
        alter_attr(param, '_currents', current)
        alter_attr(param, '_z', z)
        alter_attr(param, '_y', y)
        alter_attr(param, '_r', r)
        alter_attr(param, '_g', g)
        alter_attr(param, '_dcvoltages', dcvoltage)
        alter_attr(param, '_dccurrents', dccurrent)
        alter_attr(param, '_times', time)

    def eq_add(self, expr, var, intf=False):
        """
        Add an equation to this model.

        An equation is associated with the addresses of a variable.  The
        number of equations must equal that of variables.

        Stored to ``self._equations`` is a tuple of ``(expr, var, intf,
        ty)`` where ``ty`` is in ('f', 'g')

        :param str expr: equation expression
        :param str var: variable name to be associated with
        :param bool intf: if this equation is added to an interface equation,
            namely, an equation outside this model

        :return: None

        """

        # determine the type of the equation, ('f', 'g') based on var

        # We assume that all differential equations are only modifiable by
        #   the model itself
        # Only interface to algebraic varbailes of external models

        ty = ''
        if var in self._algebs:
            ty = 'g'
        elif var in self._states:
            ty = 'f'
        else:
            for _, item in self._ac:
                if var in item:
                    ty = 'g'
                    if intf is False:
                        intf = True
            for _, item in self._dc:
                if var == item:
                    ty = 'g'
                    if intf is False:
                        intf = True
        if ty == '':
            self.log(
                'Equation associated with interface variable {var} '
                'assumed as algeb'.
                format(var=var),
                DEBUG)
            ty = 'g'

        self._equations.append((expr, var, intf, ty))

    def read_data_ext(self, model: str, field: str, idx=None, astype=None):
        """
        Return a field of a model or group at the given indices

        :param str model: name of the group or model to retrieve
        :param str field: name of the field
        :param list, int, float str idx: idx of elements to access
        :param type astype: type cast
        :return:
        """
        ret = list()

        if model in self.system.devman.devices:
            ret = self.system.__dict__[model].get_field(field, idx)

        elif model in self.system.devman.group.keys():
            # ===============================================================
            # Since ``self.system.devman.group`` is an unordered dictionary,
            #   ``idx`` must be given to retrieve ``field`` across models.
            #
            # Returns a matrix by default
            # ===============================================================
            assert idx is not None, \
                'idx must be specified when accessing group fields'
            astype = matrix

            for item in idx:
                dev_name = self.system.devman.group[model].get(item, None)
                ret.append(self.read_data_ext(dev_name, field, idx=item))

        else:
            raise NameError(
                'Model or Group <{0}> does not exist.'.format(model))

        if (ret is None) or isinstance(ret, (int, float, str)):
            return ret
        elif astype is None:
            return ret
        else:
            return astype(ret)

    def copy_data_ext(self, model, field, dest=None, idx=None, astype=None):
        """
        Retrieve the field of another model and store it as a field.

        :param model: name of the source model being a model name or a group name
        :param field: name of the field to retrieve
        :param dest: name of the destination field in ``self``
        :param idx: idx of elements to access
        :param astype: type cast

        :type model: str
        :type field: str
        :type dest: str
        :type idx: list, matrix
        :type astype: None, list, matrix

        :return: None

        """
        # use default destination

        if not dest:
            dest = field
        assert dest not in self._states + self._algebs

        self.__dict__[dest] = self.read_data_ext(
            model, field, idx, astype=astype)

    def elem_add(self, idx=None, name=None, **kwargs):
        """
        Add an element of this model

        :param idx: element idx
        :param name: element name
        :param kwargs: keyword arguments of the parameters
        :return: allocated idx
        """

        idx = self.system.devman.register_element(dev_name=self._name, idx=idx)
        self.system.__dict__[self._group].register_element(self._name, idx)

        self.uid[idx] = self.n
        self.idx.append(idx)
        # self.n += 1

        if name is None:
            self.name.append(self._name + ' ' + str(self.n))
        else:
            self.name.append(name)

        # check mandatory parameters
        for key in self._mandatory:
            if key not in kwargs.keys():
                self.log(
                    'Mandatory parameter <{:s}.{:s}> missing'.format(
                        self.name[-1], key), ERROR)
                sys.exit(1)

        # set default values
        for key, value in self._data.items():
            self.__dict__[key].append(value)

        # overwrite custom values
        for key, value in kwargs.items():
            if key not in self._data:
                self.log(
                    'Parameter <{:s}.{:s}> is not used.'.format(
                        self.name[-1], key), WARNING)
                continue
            self.__dict__[key][-1] = value

            # check data consistency
            if not value and key in self._zeros:
                if key == 'Sn':
                    default = self.system.mva
                elif key == 'fn':
                    default = self.system.config.freq
                else:
                    default = self._data[key]
                self.__dict__[key][-1] = default
                self.log(
                    'Using default value for <{:s}.{:s}>'.format(
                        self.name[-1], key), WARNING)

        return idx

    def elem_remove(self, idx=None):
        """
        Remove elements labeled by idx from this model instance.

        :param list,matrix idx: indices of elements to be removed
        :return: None
        """
        if idx is not None:
            if idx in self.uid:
                key = idx
                item = self.uid[idx]
            else:
                self.log('The item <{:s}> does not exist.'.format(idx), ERROR)
                return None
        else:
            return None

        convert = False
        if isinstance(self.__dict__[self._params[0]], matrix):
            self._param_to_list()
            convert = True

        # self.n -= 1
        self.uid.pop(key, '')
        self.idx.pop(item)

        for x, y in self.uid.items():
            if y > item:
                self.uid[x] = y - 1

        for param in self._data:
            self.__dict__[param].pop(item)

        for param in self._service:
            if len(self.__dict__[param]) == (self.n + 1):
                if isinstance(self.__dict__[param], list):
                    self.__dict__[param].pop(item)
                elif isinstance(self.__dict__[param], matrix):
                    service = list(self.__dict__[param])
                    service.pop(item)
                    self.__dict__[param] = matrix(service)

        for x in self._states:
            if len(self.__dict__[x]):
                self.__dict__[x].pop(item)

        for y in self._algebs:
            if self.__dict__[y]:
                self.__dict__[y].pop(item)

        for key, param in self._ac.items():
            if isinstance(param, list):
                for subparam in param:
                    if len(self.__dict__[subparam]):
                        self.__dict__[subparam].pop(item)
            else:
                self.__dict__[param].pop(item)

        for key, param in self._dc.items():
            self.__dict__[param].pop(item)

        self.name.pop(item)
        if convert and self.n:
            self._param_to_matrix()

    def data_to_sys_base(self):
        """
        Converts parameters to system base. Stores a copy in ``self._store``.
        Sets the flag ``self.flag['sysbase']`` to True.

        :return: None
        """
        if (not self.n) or self._flags['sysbase']:
            return
        Sb = self.system.mva
        Vb = matrix([])
        if 'bus' in self._ac.keys():
            Vb = self.read_data_ext('Bus', 'Vn', idx=self.bus)
        elif 'bus1' in self._ac.keys():
            Vb = self.read_data_ext('Bus', 'Vn', idx=self.bus1)

        for var in self._voltages:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], self.Vn)
            self.__dict__[var] = div(self.__dict__[var], Vb)
        for var in self._powers:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], self.Sn)
            self.__dict__[var] /= Sb
        for var in self._currents:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], self.Sn)
            self.__dict__[var] = div(self.__dict__[var], self.Vn)
            self.__dict__[var] = mul(self.__dict__[var], Vb)
            self.__dict__[var] /= Sb
        if len(self._z) or len(self._y):
            Zn = div(self.Vn**2, self.Sn)
            Zb = (Vb**2) / Sb
            for var in self._z:
                self._store[var] = self.__dict__[var]
                self.__dict__[var] = mul(self.__dict__[var], Zn)
                self.__dict__[var] = div(self.__dict__[var], Zb)
            for var in self._y:
                self._store[var] = self.__dict__[var]
                if self.__dict__[var].typecode == 'd':
                    self.__dict__[var] = div(self.__dict__[var], Zn)
                    self.__dict__[var] = mul(self.__dict__[var], Zb)
                elif self.__dict__[var].typecode == 'z':
                    self.__dict__[var] = div(self.__dict__[var], Zn + 0j)
                    self.__dict__[var] = mul(self.__dict__[var], Zb + 0j)

        if len(self._dcvoltages) or len(self._dccurrents) or len(
                self._r) or len(self._g):
            dckey = sorted(self._dc.keys())[0]
            Vbdc = self.read_data_ext('Node', 'Vdcn', self.__dict__[dckey])
            Ib = div(Sb, Vbdc)
            Rb = div(Vbdc, Ib)

        for var in self._dcvoltages:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], self.Vdcn)
            self.__dict__[var] = div(self.__dict__[var], Vbdc)

        for var in self._dccurrents:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], self.Idcn)
            self.__dict__[var] = div(self.__dict__[var], Ib)

        for var in self._r:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = div(self.__dict__[var], Rb)

        for var in self._g:
            self._store[var] = self.__dict__[var]
            self.__dict__[var] = mul(self.__dict__[var], Rb)

        self._flags['sysbase'] = True

    def data_to_elem_base(self):
        """
        Convert parameter data to element base

        Returns
        -------
        None
        """
        if self._flags['sysbase'] is False:
            return

        for key, val in self._store:
            self.__dict__[key] = val

        self._flags['sysbase'] = False

    def setup(self):
        """
        Set up empty class structure and allocate empty memory
        for variable addresses.

        :return: None
        """
        if not self.n:
            return

        self._param_to_matrix()
        self._alloc()

    def _intf_network(self):
        """
        Retrieve the ac and dc network interface variable indices.

        :Example:

            ``self._ac = {'bus1': (a1, v1)}`` gives
             - indices: self.bus1
             - system.Bus.a -> self.a1
             - system.Bus.v -> self.v1

            ``self._dc = {'node1': v1}`` gives
              - indices: self.node1
              - system.Node.v-> self.v1

        :return: None
        """

        for key, val in self._ac.items():
            self.copy_data_ext(
                model='Bus', field='a', dest=val[0], idx=self.__dict__[key])
            self.copy_data_ext(
                model='Bus', field='v', dest=val[1], idx=self.__dict__[key])

        for key, val in self._dc.items():
            self.copy_data_ext(
                model='Node', field='v', dest=val, idx=self.__dict__[key])

        # check for interface voltage differences
        self._check_Vn()

    def _intf_ctrl(self):
        """
        Retrieve variable indices of controlled models.

        Control interfaces are specified in ``self._ctrl``.
        Each ``key:value`` pair has ``key`` being the variable names
        for the reference idx and ``value`` being a tuple of
        ``(model name, field to read, destination field, return type)``.

        :Example:

            ``self._ctrl = {'syn', ('Synchronous', 'omega', 'w', list)}``
             - indices: self.syn
             - self.w = list(system.Synchronous.omega)

        :return: None
        """

        for key, val in self._ctrl.items():
            model, field, dest, astype = val
            self.copy_data_ext(
                model, field, dest=dest, idx=self.__dict__[key], astype=astype)

    def _addr(self):
        """
        Assign dae addresses for algebraic and state variables.

        Addresses are stored in ``self.__dict__[var]``.
        ``dae.m`` and ``dae.n`` are updated accordingly.

        Returns
        -------
        None

        """
        group_by = self._config['address_group_by']

        assert not self._flags['address']
        assert group_by in ('element', 'variable')

        m0 = self.system.dae.m
        n0 = self.system.dae.n
        mend = m0 + len(self._algebs) * self.n
        nend = n0 + len(self._states) * self.n

        if group_by == 'variable':
            for idx, item in enumerate(self._algebs):
                self.__dict__[item] = list(
                    range(m0 + idx * self.n, m0 + (idx + 1) * self.n))
            for idx, item in enumerate(self._states):
                self.__dict__[item] = list(
                    range(n0 + idx * self.n, n0 + (idx + 1) * self.n))
        elif group_by == 'element':
            for idx, item in enumerate(self._algebs):
                self.__dict__[item] = list(
                    range(m0 + idx, mend, len(self._algebs)))
            for idx, item in enumerate(self._states):
                self.__dict__[item] = list(
                    range(n0 + idx, nend, len(self._states)))

        self.system.dae.m = mend
        self.system.dae.n = nend

        self._flags['address'] = True

    def _varname(self):
        """
        Set up variable names in ``self.system.varname``.

        Variable names follows the convention ``VariableName,Model Name``.
        A maximum of 24 characters are allowed for each variable.

        :return: None
        """
        if not self._flags['address']:
            self.log('Unable to assign Varname before allocating address',
                     ERROR)
            return

        varname = self.system.varname
        for i in range(self.n):
            iname = str(self.name[i])

            for e, var in enumerate(self._states):
                unamex = self._unamex[e]
                fnamex = self._fnamex[e]
                idx = self.__dict__[var][i]

                varname.unamex[idx] = '{} {}'.format(unamex, iname)[:24]
                varname.fnamex[idx] = '$' + '{}\ {}'.format(
                    fnamex, iname.replace(' ', '\\ '))[:24] + '$'

            for e, var in enumerate(self._algebs):
                unamey = self._unamey[e]
                fnamey = self._fnamey[e]
                idx = self.__dict__[var][i]

                varname.unamey[idx] = '{} {}'.format(unamey, iname)[:24]
                varname.fnamey[idx] = '$' + '{}\ {}'.format(
                    fnamey, iname.replace(' ', '\\ '))[:24] + '$'

    def _param_to_matrix(self):
        """
        Convert parameters defined in `self._params` to `cvxopt.matrix`

        :return None
        """
        for item in self._params:
            self.__dict__[item] = matrix(self.__dict__[item], tc='d')

    def _param_to_list(self):
        """
        Convert parameters defined in `self._param` to list

        :return None
        """
        for item in self._params:
            self.__dict__[item] = list(self.__dict__[item])

    def log(self, msg, level=INFO):
        """Record a line of log in logger

        :param str msg: content of the messag

        :param level: logging level
        :return: None

        """
        logger.log(level, '<{}> - '.format(self._name) + msg)

    def init_limit(self, key, lower=None, upper=None, limit=False):
        """ check if data is within limits. reset if violates"""
        above = agtb(self.__dict__[key], upper)
        for idx, item in enumerate(above):
            if item == 0.:
                continue
            maxval = upper[idx]
            self.log(
                '{0} <{1}.{2}> above its maximum of {3}.'.format(
                    self.name[idx], self._name, key, maxval), ERROR)
            if limit:
                self.__dict__[key][idx] = maxval

        below = altb(self.__dict__[key], lower)
        for idx, item in enumerate(below):
            if item == 0.:
                continue
            minval = lower[idx]
            self.log(
                '{0} <{1}.{2}> below its minimum of {3}.'.format(
                    self.name[idx], self._name, key, minval), ERROR)
            if limit:
                self.__dict__[key][idx] = minval

    def __repr__(self):
        ret = ''
        ret += '\n'
        ret += 'Model <{:s}> parameters in element base\n'.format(self._name)
        ret += self.data_to_df(sysbase=False).to_string()

        ret += '\n'
        ret += 'Model <{:s}> variable snapshot\n'.format(self._name)
        ret += self.var_to_df().to_string()

        return ret

    def __str__(self):
        super(ModelBase, self).__repr__()
        return '<{name}> with {n} elements [{mem}]'.format(
            name=self._name,
            mem=hex(id(self)),
            n=self.n,
        )

    def doc(self, export='plain'):
        """
        Build help document into a Texttable table

        :param ('plain', 'latex') export: export format
        :param save: save to file ``help_model.extension`` or not
        :param writemode: file write mode

        :return: None
        """
        title = '<{}.{}>'.format(self._group, self._name)
        table = Tab(export=export, title=title, descr=self.__doc__)
        rows = []
        keys = sorted(self._data.keys())
        for key in keys:
            val = self._data[key]
            suf = ''
            if key in self._mandatory:
                suf = ' *'

            elif key in self._powers + \
                    self._voltages + \
                    self._currents + \
                    self._z + \
                    self._y + \
                    self._dccurrents + \
                    self._dcvoltages + \
                    self._r + \
                    self._g + \
                    self._times:
                suf = ' #'

            c1 = key + suf
            c2 = self._descr.get(key, '')
            c3 = val
            c4 = self._units.get(key, '-')
            rows.append([c1, c2, c3, c4])

        table.add_rows(rows, header=False)
        table.header(['Parameter', 'Description', 'Default', 'Unit'])

        if export == 'plain':
            pass
        elif export == 'latex':
            raise NotImplementedError('LaTex output not implemented')

        return table.draw()

    def check_limit(self, varname, vmin=None, vmax=None):
        """
        Check if the variable values are within the limits.

        Return False if fails.

        """
        retval = True
        assert varname in self.__dict__

        if varname in self._algebs:
            val = self.system.dae.y[self.__dict__[varname]]
        elif varname in self._states:
            val = self.system.dae.x[self.__dict__[varname]]
        else:  # service or temporary variable
            val = matrix(self.__dict__[varname])

        vmin = matrix(self.__dict__[vmin])
        comp = altb(val, vmin)
        comp = mul(self.u, comp)
        for c, n, idx in zip(comp, self.name, range(self.n)):
            if c == 1:
                v = val[idx]
                vm = vmin[idx]
                self.log(
                    'Init of <{}.{}>={:.4g} is lower than min={:6.4g}'.format(
                        n, varname, v, vm), ERROR)
                retval = False

        vmax = matrix(self.__dict__[vmax])
        comp = agtb(val, vmax)
        comp = mul(self.u, comp)
        for c, n, idx in zip(comp, self.name, range(self.n)):
            if c == 1:
                v = val[idx]
                vm = vmax[idx]
                self.log(
                    'Init of <{}.{}>={:.4g} is higher than max={:.4g}'.format(
                        n, varname, v, vm), ERROR)
                retval = False
        return retval

    def on_bus(self, bus_idx):
        """
        Return the indices of elements on the given buses for shunt-connected
        elements

        :param bus_idx: idx of the buses to which the elements are connected
        :return: idx of elements connected to bus_idx

        """
        assert hasattr(self, 'bus')

        ret = []
        if isinstance(bus_idx, (int, float, str)):
            bus_idx = [bus_idx]

        for item in bus_idx:
            idx = []
            for e, b in enumerate(self.bus):
                if b == item:
                    idx.append(self.idx[e])
            if len(idx) == 1:
                idx = idx[0]
            elif len(idx) == 0:
                idx = None

            ret.append(idx)

        return ret

    def link_bus(self, bus_idx):
        """
        Return the indices of elements linking the given buses

        :param bus_idx:
        :return:
        """
        ret = []

        if not self._config['is_series']:
            self.log(
                'link_bus function is not valid for non-series model <{}>'.
                format(self.name))
            return []

        if isinstance(bus_idx, (int, float, str)):
            bus_idx = [bus_idx]

        fkey = list(self._ac.keys())
        if 'bus' in fkey:
            fkey.remove('bus')

        nfkey = len(fkey)
        fkey_val = [self.__dict__[i] for i in fkey]

        for item in bus_idx:
            idx = []
            key = []
            for i in range(self.n):
                for j in range(nfkey):
                    if fkey_val[j][i] == item:
                        idx.append(self.idx[i])
                        key.append(fkey[j])
                        # <= 1 terminal should connect to the same bus
                        break

            if len(idx) == 0:
                idx = None
            if len(key) == 0:
                key = None

            ret.append((idx, key))

        return ret

    def elem_find(self, field, value):
        """
        Return the indices of elements whose field first satisfies the given values

        ``value`` should be unique in self.field.
        This function does not check the uniqueness.

        :param field: name of the supplied field
        :param value: value of field of the elemtn to find
        :return: idx of the elements
        :rtype: list, int, float, str
        """
        if isinstance(value, (int, float, str)):
            value = [value]

        f = list(self.__dict__[field])
        uid = np.vectorize(f.index)(value)
        return self.get_idx(uid)

    def _check_Vn(self):
        """Check data consistency of Vn and Vdcn if connected to Bus or Node

        :return None
        """
        if hasattr(self, 'bus') and hasattr(self, 'Vn'):
            bus_Vn = self.read_data_ext('Bus', field='Vn', idx=self.bus)
            for name, bus, Vn, Vn0 in zip(self.name, self.bus, self.Vn,
                                          bus_Vn):
                if Vn != Vn0:
                    self.log(
                        '<{}> has Vn={} different from bus <{}> Vn={}.'.format(
                            name, Vn, bus, Vn0), WARNING)

        if hasattr(self, 'node') and hasattr(self, 'Vdcn'):
            node_Vdcn = self.read_data_ext('Node', field='Vdcn', idx=self.node)
            for name, node, Vdcn, Vdcn0 in zip(self.name, self.node, self.Vdcn,
                                               node_Vdcn):
                if Vdcn != Vdcn0:
                    self.log(
                        '<{}> has Vdcn={} different from node <{}> Vdcn={}.'
                        .format(name, Vdcn, node, Vdcn0), WARNING)

    #
    # def var_store_snapshot(self, variable='all'):
    #     """
    #     Store a snapshot of variable values to self._snapshot.
    #
    #     :param variable: name of the variable, or ``all``
    #     :return: None
    #     """
    #
    #     # store (variable name, equation name) in the list ``var_eq_pairs``
    #     # equation name is in ('x', 'y')
    #
    #     var_eq_pairs = []
    #     dae = self.system.dae
    #     if variable != 'all':
    #         if variable in self._states:
    #             var_eq_pairs.append((variable, 'x'))
    #         elif variable in self._algebs:
    #             var_eq_pairs.append((variable, 'y'))
    #         else:
    #             raise NameError('<{}> is not a variable of model {}'
    #                             .format(variable, self._name))
    #     else:
    #         for var in self._states:
    #             var_eq_pairs.append((var, 'x'))
    #         for var in self._algebs:
    #             var_eq_pairs.append((var, 'y'))
    #
    #     # retrieve values and store in ``self._snapshot``
    #
    #     for var, eq in var_eq_pairs:
    #         idx = self.__dict__[var]
    #         self._snapshot[var] = dae.__dict__[eq][idx]
