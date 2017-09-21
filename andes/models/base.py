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
import sys
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

from cvxopt import matrix, spmatrix
from cvxopt import mul, div

from ..utils.math import agtb, altb, findeq
from ..utils.tab import Tab


class ModelBase(object):
    """base class for power system device models"""

    def __init__(self, system, name):
        """meta-data to be overloaded by subclasses"""
        self.system = system
        self.n = 0    # device count
        self.u = []   # device status
        self.idx = []    # internal index list
        self.int = {}    # external index to internal
        self.name = []  # element name list

        # identifications
        self._name = name
        self._group = None
        self._category = None

        # interfaces
        self._ac = {}    # ac bus variables
        self._dc = {}    # dc node variables
        self._ctrl = {}  # controller interfaces

        # variables
        self._states = []
        self._algebs = []

        # variable names
        self._unamex = []
        self._unamey = []
        self._fnamex = []
        self._fnamey = []

        # parameters to be converted to matrix
        self._params = ['u', 'Sn', 'Vn']

        # default parameter data
        self._data = {'u': 1,
                      'Sn': 100.0,
                      'Vn': 110.0,
                      }

        # units of parameters
        self._units = {'u': 'boolean',
                       'Sn': 'MVA',
                       'Vn': 'kV',
                       }

        # variable descriptions
        self._descr = {'u': 'Connection status',
                       'Sn': 'Power rating',
                       'Vn': 'AC Voltage rating',
                       }
        # non-zero parameters
        self._zeros = ['Sn', 'Vn']

        # mandatory variables
        self._mandatory = []

        # service/temporary variables
        self._service = []
        self._store = {}

        # parameters to be per-unitized
        self._powers = []      # powers, inertia and damping
        self._voltages = []    # ac voltages
        self._currents = []    # ac currents
        self._z = []           # ac impedance
        self._y = []           # ac admittance

        self._dccurrents = []  # dc currents
        self._dcvoltages = []  # dc voltages
        self._r = []           # dc resistance
        self._g = []           # dc susceptance

        self._times = []       # time constants

        # property functions this device has

        self.calls = dict(pflow=False, addr1=False,
                          init0=False, init1=False,
                          jac0=False, windup=False,
                          gcall=False, fcall=False,
                          gycall=False, fxcall=False,
                          series=False, shunt=False,
                          flows=False, dcseries=False,
                          )
        self.addr = False
        self.ispu = False

    def _inst_meta(self):
        """instantiate meta-data defined in __init__().
        Call this function at the end of __init__() of child classes
        """
        if not self._name:
            self._name = self._group
        if not self._unamey:
            self._unamey = self._algebs
        if not self._unamex:
            self._unamex = self._states

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

    def _alloc(self):
        """Allocate memory for DAE variable indices. Called after finishing adding components
        """
        zeros = [0] * self.n
        for var in self._states:
            self.__dict__[var] = zeros[:]
        for var in self._algebs:
            self.__dict__[var] = zeros[:]

    def remove_param(self, param):
        """Remove a param from this class"""
        if param in self._data.keys():
            self._data.pop(param)
        if param in self._descr.keys():
            self._descr.pop(param)
        if param in self._units:
            self._units.pop(param)
        if param in self._params:
            self._params.remove(param)
        if param in self._zeros:
            self._zeros.remove(param)
        if param in self._mandatory:
            self._mandatory.remove(param)

    def read_param(self, model, src, fkey=None):
        """Return the param of the `model` group or class indexed by fkey"""
        if not self.n:
            return
        # input check
        retval = None
        dev_type = None
        val = list()
        if model in self.system.DevMan.devices:
            dev_type = 'model'
        elif model in self.system.DevMan.group.keys():
            dev_type = 'group'
        if not dev_type:
            self.message('Model or group <{0}> does not exist.'.format(model), ERROR)
            return

        # do param copy
        if dev_type == 'model':
            # check if fkey exists
            for item in fkey:
                if item not in self.system.__dict__[model].idx:
                    self.message('Model <{}> does not have element <{}>'.format(model, item), ERROR)
                    return
            retval = self.system.__dict__[model]._slice(src, fkey)
            src_type = type(self.system.__dict__[model].__dict__[src])

        elif dev_type == 'group':
            if not fkey:
                fkey = self.system.DevMan.group.keys()
                if not fkey:
                    self.message('Group <{}> does not have any element.'.format(model), ERROR)
                    return
            for item in fkey:
                dev_name = self.system.DevMan.group[model].get(item, None)
                if not dev_name:
                    self.message('Group <{}> does not have element {}.'.format(model, item), ERROR)
                    return
                pos = self.system.__dict__[dev_name].int[item]
                val.append(self.system.__dict__[dev_name].__dict__[src][pos])

            retval = val
            src_type = type(self.system.__dict__[dev_name].__dict__[src])

        return src_type(retval)

    def copy_param(self, model, src, dest=None, fkey=None, astype=None):
        """get a copy of the system.model.src as self.dest"""
        # use default destination
        if not dest:
            dest = src

        if astype == None:
            pass
        elif astype not in (list, matrix):
            astype = matrix

        self.__dict__[dest] = self.read_param(model, src, fkey)

        # do conversion if needed
        if astype:
            self.__dict__[dest] = astype(self.__dict__[dest])
        return self.__dict__[dest]

    def _slice(self, param, idx=None):
        """slice list or matrix with idx and return (type, sliced)"""
        ty = type(self.__dict__[param])
        if ty not in [list, matrix]:
            self.message('Unsupported type <{0}> to slice.'.format(ty))
            return None

        if not idx:
            idx = list(range(self.n))
        if type(idx) != list:
            idx = list(idx)

        if ty == list:
            return [self.__dict__[param][self.int[i]] for i in idx]
        elif ty == matrix:
            return matrix([self.__dict__[param][self.int[i]] for i in idx])
        else:
            raise NotImplementedError

    def add(self, idx=None, name=None, **kwargs):
        """add an element of this model"""
        idx = self.system.DevMan.register_element(dev_name=self._name, idx=idx)
        self.int[idx] = self.n
        self.idx.append(idx)
        self.n += 1

        if name is None:
            self.name.append(self._name + ' ' + str(self.n))
        else:
            self.name.append(name)

        # check mandatory parameters
        for key in self._mandatory:
            if key not in kwargs.keys():
                self.message('Mandatory parameter <{:s}.{:s}> missing'.format(self.name[-1], key), ERROR)
                sys.exit(1)

        # set default values
        for key, value in self._data.items():
            self.__dict__[key].append(value)

        # overwrite custom values
        for key, value in kwargs.items():
            if key not in self._data:
                self.message('Parameter <{:s}.{:s}> is not used.'.format(self.name[-1], key), WARNING)
                continue
            self.__dict__[key][-1] = value

            # check data consistency
            if not value and key in self._zeros:
                if key == 'Sn':
                    default = self.system.Settings.mva
                elif key == 'fn':
                    default = self.system.Settings.freq
                else:
                    default = self._data[key]
                self.__dict__[key][-1] = default
                self.message('Using default value for <{:s}.{:s}>'.format(self.name[-1], key), WARNING)

        return idx

    def remove(self, idx=None):
        if idx is not None:
            if idx in self.int:
                key = idx
                item = self.int[idx]
            else:
                self.message('The item <{:s}> does not exist.'.format(idx), ERROR)
                return None
        else:
            return None

        convert = False
        if isinstance(self.__dict__[self._params[0]], matrix):
            self._param2list()
            convert = True

        self.n -= 1
        self.int.pop(key, '')
        self.idx.pop(item)

        for x, y in self.int.items():
            if y > item:
                self.int[x] = y - 1

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
            self._param2matrix()

    def base(self):
        """Per-unitize parameters. Store a copy."""
        if (not self.n) or self.ispu:
            return
        if 'bus' in self._ac.keys():
            bus_idx = self.__dict__[self._ac['bus'][0]]
        elif 'bus1' in self._ac.keys():
            bus_idx = self.__dict__[self._ac['bus1'][0]]
        else:
            bus_idx = []
        Sb = self.system.Settings.mva
        Vb = self.system.Bus.Vn[bus_idx]
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
            Zn = div(self.Vn ** 2, self.Sn)
            Zb = (Vb ** 2) / Sb
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
        if len(self._dcvoltages) or len(self._dccurrents) or len(self._r) or len(self._g):
            Vdc = self.system.Node.Vdcn
            if Vdc is None:
                Vdc = matrix(self.Vdcn)
            else:
                Vbdc = matrix(0.0, (self.n, 1), 'd')
                temp = sorted(self._dc.keys())
                for item in range(self.n):
                    idx = self.__dict__[temp[0]][item]
                    Vbdc[item] = Vdc[self.system.Node.int[idx]]
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

        self.ispu = True

    def setup(self):
        """
        Set up device parameters and variable addresses
        Called AFTER parsing the input file
        """
        self._interface()
        self._param2matrix()
        self._alloc()
        self.check_Vn()

    def _interface(self):
        """implement bus, node and controller interfaces"""
        self._ac_interface()
        self._dc_interface()
        self._ctrl_interface()

    def _ac_interface(self):
        """retrieve ac bus a and v addresses"""
        for key, val in self._ac.items():
            self.copy_param(model='Bus', src='a', dest=val[0], fkey=self.__dict__[key])
            self.copy_param(model='Bus', src='v', dest=val[1], fkey=self.__dict__[key])

    def _dc_interface(self):
        """retrieve v addresses of dc buses"""
        for key, val in self._dc.items():
            if type(val) == list:
                for item in val:
                    self.copy_param(model='Node', src='v', dest=item, fkey=self.__dict__[key])
            else:
                self.copy_param(model='Node', src='v', dest=val, fkey=self.__dict__[key])

    def _ctrl_interface(self):
        """Retrieve parameters of controlled model
        as: {model, param, fkey}
        """
        for key, val in self._ctrl.items():
            args = {'dest': key,
                    'fkey': self.__dict__[val[2]],
                    }
            self.copy_param(val[0], val[1], **args)

    def _addr(self):
        """
        Assign address for xvars and yvars
        Function calls aggregated in class PowerSystem and called by main()
        """
        if self.addr is True:
            self.message('Address already assigned for <{}>'.format(self._name), WARNING)
            return
        for var in range(self.n):
            for item in self._states:
                self.__dict__[item][var] = self.system.DAE.n
                self.system.DAE.n += 1
            for item in self._algebs:
                m = self.system.DAE.m
                self.__dict__[item][var] = m
                self.system.DAE.m += 1
        self.addr = True

    def _varname(self):
        """ Set up xvars and yvars names in Varname"""
        if not self.addr:
            self.message('Unable to assign Varname before allocating address', ERROR)
            return
        if not self.n:
            return
        for idx, item in enumerate(self._states):
            self.system.VarName.append(listname='unamex', xy_idx=self.__dict__[item][:],
                                       var_name=self._unamex[idx], element_name=self.name)
        for idx, item in enumerate(self._algebs):
            self.system.VarName.append(listname='unamey', xy_idx=self.__dict__[item][:],
                                       var_name=self._unamey[idx], element_name=self.name)
        try:
            for idx, item in enumerate(self._states):
                self.system.VarName.append(listname='fnamex', xy_idx=self.__dict__[item][:],
                                           var_name=self._fnamex[idx], element_name=self.name)
            for idx, item in enumerate(self._algebs):
                self.system.VarName.append(listname='fnamey', xy_idx=self.__dict__[item][:],
                                           var_name=self._fnamey[idx], element_name=self.name)
        except IndexError:
            self.message('Formatted names missing in class <{0}> definition.'.format(self._name))

    def _param2matrix(self):
        """convert _params from list to matrix"""
        for item in self._params:
            try:
                self.__dict__[item] = matrix(self.__dict__[item], tc='d')
            except:
                pass

    def _param2list(self):
        """convert _param from matrix to list"""
        for item in self._params:
            self.__dict__[item] = list(self.__dict__[item])

    def message(self, msg, level=INFO):
        """keep a line of message"""
        if level not in (DEBUG, INFO, WARNING, ERROR, CRITICAL):
            self.system.Log.error('Message logging level does not exist.')
            return
        self.system.Log.message(msg, level)

    def init_limit(self, key, lower=None, upper=None, limit=False):
        """ check if data is within limits. reset if violates"""
        above = agtb(self.__dict__[key], upper)
        idx = findeq(above, 1.0)
        for item in idx:
            maxval = upper[item]
            self.message('{0} <{1}.{2}> above its maximum of {3}.'.format(self.name[item], self._name, key, maxval), ERROR)
            if limit:
                self.__dict__[key][item] = maxval

        below = altb(self.__dict__[key], lower)
        idx = findeq(below, 1.0)
        for item in idx:
            minval = lower[item]
            self.message('{0} <{1}.{2}> below its minimum of {3}.'.format(self.name[item], self._name, key, minval), ERROR)
            if limit:
                self.__dict__[key][item] = minval

    def add_jac(self, m, val, row, col):
        """Add spmatrix(m, val, row) to DAE.(m)"""
        if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
            raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))

        size = self.system.DAE.__dict__[m].size
        self.system.DAE.__dict__[m] += spmatrix(val, row, col, size, 'd')

    def set_jac(self, m, val, row, col):
        """Set spmatrix(m, val, row) on DAE.(m)"""
        if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
            raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))

        size = self.system.DAE.__dict__[m].size
        oldval = []
        if type(row) is int:
            row = [row]
        if type(col) is int:
            col = [col]
        if type(row) is range:
            row = list(row)
        if type(col) is range:
            col = list(col)
        for i, j in zip(row, col):
            oldval.append(self.system.DAE.__dict__[m][i, j])
        self.system.DAE.__dict__[m] -= spmatrix(oldval, row, col, size, 'd')
        self.system.DAE.__dict__[m] += spmatrix(val, row, col, size, 'd')

    def reset_offline(self):
        """Reset mismatch and differential for disabled elements"""
        for idx in range(self.n):
            if self.u[idx] == 0:
                for item in self._states:
                    self.system.DAE.f[self.__dict__[item][idx]] = 0
                    self.system.DAE.xu[self.__dict__[item][idx]] = 0
                for item in self._algebs:
                    self.system.DAE.g[self.__dict__[item][idx]] = 0
                    self.system.DAE.yu[self.__dict__[item][idx]] = 0

    def insight(self, idx=None):
        """Print the parameter values as a list"""
        if not self.n:
            print('Model <{:s}> has no element'.format(self._name))
        if not idx:
            idx = sorted(self.int.keys())
        count = 2
        header_fmt = '{:^8s}{:^10s}{:^3s}'
        header = ['idx', 'name', 'u']
        if 'Sn' in self._data:
            count += 1
            header_fmt += '{:^6}'
            header.append('Sn')
        if 'Vn' in self._data:
            count += 1
            header_fmt += '{:^6}'
            header.append('Vn')

        keys = list(self._data.keys())
        for item in header:
            if item in keys:
                keys.remove(item)
        keys = sorted(keys)

        header_fmt += '|' + '{:^10s}' * len(keys)
        header += keys

        svckeys = sorted(self._service)
        keys += svckeys
        header_fmt += '|' + '{:^10s}' * len(svckeys)
        header += svckeys

        print(' ')
        print('Model <{:s}> parameter view: per-unit values'.format(self._name))
        print(header_fmt.format(*header))

        header.remove('idx')
        for i in idx:
            data = list()
            data.append(str(i))
            for item in header:
                try:
                    value = self.__dict__[item][self.int[i]]
                except:
                    value = None
                if value is not None:
                    if type(value) in [int, float]:
                        value = round(value, 6)
                        value = '{:g}'.format(value)
                else:
                    value = '-'
                data.append(value)
            print(header_fmt.format(*data))

    def var_insight(self):
        """Print variable values for debugging"""
        if not self.n:
            return
        m = len(self._algebs)
        n = len(self._states)
        dae = self.system.DAE
        out = []
        header = '{:^10s}{:^4s}' + '{:^10s}' * (n + m)
        tpl = '{:^10s}{:^4d}' + '{:^10g}' * (n + m)
        out.append(header.format(*(['idx', 'u'] + self._states + self._algebs)))
        for i in range(self.n):
            vals = [self.idx[i]]
            vals += [self.u[i]]
            vals += [dae.x[self.__dict__[var][i]] for var in self._states]
            vals += [dae.y[self.__dict__[var][i]] for var in self._algebs]
            out.append(tpl.format(*vals))
        for line in out:
            print(line)

    def __str__(self):
        self.insight()
        self.var_insight()

    def get_by_idx(self, field, idx):
        """Get values of a field by idx"""
        ret = []
        int_idx = idx
        if type(idx) == int:
            int_idx = self.int[idx]
        elif type(idx) == list or matrix:
            int_idx = [self.int[item] for item in idx]
        else:
            raise TypeError
        if not field in self.__dict__:
            raise KeyError('<{}> is not a field of model <{}>'.format(field, self._name))

        if type(self.__dict__[field]) == matrix:
            ret = self.__dict__[field][int_idx]
        elif type(self.__dict__[field]) == list:
            if type(int_idx) == int:
                ret = self.__dict__[field][int_idx]
            else:
                ret = [self.__dict__[field][item] for item in int_idx]

        return ret

    def help_doc(self, export='plain', save=None, writemode='a'):
        """Build help document into a Texttable table"""
        title = '<{}.{}>'.format(self._group, self._name)
        table = Tab(export=export, title=title, descr=self.__doc__)
        rows = []
        keys = sorted(self._data.keys())
        for key in keys:
            val = self._data[key]
            suf = ''
            if key in self._mandatory:
                suf = ' *'
            elif key in self._powers + self._voltages + self._currents + self._z + self._y +\
                        self._dccurrents + self._dcvoltages + self._r + self._g + self._times:
                suf = ' #'

            c1 = key + suf
            c2 = self._descr.get(key, '')
            c3 = val
            c4 = self._units.get(key, '-')
            rows.append([c1, c2, c3, c4])
        table.add_rows(rows, header=False)
        table.header(['Parameter', 'Description', 'Default', 'Unit'])

        if export == 'plain':
            ext = '.txt'
        elif export == 'latex':
            ext = '.tex'
        else:
            ext = '.txt'
        outfile = 'help_model' + ext

        if not save:
            print(table.draw())
            return True

        try:
            fid = open(outfile, writemode)
            fid.write(table.draw())
            fid.close()
        except IOError:
            raise IOError('Error writing model help file.')

    def check_limit(self, varname, vmin=None, vmax=None):
        """Check if the variable values are within the limits. Return False if fails."""
        assert type(varname) == str
        mat = None
        val = None
        retval = True
        if varname not in self.__dict__.keys():
            self.system.Log.error('Model <{}> does not have attribute <{}>'.format(self._name, varname))
            return
        elif varname in self._algebs:
            val = self.system.DAE.y[self.__dict__[varname]]
        elif varname in self._states:
            val = self.system.DAE.x[self.__dict__[varname]]
        else:  # service or temporary variable
            val = matrix(self.__dict__[varname])

        if min:
            vmin = matrix(self.__dict__[vmin])
            comp = altb(val, vmin)
            comp = mul(self.u, comp)
            for c, n, idx in zip(comp, self.name, range(self.n)):
                if c == 1:
                    v = val[idx]
                    vm = vmin[idx]
                    self.system.Log.error('Initialization of <{}.{}> = {:6.4g} is lower that minimum {:6.4g}'.format(n, varname, v, vm))
                    retval = False
        if max:
            vmax = matrix(self.__dict__[vmax])
            comp = agtb(val, vmax)
            comp = mul(self.u, comp)
            for c, n, idx in zip(comp, self.name, range(self.n)):
                if c == 1:
                    v = val[idx]
                    vm = vmax[idx]
                    self.system.Log.error('Initialization of <{}.{}> = {:.4g} is higher that maximum {:.4g}'.format(n, varname, v, vm))
                    retval = False
        return retval

    def on_bus(self, bus_id):
        if not hasattr(self, 'bus'):
            return
        for idx, bus in enumerate(self.bus):
            if bus == bus_id:
                return self.idx[idx]

    def check_Vn(self):
        """Check data consistency of Vn and Vdcn if connected to bus or node"""
        if not self.n:
            return
        if hasattr(self, 'bus') and hasattr(self, 'Vn'):
            bus_Vn = self.read_param('Bus', src='Vn', fkey=self.bus)
            for name, bus, Vn, Vn0 in zip(self.name, self.bus, self.Vn, bus_Vn):
                if Vn != Vn0:
                    self.message('Device <{}> has Vn={} different from bus <{}> Vn={}.'.format(name, Vn, bus, Vn0), WARNING)
        if hasattr(self, 'node') and hasattr(self, 'Vdcn'):
            node_Vdcn = self.read_param('Node', src='Vdcn', fkey=self.node)
            for name, node, Vdcn, Vdcn0 in zip(self.name, self.node, self.Vdcn, node_Vdcn):
                if Vdcn != Vdcn0:
                    self.message('Device <{}> has Vdcn={} different from node <{}> Vdcn={}.'.format(name, Vdcn, node, Vdcn0), WARNING)
    #
    # def use_model(self, model, dest=None, fkey=None):
    #     """Create reference of `self.system.__dict__[model]` to self.__dict__[model]"""
    #     if not dest:
    #         dest = model
    #     if model in self.system.DevMan.devices:
    #         # `model` is a device
    #         self.__dict__[dest] = self.system.__dict__[model]
    #     elif model in self.system.DevMan.group.keys():
    #         # `model` is a group
    #