from cvxopt import matrix, sparse, spmatrix
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
import copy
import sys


class base(object):
    """base class for power system device models"""

    def __init__(self, system):
        """class initialization of base device models
        define meta-data and default values"""
        self.system = system
        self.n = 0    # device count
        self.u = []   # device status
        self.idx = []    # internal index list
        self.int = {}    # external index to internal
        self.names = []  # element name list

        # identifications
        self._name = None
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
                       'Vn': 'KV',
                       }

        # variable descriptions
        self._descr = {'u': 'connection status',
                       'Sn': 'power rating',
                       'Vn': 'voltage rating',
                       }
        # non-zero parameters
        self._zeros = ['Sn', 'Vn']

        # mandatory variables
        self._mandatory = []

        # service/temporary variables
        self._service = []

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

        self.calls = dict(addr0=False, addr1=False,
                          init0=False, init1=False,
                          jac0=False, windup=False,
                          gcall=False, fcall=False,
                          gycall=False, fxcall=False,
                          series=False, shunt=False,
                          flows=False, dcseries=False,
                          )

    def _inst_meta(self):
        """instantiate meta-data defined in __init__().
        Call this function at the end of __init__() of child classes
        """
        if not self._name:
            self._name = self._group

        for item in self._data.keys():
            self.__dict__[item] = []
        for bus in self._ac.keys():
            for var in self._ac[bus]:
                self.__dict__[var] = []
        for node in self._dc.keys():
            for var in self._dc[node]:
                self.__dict__[var] = []

        for var in self._states:
            self.__dict__[var] = []
        for var in self._algebs:
            self.__dict__[var] = []
        for var in self._service:
            self.__dict__[var] = []

    def _alloc(self):
        """allocate memory for DAE variable indices
        called after finishing adding components
        """
        zeros = [0] * self.n
        for var in self._states:
            self.__dict__[var] = zeros[:]
        for var in self._algebs:
            self.__dict__[var] = zeros[:]

    def get_param(self, model, src, dest=None, fkey=None, astype=None):
        """get a copy of the system.model.src as self.dest"""
        # input check
        if not hasattr(self.system, model):
            self.message('Model {} is trying to copy {} which does not exist.'.format(self._name, model), ERROR)
            return
        if not hasattr(self.system.__dict__[model], src):
            self.message('Model {} is trying to Model {}.{} which does not exist.'.format(self._name, model, src), ERROR)
            return
        if fkey and type(fkey) not in (list, matrix):
            self.message('Order type is not list or matrix', ERROR)
            return
        if not dest:
            dest = src
        if hasattr(self, dest) and self.__dict__[dest] is not None:
            self.message('Model {} already has member {}.'.format(self._name, dest), WARNING)
        if astype and astype not in ('list', 'matrix'):
            self.message('Only converting to list or matrix is supported.', WARNING)
            astype = None

        if not fkey:  # copy full
            self.__dict__[dest] = copy.deepcopy(self.system.__dict__[model].__dict__[src])
        else:  # copy based on the order of fkey
            idx = self.system.__dict__[model].int[fkey]
            self.__dict__[dest] = copy.deepcopy(self.system.__dict__[model].__dict__[src][idx])
        if astype:
            if astype == 'list':
                self.__dict__[dest] = list(self.__dict__[dest])
            elif astype == 'matrix':
                self.__dict__[dest] = matrix(self.__dict__[dest])
            else:
                pass


    def add(self, idx=None, name=None, **kwargs):
        self.n += 1
        if idx is None:
            idx = self._group + '_' + str(self.n)
        self.int[idx] = self.n - 1
        self.idx.append(self.n)
        self.system.Groups[self._group].append(self._name)

        if name is None:
            self.names.append(self._name + '_' + str(self.n))
        else:
            self.names.append(name)

        # check mandatory parameters
        for key in self._mandatory:
            if key not in kwargs.keys():
                self.message('Mandatory parameter <{:s}.{:s}> missing'.format(self.names[-1], key), ERROR)
                sys.exit(1)

        # set default values
        for key, value in self._data.items():
            self.__dict__[key].append(value)

        # overwrite custom values
        for key, value in kwargs.items():
            if key not in self._data:
                self.message('Parameter <{:s}.{:s}> is undefined'.format(self.names[-1], key), WARNING)
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
                self.message('Using default value for <{:s}.{:s}>'.format(name, key), WARNING)

        return idx

    def remove(self, idx=None):
        if idx is not None:
            if idx in self.int:
                key = idx
                item = self.int[idx]
            else:
                self.system.Log.error('The item <{:s}> does not exist.'.format(idx))
                return None
        else:
            # nothing to remove
            return None

        convert = False
        if isinstance(self.__dict__[self._params[0]], matrix):
            self._matrix2list()
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

        self.names.pop(item)
        if convert and self.n:
            self._list2matrix()

    def setup(self):
        """
        Set up device parameters and variable addresses
        Called AFTER parsing the input file
        """
        self._interface()
        self._list2matrix()
        self._alloc()

    def _interface(self):
        """implement bus, node and controller interfaces"""
        self._ac_interface()
        self._dc_interface()
        self._ctrl_interface()

    def _ac_interface(self):
        """retrieve ac bus a and v addresses"""
        for key, val in self._ac.items():
            for item in val:
                self.get_param('Bus', src=item, dest=item, fkey=self.__dict__[key])

    def _dc_interface(self):
        """retrieve v addresses of dc buses"""
        for key, val in self._dc.items():
            for item in val:
                self.get_param('Node', src=item, dest=item, fkey=self.__dict__[key])

    def _ctrl_interface(self):
        pass

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

    def _list2matrix(self):
        for item in self._params:
            self.__dict__[item] = matrix(self.__dict__[item])

    def _matrix2list(self):
        for item in self._params:
            self.__dict__[item] = list(self.__dict__[item])

    def message(self, msg, level):
        """keep a line of message"""
        if level not in (DEBUG, INFO, WARNING, ERROR, CRITICAL):
            print('Message logging level does not exist.')
            return
            # todo: record the message to logging

    def limit_check(self, data, min=None, max=None):
        """ check if data is within limits. reset if violates"""
        pass