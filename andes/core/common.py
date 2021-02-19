import pprint
import logging

from sympy import Expr

from typing import Iterable
from collections import defaultdict, OrderedDict

from andes.shared import jac_full_names, jac_names, jac_types
from andes.utils.tab import math_wrap, make_doc_table

logger = logging.getLogger(__name__)


class ModelFlags:
    """
    Model flags.

    Parameters
    ----------
    collate : bool
        True: collate variables by device; False: by variable.
        Non-collate (continuous memory) has faster computation speed.
    pflow : bool
        True: called during power flow
    tds : bool
        True if called during tds; if is False, ``dae_t`` cannot be used
    pflow_init : bool or None
        True if initialize pflow; False otherwise; None default to `pflow`
    tds_init : bool or None
        True if initialize tds; False otherwise; None default to `tds`
    series : bool
        True if is series device
    nr_iter : bool
        True if is series device
    f_num : bool
        True if the model defines `f_numeric`
    g_num : bool
        True if the model defines `g_numeric`
    j_num : bool
        True if the model defines `j_numeric`
    s_num : bool
        True if the model defines `s_numeric`
    sv_num : bool
        True if the model defines `s_numeric_var`
    jited : bool
        True if numba JIT code is generated
    """

    def __init__(self, collate=False, pflow=False, tds=False,
                 pflow_init=None, tds_init=None, series=False,
                 nr_iter=False, f_num=False, g_num=False, j_num=False,
                 s_num=False, sv_num=False):

        self.collate = collate
        self.pflow = pflow
        self.tds = tds
        self.pflow_init = pflow_init
        self.tds_init = tds_init
        self.series = series
        self.nr_iter = nr_iter
        self.f_num = f_num
        self.g_num = g_num
        self.j_num = j_num
        self.s_num = s_num
        self.sv_num = sv_num
        self.sys_base = False
        self.address = False
        self.initialized = False
        self.jited = False

    def update(self, dct):
        self.__dict__.update(dct)

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class DummyValue:
    """
    Class for converting a scalar value to a dummy parameter with `name` and `tex_name` fields.

    A DummyValue object can be passed to Block, which utilizes the `name` field to dynamically generate equations.

    Notes
    -----
    Pass a numerical value to the constructor for most use cases, especially when passing as a v-provider.
    """
    def __init__(self, value):
        if isinstance(value, str):
            self.name = f'({value})'
        else:
            self.name = value
        self.tex_name = value
        self.v = value


def dummify(param):
    """
    Dummify scalar parameter and return a DummyValue object. Do nothing for BaseParam instances.

    Parameters
    ----------
    param : float, int, str, BaseParam
        parameter object or scalar value

    Returns
    -------
    DummyValue(param) if param is a scalar; param itself, otherwise.

    """
    if isinstance(param, (int, float, str)):
        return DummyValue(param)
    else:
        return param


class JacTriplet:
    """
    Storage class for Jacobian triplet lists.
    """
    def __init__(self):
        self.ijac = defaultdict(list)
        self.jjac = defaultdict(list)
        self.vjac = defaultdict(list)

    def clear_ijv(self):
        """
        Clear stored triplets for all sparse Jacobian matrices
        """
        for j_full_name in jac_full_names:
            self.ijac[j_full_name] = list()
            self.jjac[j_full_name] = list()
            self.vjac[j_full_name] = list()

    def append_ijv(self, j_full_name, ii, jj, vv):
        """
        Append triplets to the given sparse matrix triplets.

        Parameters
        ----------
        j_full_name : str
            Full name of the sparse Jacobian. If is a constant Jacobian, append 'c' to the Jacobian name.
        ii : array-like
            Row indices
        jj : array-like
            Column indices
        vv : array-like
            Value indices
        """
        if len(ii) == 0 and len(jj) == 0:
            return
        self.ijac[j_full_name].append(ii)
        self.jjac[j_full_name].append(jj)
        self.vjac[j_full_name].append(vv)

    def ijv(self, j_full_name):
        """
        Return triplet lists in a tuple in the order or (ii, jj, vv)
        """
        return self.ijac[j_full_name], self.jjac[j_full_name], self.vjac[j_full_name]

    def zip_ijv(self, j_full_name):
        """
        Return a zip iterator in the order of (ii, jj, vv)
        """
        return zip(*self.ijv(j_full_name))

    def merge(self, triplet):
        """
        Merge another triplet into this one.
        """
        for jname in jac_names:
            for jtype in jac_types:
                self.ijac[jname + jtype] += triplet.ijac[jname + jtype]
                self.jjac[jname + jtype] += triplet.jjac[jname + jtype]
                self.vjac[jname + jtype] += triplet.vjac[jname + jtype]


class Config:
    """
    A class for storing system, model and routine configurations.
    """

    def __init__(self, name, dct=None, **kwargs):
        """
        Constructor with a dictionary or keyword arguments
        """
        self._name = name
        self._dict = OrderedDict()
        self._help = OrderedDict()
        self._tex = OrderedDict()
        self._alt = OrderedDict()
        self.add(dct, **kwargs)

    def load(self, config):
        """
        Load from a ConfigParser object, ``config``.
        """
        if config is None:
            return
        if self._name in config:
            config_section = config[self._name]
            self.add(OrderedDict(config_section))

    def add(self, dct=None, **kwargs):
        """
        Add config fields from a dictionary or keyword args.

        Existing configs will NOT be overwritten.
        """
        def warn_upper(s):
            if any(x.isupper() for x in s):
                logger.warning("Config fields must be in lower case, found %s", s)

        if dct is not None:
            for s in dct.keys():
                warn_upper(s)

            self._add(**dct)

        for s in kwargs.keys():
            warn_upper(s)

        self._add(**kwargs)

    def add_extra(self, dest, dct=None, **kwargs):
        """
        Add extra contents for config.

        Parameters
        ----------
        dest : str
            Destination string in `_alt`, `_help` or `_tex`.
        dct : OrderedDict, dict
            key: value pairs
        """

        if dct is not None:
            kwargs.update(dct)
        for key, value in kwargs.items():
            if key not in self.__dict__:
                logger.warning("Config field name %s for %s is invalid.", key, dest)
                continue
            self.__dict__[dest][key] = value

    def _add(self, **kwargs):
        for key, val in kwargs.items():
            # skip existing entries that are already loaded (from config files)
            if key in self.__dict__:
                continue

            if isinstance(val, str):
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

            self.__dict__[key] = val

    def as_dict(self, refresh=False):
        """
        Return the config fields and values in an ``OrderedDict``.

        Values are cached in `self._dict` unless refreshed.
        """
        if refresh is True or len(self._dict) == 0:
            out = []
            for key, val in self.__dict__.items():
                if not key.startswith('_'):
                    out.append((key, val))
            self._dict = OrderedDict(out)

        return self._dict

    def __repr__(self):
        return pprint.pformat(self.as_dict())

    def doc(self, max_width=78, export='plain', target=False, symbol=True):
        out = ''
        if len(self.as_dict()) == 0:
            return out

        if export == 'rest' and target is True:
            max_width = 0
            model_header = '-' * 80 + '\n'
            out += f'.. _{self._name}:\n\n'
            out += model_header + f'{self._name}\n' + model_header
        else:
            model_header = '\n'
            out += model_header + f'Config Fields in [{self._name}]\n' + model_header

        names, value, info = list(), list(), list()
        alt, tex = list(), list()

        for key in self._dict:
            names.append(key)
            value.append(self._dict[key])
            info.append(self._help[key] if key in self._help else '')
            alt.append(self._alt[key] if key in self._alt else '')
            tex.append(self._tex[key] if key in self._tex else '')

        tex = math_wrap(tex, export=export)

        plain_dict = OrderedDict([('Option', names),
                                  ('Value', value),
                                  ('Info', info),
                                  ('Acceptable values', alt)])
        rest_dict = OrderedDict([('Option', names),
                                 ('Symbol', tex),
                                 ('Value', value),
                                 ('Info', info),
                                 ('Accepted values', alt)])

        if not symbol:
            rest_dict.pop("Symbol")

        out += make_doc_table(title="", max_width=max_width, export=export,
                              plain_dict=plain_dict, rest_dict=rest_dict)
        return out

    def check(self):
        """
        Check the validity of config values.
        """
        for key, val in self.as_dict().items():
            if key not in self._alt:
                continue

            _alt = self._alt[key]
            if not isinstance(_alt, Iterable):
                continue
            if isinstance(_alt, str):
                continue
            if val not in _alt:
                raise ValueError(f"[{self._name}].{key}={val} is not a choice from {_alt}.")

        return True

    @property
    def tex_names(self):
        return self._tex


class Indicator(Expr):
    """
    Indicator class for printing SymPy Relational.

    Relational expressions in SymPy need to be wrapped by `Indicator`.

    Examples
    --------
    To compare ``dae_t`` with ``0``, one need to use
    ``Indicator(dae_t < 0)```.
    """

    def _numpycode(self, printer):
        return printer._print(self.args[0])
