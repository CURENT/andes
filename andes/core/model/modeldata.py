"""
Module for ModelData.
"""

import logging
from collections import OrderedDict
from typing import Iterable, Sized

import numpy as np
from andes.core.model.modelcache import ModelCache
from andes.core.param import (BaseParam, DataParam, IdxParam, NumParam,
                              TimerParam)
from andes.shared import pd

logger = logging.getLogger(__name__)


class ModelData:
    r"""
    Class for holding parameter data for a model.

    This class is designed to hold the parameter data separately from model equations.
    Models should inherit this class to define the parameters from input files.

    Inherit this class to create the specific class for holding input parameters for a new model.
    The recommended name for the derived class is the model name with ``Data``. For example, data for `GENROU`
    should be named `GENROUData`.

    Parameters should be defined in the ``__init__`` function of the derived class.

    Refer to :py:mod:`andes.core.param` for available parameter types.

    Attributes
    ----------
    cache
        A cache instance for different views of the internal data.

    flags : dict
        Flags to control the routine and functions that get called. If the model is using user-defined
        numerical calls, set `f_num`, `g_num` and `j_num` properly.

    Notes
    -----
    Three default parameters are pre-defined in ``ModelData``
    and will be inherited by all models. They are

    - ``idx``, unique device idx of type :py:class:`andes.core.param.DataParam`
    - ``u``, connection status of type :py:class:`andes.core.param.NumParam`
    - ``name``, (device name of type :py:class:`andes.core.param.DataParam`

    In rare cases one does not want to define these three parameters,
    one can pass `three_params=True` to the constructor of ``ModelData``.

    Examples
    --------
    If we want to build a class ``PQData`` (for static PQ load) with three parameters, `Vn`, `p0`
    and `q0`, we can use the following ::

        from andes.core.model import ModelData, Model
        from andes.core.param import IdxParam, NumParam

        class PQData(ModelData):
            super().__init__()
            self.Vn = NumParam(default=110,
                               info="AC voltage rating",
                               unit='kV', non_zero=True,
                               tex_name=r'V_n')
            self.p0 = NumParam(default=0,
                               info='active power load in system base',
                               tex_name=r'p_0', unit='p.u.')
            self.q0 = NumParam(default=0,
                               info='reactive power load in system base',
                               tex_name=r'q_0', unit='p.u.')

    In this example, all the three parameters are defined as
    :py:class:`andes.core.param.NumParam`.
    In the full `PQData` class, other types of parameters also exist.
    For example, to store the idx of `owner`, `PQData` uses ::

        self.owner = IdxParam(model='Owner', info="owner idx")

    """

    def __init__(self, *args, three_params=True, **kwargs):
        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.n = 0
        self.uid = {}

        # indexing bases. Most vectorized models only have one base: self.idx
        self.index_bases = []

        if not hasattr(self, 'cache'):
            self.cache = ModelCache()
        self.cache.add_callback('dict', self.as_dict)
        self.cache.add_callback('df', lambda: self.as_df())
        self.cache.add_callback('dict_in', lambda: self.as_dict(True))
        self.cache.add_callback('df_in', lambda: self.as_df(vin=True))

        if three_params is True:
            self.idx = DataParam(info='unique device idx')
            self.u = NumParam(default=1, info='connection status', unit='bool', tex_name='u')
            self.name = DataParam(info='device name')

            self.index_bases.append(self.idx)

    def __len__(self):
        return self.n

    def __setattr__(self, key, value):
        if isinstance(value, BaseParam):
            value.owner = self
            if not value.name:
                value.name = key

            if key in self.__dict__:
                logger.warning("%s: redefining <%s>. This is likely a modeling error.",
                               self.class_name, key)

            self.params[key] = value

        if isinstance(value, NumParam):
            self.num_params[key] = value
        elif isinstance(value, IdxParam):
            self.idx_params[key] = value

        # `TimerParam` is a subclass of `NumParam` and thus tested separately
        if isinstance(value, TimerParam):
            self.timer_params[key] = value

        super(ModelData, self).__setattr__(key, value)

    def add(self, **kwargs):
        """
        Add a device (an instance) to this model.

        Warnings
        --------
        This function is not intended to be used directly.
        Use the ``add`` method from System so that the index
        can be registered correctly.

        Parameters
        ----------
        kwargs
            model parameters are collected into the kwargs dictionary
        """
        idx = kwargs['idx']
        self.uid[idx] = self.n
        self.n += 1
        if "name" in self.params:
            name = kwargs.get("name")
            if (name is None) or (not isinstance(name, str) and np.isnan(name)):
                kwargs["name"] = idx

        if "idx" not in self.params:
            kwargs.pop("idx")

        for name, instance in self.params.items():
            value = kwargs.pop(name, None)
            instance.add(value)
        if len(kwargs) > 0:
            logger.warning("%s: unused data %s", self.class_name, str(kwargs))

    def as_dict(self, vin=False):
        """
        Export all parameters as a dict.

        Returns
        -------
        dict
            a dict with the keys being the `ModelData` parameter names
            and the values being an array-like of data in the order of adding.
            An additional `uid` key is added with the value default to range(n).
        """
        out = dict()
        out['uid'] = np.arange(self.n)

        for name, instance in self.params.items():
            # skip non-exported parameters
            if instance.export is False:
                continue

            out[name] = instance.v

            # use the original input if `vin` is True
            if (vin is True) and hasattr(instance, 'vin') and (instance.vin is not None):
                out[name] = instance.vin

            conv = instance.oconvert
            if conv is not None:
                out[name] = np.array([conv(item) for item in out[name]])

        return out

    def as_df(self, vin=False):
        """
        Export all parameters as a `pandas.DataFrame` object.
        This function utilizes `as_dict` for preparing data.

        Returns
        -------
        DataFrame
            A dataframe containing all model data. An `uid` column is added.
        vin : bool
            If True, export all parameters from original input (``vin``).
        """
        if vin is False:
            out = pd.DataFrame(self.as_dict()).set_index('uid')
        else:
            out = pd.DataFrame(self.as_dict(vin=True)).set_index('uid')

        return out

    def as_df_local(self):
        """
        Export local variable values and services to a DataFrame.
        """

        out = dict()
        out['uid'] = np.arange(self.n)
        out['idx'] = self.idx.v

        for name, instance in self.cache.all_vars.items():
            out[name] = instance.v

        for name, instance in self.services.items():
            out[name] = instance.v

        return pd.DataFrame(out).set_index('uid')

    def update_from_df(self, df, vin=False):
        """
        Update parameter values from a DataFrame.

        Adding devices are not allowed.
        """
        if vin is False:
            for name, instance in self.params.items():
                if instance.export is False:
                    continue
                instance.set_all('v', df[name])
        else:
            for name, instance in self.params.items():
                if instance.export is False:
                    continue
                try:
                    instance.set_all('vin', df[name])
                    instance.v[:] = instance.vin * instance.pu_coeff
                except KeyError:
                    # fall back to `v`
                    instance.set_all('v', df[name])

        return True

    def find_param(self, prop):
        """
        Find params with the given property and return in an OrderedDict.

        Parameters
        ----------
        prop : str
            Property name

        Returns
        -------
        OrderedDict
        """
        out = OrderedDict()
        for name, instance in self.params.items():
            if instance.get_property(prop) is True:
                out[name] = instance

        return out

    def find_idx(self, keys, values, allow_none=False, default=False):
        """
        Find `idx` of devices whose values match the given pattern.

        Parameters
        ----------
        keys : str, array-like, Sized
            A string or an array-like of strings containing the names of parameters for the search criteria
        values : array, array of arrays, Sized
            Values for the corresponding key to search for. If keys is a str, values should be an array of
            elements. If keys is a list, values should be an array of arrays, each corresponds to the key.
        allow_none : bool, Sized
            Allow key, value to be not found. Used by groups.
        default : bool
            Default idx to return if not found (missing)

        Returns
        -------
        list
            indices of devices
        """
        if isinstance(keys, str):
            keys = (keys,)
            if not isinstance(values, (int, float, str, np.floating)) and not isinstance(values, Iterable):
                raise ValueError(f"value must be a string, scalar or an iterable, got {values}")

            if len(values) > 0 and not isinstance(values[0], (list, tuple, np.ndarray)):
                values = (values,)

        elif isinstance(keys, Sized):
            if not isinstance(values, Iterable):
                raise ValueError(f"value must be an iterable, got {values}")

            if len(values) > 0 and not isinstance(values[0], Iterable):
                raise ValueError(f"if keys is an iterable, values must be an iterable of iterables. got {values}")

            if len(keys) != len(values):
                raise ValueError("keys and values must have the same length")

        v_attrs = [self.__dict__[key].v for key in keys]

        idxes = []
        for v_search in zip(*values):
            v_idx = None
            for pos, v_attr in enumerate(zip(*v_attrs)):
                if all([i == j for i, j in zip(v_search, v_attr)]):
                    v_idx = self.idx.v[pos]
                    break
            if v_idx is None:
                if allow_none is False:
                    raise IndexError(f'{list(keys)}={v_search} not found in {self.class_name}')
                else:
                    v_idx = default

            idxes.append(v_idx)

        return idxes
