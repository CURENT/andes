"""
Deprecation warning: This module is deprecated and will be removed in the
future.

Module for ModelCache.
"""


from collections import OrderedDict
from andes.shared import pd
import numpy as np


class ModelCache:
    """
    Class for caching the return value of callback functions.

    Check ``ModelCache.__dict__.keys()`` for fields.
    """

    def __init__(self, model):
        self.model = model
        self._callbacks = {}

    def __getattr__(self, item):
        if item == "_callbacks":
            return self.__getattribute__(item)

        if item not in self.__dict__:
            if item in self._callbacks:
                self.__dict__[item] = self._call(item)

        return self.__getattribute__(item)

    def __getstate__(self):
        return self.__dict__

    def add_callback(self, name: str, callback):
        """
        Add a cache attribute and a callback function for updating the attribute.

        Parameters
        ----------
        name : str
            name of the cached function return value
        callback : callable
            callback function for updating the cached attribute
        """
        self._callbacks[name] = callback

    def refresh(self, name=None):
        """
        Refresh the cached values

        Parameters
        ----------
        name : str, list, optional
            name or list of cached to refresh, by default None for refreshing all

        """
        if name is None:
            for name in self._callbacks.keys():
                self.__dict__[name] = self._call(name)
        elif isinstance(name, str):
            self.__dict__[name] = self._call(name)
        elif isinstance(name, list):
            for n in name:
                self.__dict__[n] = self._call(n)

    def _call(self, name):
        """
        Helper function for calling callback functions.

        Parameters
        ----------
        name : str
            attribute name to be updated

        Returns
        -------
        callback result
        """
        if name not in self._callbacks:
            return None
        else:
            if callable(self._callbacks[name]):
                return self._callbacks[name]()
            else:
                return self._callbacks[name]

    def initialize(self):

        # cached class attributes
        self.add_callback('dict', self.as_dict)
        self.add_callback('df', lambda: self.as_df())
        self.add_callback('dict_in', lambda: self.as_dict(True))
        self.add_callback('df_in', lambda: self.as_df(vin=True))

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

        for name, instance in self.all_vars().items():
            out[name] = instance.v

        for name, instance in self.services.items():
            out[name] = instance.v

        return pd.DataFrame(out).set_index('uid')

    def _v_getters(self):
        out = OrderedDict()
        for name, var in self.all_vars().items():
            if var.v_inplace:
                continue
            out[name] = var
        return out

    def _v_adders(self):
        out = OrderedDict()
        for name, var in self.all_vars().items():
            if var.v_inplace is True:
                continue
            if var.v_str is None and var.v_iter is None:
                continue
            if var.v_setter is True:
                continue

            out[name] = var
        return out

    def _v_setters(self):
        out = OrderedDict()
        for name, var in self.all_vars().items():
            if var.v_inplace is True:
                continue
            if var.v_str is None and var.v_iter is None:
                continue
            if var.v_setter is False:
                continue

            out[name] = var
        return out

    def _e_adders(self):
        out = OrderedDict()
        for name, var in self.all_vars().items():
            if var.e_inplace is True:
                continue
            if var.e_str is None:
                continue
            if var.e_setter is True:
                continue

            out[name] = var
        return out

    def _e_setters(self):
        out = OrderedDict()
        for name, var in self.all_vars().items():
            if var.e_inplace is True:
                continue
            if var.e_str is None:
                continue
            if var.e_setter is False:
                continue

            out[name] = var
        return out

    def _input_vars(self):
        out = list()
        for name, var in self.all_vars().items():
            if var.is_input:
                out.append(name)
        return out

    def _output_vars(self):
        out = list()
        for name, var in self.all_vars().items():
            if var.is_output:
                out.append(name)
        return out
