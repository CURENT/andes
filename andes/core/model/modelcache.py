"""
Module for ModelCache.
"""

import warnings
from collections import OrderedDict


def cache_field(func):
    """
    Decorator that marks a Model method for automatic cache registration.

    The cache key is derived from the method name by stripping a single
    leading underscore (``_iter_vars`` → ``iter_vars``).

    Decorated methods are discovered by :meth:`ModelCache.register_fields`
    during ``Model.__init__``.
    """
    name = func.__name__
    func._cache_key = name.lstrip('_') if name.startswith('_') else name
    return func


class ModelCache:
    """
    Class for caching the return value of callback functions.

    Check ``ModelCache.__dict__.keys()`` for fields.
    """

    _DEPRECATED_FIELDS = {
        'dict': 'as_dict()',
        'df': 'as_df()',
        'dict_in': 'as_dict(vin=True)',
        'df_in': 'as_df(vin=True)',
    }

    def __init__(self):
        self._callbacks = {}
        self._owner = None

    def __getattr__(self, item):
        if item == "_callbacks":
            return self.__getattribute__(item)

        if item in self._DEPRECATED_FIELDS:
            replacement = self._DEPRECATED_FIELDS[item]
            warnings.warn(
                f"'cache.{item}' is deprecated and will be removed in v3.0.0. "
                f"Use 'model.{replacement}' instead.",
                FutureWarning,
                stacklevel=2,
            )
            if self._owner is not None:
                vin = item.endswith('_in')
                if item.startswith('dict'):
                    return self._owner.as_dict(vin=vin)
                else:
                    return self._owner.as_df(vin=vin)

        if item in self._callbacks:
            val = self._call(item)
            self.__dict__[item] = val
            return val

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

    def add_combined(self, name, model, attrs):
        """
        Register a cache field that concatenates OrderedDicts from *model*.

        Parameters
        ----------
        name : str
            Cache key (e.g. ``'all_vars'``).
        model : Model
            The model instance whose attributes will be read.
        attrs : tuple of str
            Attribute names on *model*, each an ``OrderedDict``.
            They are concatenated in order.
        """
        def _combine():
            items = []
            for attr in attrs:
                items.extend(getattr(model, attr, OrderedDict()).items())
            return OrderedDict(items)
        self._callbacks[name] = _combine

    def register_fields(self, instance):
        """
        Discover ``@cache_field``-decorated methods on *instance* and
        register each as a cache callback.

        Walks the class MRO so that inherited decorated methods are
        also picked up.  First definition wins (child overrides parent).
        """
        seen = set()
        for cls in type(instance).__mro__:
            for attr_name, method in vars(cls).items():
                if attr_name in seen:
                    continue
                key = getattr(method, '_cache_key', None)
                if key is not None:
                    # bind *method* and *instance* via default args to
                    # avoid late-binding closure issues
                    self._callbacks[key] = (
                        lambda m=instance, f=method: f(m)
                    )
                    seen.add(attr_name)

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
