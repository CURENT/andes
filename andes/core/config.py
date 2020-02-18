import pprint
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)


class Config(object):
    """
    A class for storing system, model and routine configurations.
    """

    def __init__(self, name, dct=None, **kwargs):
        """Constructor with a dictionary or keyword arguments"""
        self._name = name
        self._dict = OrderedDict()
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
        if dct is not None:
            self._add(**dct)

        self._add(**kwargs)

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
