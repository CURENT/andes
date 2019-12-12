import pprint
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


class Config(object):
    """
    Class for storing model configurations that will be used in equations and routines.

    All config entries must be numerical.
    """

    def __init__(self, name, dct=None, **kwargs):
        """Constructor with a dictionary or keyword arguments"""
        self._name = name
        self.add(dct, **kwargs)

    def load(self, config):
        """
        Load from ConfigParser config object

        Parameters
        ----------
        config

        Returns
        -------

        """
        if config is None:
            return
        if self._name in config:
            config_section = config[self._name]
            self.add(OrderedDict(config_section))

    def add(self, dct=None, **kwargs):
        """
        Add additional configs. Existing configs will not be overwritten.

        Parameters
        ----------
        dct
        kwargs

        Returns
        -------

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
                        logger.debug(f'Non-numeric value in config {key} = {val}')

            self.__dict__[key] = val

    def as_dict(self):
        out = []
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                out.append((key, val))

        return OrderedDict(out)

    def __repr__(self):
        return pprint.pformat(self.as_dict())
