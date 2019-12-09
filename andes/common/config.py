import pprint
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


class Config(object):
    """
    Class for storing model configurations that will be used in equations

    All config entries must be numerical.
    """

    def __init__(self, dct=None, **kwargs):
        """Constructor with a dictionary or keyword arguments"""
        self.add(dct, **kwargs)

    def add(self, dct=None, **kwargs):
        if dct is not None:
            self._add(**dct)

        self._add(**kwargs)

    def _add(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.__dict__:
                # skip existing entries that are already loaded (from config files)
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
            if not key.endswith('_alt'):
                out.append((key, val))

        return OrderedDict(out)

    def __repr__(self):
        return pprint.pformat(self.__dict__)
