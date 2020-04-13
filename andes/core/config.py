import pprint
import logging
from collections import OrderedDict
from andes.utils.tab import make_doc_table, math_wrap
logger = logging.getLogger(__name__)


class Config(object):
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
        if dct is not None:
            self._add(**dct)

        self._add(**kwargs)

    def add_extra(self, dest, dct=None, **kwargs):
        if dct is not None:
            kwargs.update(dct)
        for key, value in kwargs.items():
            if key not in self.__dict__:
                logger.warning(f"Config field name {key} for {dest} is invalid.")
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

    def doc(self, max_width=80, export='plain'):
        if len(self.as_dict()) == 0:
            return ''

        names, value, info = list(), list(), list()
        alt, tex = list(), list()

        for key in self._dict:
            names.append(key)
            value.append(self._dict[key])
            info.append(self._help[key] if key in self._help else '')
            alt.append(self._alt[key] if key in self._alt else '')
            tex.append(self._tex[key] if key in self._tex else key)

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

        return make_doc_table(title=f'Config Fields in [{self._name}]',
                              max_width=max_width, export=export,
                              plain_dict=plain_dict, rest_dict=rest_dict)

    @property
    def tex_names(self):
        return self._tex
