import configparser

from ..utils.cached import cached
from ..utils.tab import Tab
import logging

logger = logging.getLogger(__name__)


class ConfigBase(object):
    """base setting class"""
    def __init__(self, conf=None, **kwargs):

        if conf is not None:
            self.load_config(conf)

        self.check()

    def get_value(self, option):
        """
        Return the value of the given option

        Parameters
        ----------
        option: str
            option name to retrieve value

        Returns
        -------
        str
            value of `option`
        """

        return self.__dict__[option]

    def get_alt(self, option):
        """
        Return the alternative values of an option

        Parameters
        ----------
        option: str
            option name

        Returns
        -------
        str
            a string of alternative options

        """
        assert hasattr(self, option)

        alt = option + '_alt'
        if not hasattr(self, alt):
            return ''
        return ', '.join(self.__dict__[alt])

    def doc(self, export='plain'):
        """
        Dump help document for setting classes
        """
        rows = []
        title = '<{:s}> config options'.format(self.__class__.__name__)
        table = Tab(export=export, title=title)

        for opt in sorted(self.config_descr):
            if hasattr(self, opt):
                c1 = opt
                c2 = self.config_descr[opt]
                c3 = self.__dict__.get(opt, '')
                c4 = self.get_alt(opt)
                rows.append([c1, c2, c3, c4])
            else:
                print('Setting {:s} has no {:s} option. Correct in config_descr.'.
                      format(self.__class__.__name__, opt))

        table.add_rows(rows, header=False)
        table.header(['Option', 'Description', 'Value', 'Alt.'])

        return table.draw()

    def dump_conf(self, conf=None):
        """
        Dump settings to an rc config file

        Parameters
        ----------
        conf
            configparser.ConfigParser() object

        Returns
        -------
        None
        """
        if conf is None:
            conf = configparser.ConfigParser()

        tab = self.__class__.__name__

        conf[tab] = {}

        for key, val in self.__dict__.items():
            if key.endswith('_alt'):
                continue
            conf[tab][key] = str(val)

        return conf

    def load_config(self, conf):
        """
        Load configurations from an rc file

        Parameters
        ----------
        rc: str
            path to the rc file

        Returns
        -------
        None
        """
        section = self.__class__.__name__

        if section not in conf.sections():
            logger.debug('Config section {} not in rc file'.format(
                self.__class__.__name__))
            return

        for key in conf[section].keys():
            if not hasattr(self, key):
                logger.debug('Config key {}.{} skipped'.format(section, key))
                continue

            val = conf[section].get(key)

            try:
                val = conf[section].getfloat(key)
            except ValueError:
                try:
                    val = conf[section].getboolean(key)
                except ValueError:
                    pass

            self.__dict__.update({key: val})

    def check(self):
        """
        Check for consistency

        Returns
        -------
        bool
            True for pass, False for fail
        """
        # TODO: check key in _alt
        return True

    @cached
    def config_descr(self):
        descriptions = {}
        return descriptions
