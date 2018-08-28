import os
import configparser


from ..utils.cached import cached
from ..utils.tab import Tab


class SettingsBase(object):
    """base setting class"""

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
        title = 'Setting <{:s}>'.format(self.__class__.__name__)
        table = Tab(export=export, title=title)

        for opt in sorted(self.descr):
            if hasattr(self, opt):
                c1 = opt
                c2 = self.descr[opt]
                c3 = self.__dict__.get(opt, '')
                c4 = self.get_alt(opt)
                rows.append([c1, c2, c3, c4])
            else:
                print('Setting {:s} has no {:s} option. Correct in descr.'.format(self.__class__.__name__, opt))

        table.add_rows(rows, header=False)
        table.header(['Option', 'Description', 'Value', 'Alt.'])

        ext = 'txt'
        if export == 'latex':
            ext = 'tex'
            raise NotImplementedError

        return table.draw()

    def dump_rc(self, path=None, mode='w+'):
        """
        Dump settings to an rc config file

        Parameters
        ----------
        path: str
            rc file path
        mode: str
            output file write mode

        Returns
        -------
        None
        """
        if not path:
            path = os.path.abspath('.')
        out = []
        tab = type(self).__name__
        dct = self.__dict__
        out.append('[{}]'.format(tab))
        keys_sorted = sorted(dct.keys())

        for key in keys_sorted:
            val = dct[key]
            out.append('{} = {}'.format(key, val))

        with open(path, mode=mode) as f:
            f.write('\n'.join(out))

    def load_config(self, rc):
        """
        Load configurations from an rc file

        Parameters
        ----------
        rc: str
            path to the rc file

        Returns
        -------
        bool
            success flag

        """
        ret = False

        if not os.path.isfile(rc):
            self.system.Log.warning('Config file {} does not exist.'.format(rc))
            return ret

        config = configparser.ConfigParser()
        config.read(rc)

        for section in config.sections():
            if section not in self.__dict__:
                self.system.warning('Skipping Config section [{}].'.format(section))
                continue
            for key in config[section].keys():
                if not hasattr(self.__dict__[section], key):
                    self.system.warning('Skipping Config [{}].<{}>'.format(section, key))
                val = config[section].get(key)
                try:
                    val = config[section].getfloat(key)
                except ValueError:
                    try:
                        val = config[section].getboolean(key)
                    except ValueError:
                        pass

                if hasattr(self.__dict__[section], key + '_alt'):
                    if val not in self.__dict__[section].__dict__[key + '_alt']:
                        self.system.warning('Invalid Value <{}> for Config [{}].<{}>'.format(val, section, key))
                self.__dict__[section].__dict__.update({key: val})

    @cached
    def descr(self):
        descriptions = {}
        return descriptions


from .settings import Settings
from .spf import SPF
from .cpf import CPF
from .tds import TDS
from .sssa import SSSA


__all__ = ['settings',
           'spf',
           'tds',
           'sssa',
           'cpf',
           ]
