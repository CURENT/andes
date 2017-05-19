from ..utils.cached import cached
from ..utils.tab import Tab


class SettingsBase(object):
    """base setting class"""

    def get_value(self, option):
        """return the value of the given option"""
        if not hasattr(self, option):
            return 'N/A'
        return self.__dict__[option]

    def get_alt(self, option):
        """get alternative values of an option"""
        alt = option + '_alt'
        if not hasattr(self, alt):
            return ''
        return ', '.join(self.__dict__[alt])

    def dump_help(self, export='plain', save=None, writemode='w'):
        """dump help document for setting classes"""
        rows = []
        title = 'Setting class <{:s}>'.format(self.__class__.__name__)
        table = Tab(export=export, title=title)

        for opt in sorted(self.doc_help):
            if hasattr(self, opt):
                c1 = opt
                c2 = self.doc_help[opt]
                c3 = self.__dict__.get(opt, '')
                c4 = self.get_alt(opt)
                rows.append([c1, c2, c3, c4])
            else:
                warn_msg = 'Setting object {:s} has no \'{:s}\' option. Correct in doc_help.'.format(self.__class__.__name__, opt)
                print(warn_msg)
        table.add_rows(rows, header=False)  # first row is not header
        table.header(['Option', 'Description', 'Value', 'Alt.'])

        ext = 'txt'
        if export == 'latex':
            ext = 'tex'

        results = table.draw()

        if not save:
            print(results)
        else:
            filename = 'settings_help' + '.' + ext
            try:
                f=open(filename, writemode)
                f.write(results)
                f.close()
            except IOError:
                print(results)
                print('Error saving settings help to file')

    @cached
    def doc_help(self):
        descriptions = {}
        return descriptions
