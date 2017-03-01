from ..utils.cached import cached
from ..utils.tab import tab


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
        return self.__dict__[alt]

    def dump_help(self, export='plain', save=False):
        """dump help document for setting classes"""
        rows = []
        title = 'Setting class <{:s}>'.format(self.__class__.__name__)
        table = tab(export=export, title=title)

        for opt in sorted(self.doc_help):
            if hasattr(self, opt):
                descr = self.doc_help[opt]
                rows.append([opt, descr, self.get_value(opt)])
            else:
                warn_msg = 'Setting object {:s} has no \'{:s}\' option. Correct in doc_help.'.format(self.__name__, opt)
                print(warn_msg)
        table.add_rows(rows, header=False)  # use guess_header()

        ext = 'txt'
        if export == 'latex':
            ext = 'tex'

        results = table.draw()

        if save is False:
            print(results)
        else:
            filename = 'settings_help' + '.' + ext
            try:
                f=open(filename, 'w')
                f.write(table)
                f.close()
            except IOError:
                print(results)
                print('Error saving settings help to file')

    @cached
    def doc_help(self):
        descriptions = {}
        return descriptions
