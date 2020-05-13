from andes.shared import Texttable


class Tab(Texttable):
    """
    Use package ``texttable`` to create well-formatted tables for setting helps
    and device helps.

    Parameters
    ----------
    export : ('plain', 'rest')
        Export format in plain text or restructuredText.
    max_width : int
        Maximum table width. If there are equations in cells, set to 0 to disable wrapping.
    """

    def __init__(self,
                 title=None,
                 header=None,
                 descr=None,
                 data=None,
                 export='plain',
                 max_width=78):
        Texttable.__init__(self, max_width=max_width)
        if export == 'plain':
            self.set_chars(['-', '|', '+', '-'])
            self.set_deco(Texttable.HEADER | Texttable.VLINES)  # Texttable.BORDER | Texttable.HLINE
        self.set_precision(3)

        self._title = title
        self._descr = descr
        if header is not None:
            self.header(header)
        if data is not None:
            self.add_rows(data, header=False)

    def header(self, header_list):
        """Set the header with a list."""
        Texttable.header(self, header_list)

    def _guess_header(self):
        if self._header:
            return
        header = ''
        if self._row_size == 3:
            header = ['Option', 'Description', 'Value']
        elif self._row_size == 4:
            header = ['Parameter', 'Description', 'Value', 'Unit']
        self.header(header)

    def set_title(self, val):
        """
        Set table title to ``val``.
        """
        self._title = val

    def _add_left_space(self, nspace=1):
        """
        Add n cols of spaces before the first col. (for texttable 0.8.3)
        """
        sp = ' ' * nspace
        for item in self._rows:
            item[0] = sp + item[0]

    def draw(self):
        """
        Draw the table and return it in a string.
        """
        self._guess_header()
        self._add_left_space()

        # for Texttable, add a column of whitespace on the left for better visual effect
        if self._title and self._descr:
            pre = self._title + '\n' + self._descr + '\n\n'
        elif self._title:
            pre = self._title + '\n\n'
        elif self._descr:
            pre = 'Empty Title' + '\n' + self._descr + '\n'
        else:
            pre = ''
        return pre + str(Texttable.draw(self)) + '\n\n'


def make_doc_table(title, max_width, export, plain_dict, rest_dict):
    """
    Helper function to format documentation data into tables.
    """
    data_dict = rest_dict if export == 'rest' else plain_dict
    table = Tab(title=title, max_width=max_width, export=export)
    table.header(list(data_dict.keys()))
    rows = list(map(list, zip(*list(data_dict.values()))))
    table.add_rows(rows, header=False)

    return table.draw()


def math_wrap(tex_str_list, export):
    """
    Warp each string item in a list with latex math environment ``$...$``.

    Parameters
    ----------
    tex_str_list : list
        A list of equations to be wrapped
    export : str, ('rest', 'plain')
        Export format. Only wrap equations if export format is ``rest``.
    """
    if export != 'rest':
        return list(tex_str_list)

    out = []
    for item in tex_str_list:
        if item is None or item == '':
            out.append('')
        else:
            out.append(rf':math:`{item}`')
    return out
