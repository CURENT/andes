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
                 max_width=80):
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
        """elem_add n cols of spaces before the first col.
           (for texttable 0.8.3)"""
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
        empty_line = '\n\n'
        return pre + str(Texttable.draw(self)) + empty_line


class SimpleTab(object):
    """A simple and faster table class for static report outputs."""

    def __init__(self, header=None, title=None, data=None, side=None):
        """header: a list of strings,
        title: an str
        data: a list of list of numbers or str"""
        self.header = header
        self.title = title
        self.data = data
        self.side = side
        self._width = [10] * len(header) if header else []

    def _guess_width(self):
        """auto fit column width"""
        if len(self.header) <= 4:
            nspace = 6
        elif len(self.header) <= 6:
            nspace = 5
        else:
            nspace = 4
        ncol = len(self.header)
        self._width = [nspace] * ncol
        width = [0] * ncol

        # set initial width from header
        for idx, item in enumerate(self.header):
            width[idx] = len(str(item))

        # guess width of each column from first 10 lines of data
        samples = min(len(self.data), 10)
        for col in range(ncol):
            for idx in range(samples):
                data = self.data[idx][col]
                if not isinstance(data, (float, int)):
                    temp = len(data)
                else:
                    temp = 10
                if temp > width[col]:
                    width[col] = temp

        for col in range(ncol):
            self._width[col] += width[col]

    def draw(self):
        """Draw the stored table and return as a list (#TODO)"""

        self._guess_width()
        data = list()
        out = ''
        fmt = 's'

        # header first
        for item, width in zip(self.header, self._width):
            out += '{text:<{width}{fmt}}'.format(
                text=str(item), width=width, fmt=fmt)
        data.append(out)

        for line in self.data:
            out = ''
            for item, width in zip(line, self._width):
                if isinstance(item, (int, float)):
                    # item = round(item, 4)
                    fmt = 'g'
                    out += '{val:< {width}.{dec}{fmt}}'.format(
                        val=item, width=width, dec=5, fmt=fmt)

                elif isinstance(item, str):
                    fmt = 's'
                    out += '{val:<{width}{fmt}}'.format(
                        val=item, width=width, fmt=fmt)
            data.append(out)
        return data
