from texttable import Texttable


class Tab(Texttable):
    """
    Use package texttable to create fine-formatted tables for setting helps
    and device helps. Avoid using this class for static report formatting as
    it may be slow.
    """

    def __init__(self,
                 title=None,
                 header=None,
                 descr=None,
                 data=None,
                 export='plain'):
        Texttable.__init__(self)
        self.set_chars(['-', '|', '+', '-'])
        self.set_deco(Texttable.HEADER
                      | Texttable.VLINES)  # Texttable.BORDER | Texttable.HLINE
        self.set_precision(3)

        self._title = title
        self._descr = descr
        self._format = export  # outformat in ['plain', 'latex', 'html']
        if header is not None:
            self.header(header)
        if data is not None:
            self.add_rows(data, header=False)

    def header(self, array):
        Texttable.header(self, array)
        self.auto_style()

    def guess_header(self):
        if self._header:
            return
        header = ''
        if self._row_size == 3:
            header = ['Option', 'Description', 'Value']
        elif self._row_size == 4:
            header = ['Parameter', 'Description', 'Value', 'Unit']
        self.header(header)

    def auto_style(self):
        """
        automatic styling according to _row_size
        76 characters in a row
        """
        if self._row_size is None:
            return
        elif self._row_size == 3:
            self.set_cols_align(['l', 'l', 'l'])
            self.set_cols_valign(['t', 't', 't'])
            self.set_cols_width([12, 54, 12])
        elif self._row_size == 4:
            self.set_cols_align(['l', 'l', 'l', 'l'])
            self.set_cols_valign(['t', 't', 't', 't'])
            self.set_cols_width([10, 40, 10, 10])
            # TODO: third column use scientific notation for small values

    def set_title(self, val):
        self._title = val

    def add_left_space(self, nspace=1):
        """elem_add n cols of spaces before the first col.
           (for texttable 0.8.3)"""
        sp = ' ' * nspace
        for item in self._rows:
            item[0] = sp + item[0]

    def draw(self):
        """generate texttable formatted string"""
        self.guess_header()
        self.add_left_space(
        )
        # for Texttable, elem_add a column of whitespace on the left for
        # better visual effect
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


class simpletab(object):
    """A simple and faster table class for static report output"""

    def __init__(self, header=None, title=None, data=None, side=None):
        """header: a list of strings,
        title: an str
        data: a list of list of numbers or str"""
        self.header = header
        self.title = title
        self.data = data
        self.side = side
        self._width = [10] * len(header) if header else []

    def guess_width(self):
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
        self.guess_width()
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
