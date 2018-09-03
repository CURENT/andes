from andes.utils.tab import simpletab
from cvxopt import matrix


def format_newline():
    return '\n'


def format_title(item):
    return item


def format_item(item, val):
    return '{:20s} {:s}'.format(item, str(val))


def format_table(header, data, title=None):
    # fmt = ".4g"
    # return tabulate(data, headers=header, floatfmt=fmt)
    table = simpletab(data=data, header=header)
    return table.draw()
    # return []
    # pass


def dump_data(text, header, rowname, data, file):
    width = 14
    precision = 4
    s = ''
    out = ''
    fid = open(file, 'w')

    for Text, Header, Rowname, Data in zip(text, header, rowname, data):
        # Write Text
        if Text:
            fid.writelines(
                Text
            )

        # Write Header
        if Header:
            ncol = len(Header)
            s = ' ' * width
            s += '{:>{width}s}' * ncol + '\n'
            fid.writelines(s.format(*Header, width=width))  # Mind the asterisk
            fid.write('\n')

        # Append Rowname to Data
        # Data is a list of column lists
        if Rowname:
            ncol = 0
            for idx, item in enumerate(Rowname):  # write by row as always
                if not Data:
                    out = ''
                elif isinstance(Data[0], list):  # list of list in Data
                    ncol = len(Data)
                    out = [Data[i][idx] for i in range(ncol)]
                elif isinstance(Data[0],
                                (int, float)):  # Is just a list of numbers
                    ncol = 1
                    out = [Data[idx]]
                elif isinstance(Data, (int, float)):
                    ncol = 1
                    out = [Data]
                elif isinstance(Data, matrix):  # Data is a matrix
                    pass
                else:
                    print('Unexpected Data during output, in formats/txt.py')

                s = '{:{width}s}'  # for row header
                for col in out:
                    if type(col) in (int, float):
                        s += '{:{width}.{precision}g}'
                    elif type(col) == str:
                        if len(col) > width:
                            col = col[:width]
                        s += '{:{width}s}'
                    else:
                        pass
                s += '\n'

                fid.write(
                    s.format(
                        str(item), *out, width=width, precision=precision))
        fid.write('\n')

    fid.close()
