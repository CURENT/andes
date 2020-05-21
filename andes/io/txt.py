import os
from andes.shared import np


def dump_data(text, header, rowname, data, file, width=14, precision=5):
    out = ''

    os.makedirs(os.path.abspath(os.path.dirname(file)), exist_ok=True)
    with open(file, 'w') as fid:

        for Text, Header, Rowname, Data in zip(text, header, rowname, data):
            # Write Text
            if Text:
                fid.writelines(Text)

            # Write Header
            if Header:
                ncol = len(Header)
                s = ' ' * width
                s += '{:>{width}s}' * ncol + '\n'
                fid.writelines(s.format(*Header, width=width))  # Mind the asterisk
                fid.write('\n')

            # Append Rowname to Data
            # Data is a list of column lists
            if Rowname is not None:
                ncol = 0
                for idx, item in enumerate(Rowname):  # write by row as always
                    if not Data:
                        out = ''
                    elif isinstance(Data, (int, float)):
                        out = [Data]
                    elif isinstance(Data[0], (int, float)):  # is a list of numbers
                        out = [Data[idx]]
                    elif isinstance(Data[0], (list, np.ndarray)):  # list of list in Data
                        ncol = len(Data)
                        out = [Data[i][idx] for i in range(ncol)]
                    else:
                        print(Data)
                        print('Unexpected Data during output, in formats/txt.py')

                    s = '{:{width}s}'  # for row header
                    for ii, col in enumerate(out):
                        if isinstance(col, (int, float)):
                            s += '{:>{width}.{precision}g}'
                        elif isinstance(col, str):
                            if len(col) > width:
                                out[ii] = col[:width]
                            s += '{:>{width}s}'
                        elif col is None:
                            out[ii] = 'None'
                            s += '{:>{width}s}'
                        else:
                            pass
                    s += '\n'

                    fid.write(
                        s.format(
                            str(item), *out, width=width, precision=precision))
            fid.write('\n')
