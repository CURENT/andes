#!/usr/bin/env python3
"""
ANDES, a power system simulation tool for research.

Copyright 2015-2017 Hantao Cui

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import re
from argparse import ArgumentParser
from distutils.spawn import find_executable

from matplotlib import pyplot, rc

lfile = []
dfile = []


def cli_parse():
    """command line input parser"""
    parser = ArgumentParser(prog='andesplot')
    parser.add_argument('datfile', nargs=1, default=[], help='dat file name.')
    parser.add_argument('x', nargs=1, type=int, help='x axis variable index')
    parser.add_argument('y', nargs='*', help='y axis variable index')
    parser.add_argument('--xmax', type=float, help='x axis maximum value')
    parser.add_argument('--ymax', type=float, help='y axis maximum value')
    parser.add_argument('--ymin', type=float, help='y axis minimum value')
    parser.add_argument('--xmin', type=float, help='x axis minimum value')
    parser.add_argument('--checkinit', action='store_true', help='check initialization value')
    parser.add_argument('-x', '--xlabel', type=str, help='manual set x-axis text label')
    parser.add_argument('-y', '--ylabel', type=str, help='y-axis text label')
    parser.add_argument('-s', '--save', action='store_true', help='save to file')
    parser.add_argument('-g', '--grid', action='store_true', help='grid on')
    parser.add_argument('-d', '--no_latex', action='store_true', help='disable LaTex formatting')
    parser.add_argument('-u', '--unattended', action='store_true', help='do not show the plot window')
    parser.add_argument('--ytimes', type=str, help='y times')
    args = parser.parse_args()
    return vars(args)


def parse_y(y, nvars):
    ylist = y
    colon = re.compile('\\d*:\\d*:?\\d?')
    if len(y) == 1:
        if isint(ylist[0]):
            ylist[0] = int(ylist[0])
        elif colon.search(y[0]):
            ylist = y[0].split(':')
            ylist = [int(i) for i in ylist]
            if len(ylist) == 2:
                ylist.append(1)

            if ylist[0] > nvars or ylist[0] < 1:
                print('* Warning: Check the starting Y range')
            if ylist[1] > nvars or ylist[1] < 0:
                print('* Warning: Check the ending Y range')

            if ylist[0] < 1:
                ylist[0] = 1
            elif ylist[0] > nvars:
                ylist[0] = nvars

            if ylist[1] < 0:
                ylist[1] = 0
            elif ylist[1] > nvars:
                ylist[1] = nvars

            # ylist = eval('range({}, {}, {})'.format(*ylist))
            ylist = range(*ylist)
        else:
            print('* Error: Wrong format for y range')
    elif len(y) > 1:
        ylist = [int(i) for i in y]
    return ylist


def get_nvars(dat):
    try:
        with open(dat, 'r') as f:
            line1 = f.readline()
        line1 = line1.strip().split()
        return len(line1)
    except IOError:
        print('* Error while opening the dat file')


def read_dat(dat, x, y):
    global dfile
    errid = 0
    xv = []
    yv = [list() for _ in range(len(y))]

    try:
        dfile = open(dat)
        dfile_raw = dfile.readlines()
        dfile.close()
    except IOError:
        print('* Error while opening the dat file')
        return None, None

    for num, line in enumerate(dfile_raw):
        if num == 0:
            continue
        thisline = line.rstrip('\n').split()
        if not (x[0] <= len(thisline) and max(y) <= len(thisline)):
            errid = 1
            break

        xv.append(float(thisline[x[0]]))

        for idx, item in enumerate(y):
            yv[idx].append(float(thisline[item]))

    if errid:
        raise IndexError('x or y index out of bound')

    return xv, yv


def read_label(lst, x, y):
    global lfile
    xl = [list() for _ in range(2)]
    yl = [list() for _ in range(2)]
    yl[0] = [''] * len(y)
    yl[1] = [''] * len(y)

    xy = list(x)
    xy.extend(y)

    try:
        lfile = open(lst)
        lfile_raw = lfile.readlines()
        lfile.close()
    except IOError:
        print('* Error while opening the lst file')
        return None, None

    xidx = sorted(range(len(xy)), key=lambda i: xy[i])
    xsorted = sorted(xy)
    at = 0

    for line in lfile_raw:
        thisline = line.rstrip('\n').split(',')
        thisline = [item.lstrip() for item in thisline]
        if not isfloat(thisline[0].strip()):
            continue

        varid = int(thisline[0])
        if varid == xsorted[at]:
            if xsorted[at] == xy[0]:
                xl[0] = thisline[1]
                xl[1] = thisline[2].strip('#')
            else:
                yl[0][xidx[at] - 1] = thisline[1]
                yl[1][xidx[at] - 1] = thisline[2].strip('#')
            at += 1

        if at >= len(xy):
            break

    return xl, yl


def do_plot(x, y, xl, yl, args, fig=None, ax=None):
    xmin = args.pop('xmin', None)
    xmax = args.pop('xmax', None)
    ymax = args.pop('ymax', None)
    ymin = args.pop('ymin', None)
    xlabel = args.pop('xlabel', None)
    ylabel = args.pop('ylabel', None)
    no_latex = args.pop('no_latex', None)

    LATEX = False
    if no_latex:
        LATEX = False
    elif find_executable('dvipng'):
        LATEX = True

    if LATEX:
        rc('text', usetex=True)
    else:
        rc('text', usetex=False)
    rc('font', family='Arial', size=12)

    if not y:
        return
    style = ['-', '--', '-.', ':'] * len(y)

    if not xmin:
        xmin = x[0] - 1e-6
    if not xmax:
        xmax = x[-1] + 1e-6

    if LATEX:
        xl_data = xl[1]
        yl_data = yl[1]
    else:
        xl_data = xl[0]
        yl_data = yl[0]

    if not (fig and ax):
        fig, ax = pyplot.subplots()

    for idx in range(len(y)):
        ax.plot(x, y[idx], label=yl_data[idx], ls=style[idx])

    if not xlabel:
        ax.set_xlabel(xl_data)
    else:
        if LATEX:
            xlabel = '$' + xlabel.replace(' ', '\ ') + '$'
        ax.set_xlabel(xlabel)

    if ylabel:
        if LATEX:
            ylabel = '$' + ylabel.replace(' ', '\ ') + '$'
        ax.set_ylabel(ylabel)

    ax.ticklabel_format(useOffset=False)

    ax.set_xlim(xmin=xmin)
    ax.set_xlim(xmax=xmax)
    ax.set_ylim(ymax=ymax)
    ax.set_ylim(ymin=ymin)

    if args.pop('grid', None):
        ax.grid(b=True, linestyle='--')
    legend = ax.legend(loc='upper right')

    pyplot.draw()

    # output to file

    if args.pop('save', None) or args.pop('unattended', None):
        name, _ = os.path.splitext(args['datfile'])
        count = 1
        cwd = os.getcwd()
        for file in os.listdir(cwd):
            if file.startswith(name) and file.endswith('.png'):
                count += 1

        outfile = name + '_' + str(count) + '.png'

        try:
            fig.savefig(outfile, dpi=300)
            print('Figure saved to file {}'.format(outfile))
        except:
            print('* Error occurred while rendering. Please try disabling LaTex with "-d".')
            return

    if not args.pop('unattended', None):
        try:
            pyplot.show()
        except:
            print('* Error occurred while rendering. Please try disabling LaTex with "-d".')
            return

    return fig, ax


def add_plot(x, y, xl, yl, fig, ax, LATEX=False):
    """Add plots to an existing plot"""
    if LATEX:
        xl_data = xl[1]
        yl_data = yl[1]
    else:
        xl_data = xl[0]
        yl_data = yl[0]

    for idx in range(len(y)):
        ax.plot(x, y[idx], label=yl_data[idx])

    legend = ax.legend(loc='upper right')
    ax.set_ylim(auto=True)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def main(cli=True, **args):
    if cli:
        args = cli_parse()

    name, ext = os.path.splitext(args['datfile'][0])
    if 'out' in name:
        tds_plot(name, args)
    elif 'eig' in name:
        eig_plot(name, args)


def eig_plot(name, args):
    fullpath = os.path.join(name, '.txt')
    raw_data = []
    started = 0
    fid = open(fullpath)
    for line in fid.readline():
        if '#1' in line:
            started = 1
        elif 'PARTICIPATION FACTORS' in line:
            started = -1

        if started == 1:
            raw_data.append(line)
        elif started == -1:
            break
    fid.close()

    for line in raw_data:
        data = line.split()


def tds_plot(name, args):
    dat = os.path.join(os.getcwd(), name + '.dat')
    lst = os.path.join(os.getcwd(), name + '.lst')

    y = parse_y(args['y'], get_nvars(dat))
    try:
        xval, yval = read_dat(dat, args['x'], y)
    except IndexError:
        print('* Error: X or Y index out of bound')
        return

    xl, yl = read_label(lst, args['x'], y)

    if args.pop('checkinit', False):
        check_init(yval, yl[0])
        return
    if args.pop('ytimes', False):
        times = float(args['ytimes'])
        new_yval = []
        for val in yval:
            new_yval.append([i*times for i in val])
        yval = new_yval
    do_plot(xval, yval, xl, yl, args)


def check_init(yval, yl):
    """"Check initialization by comparing t=0 and t=end values"""
    suspect = []
    for var, label in zip(yval, yl):
        if abs(var[0] - var[-1]) >= 1e-6:
            suspect.append(label)
    if suspect:
        print('Initialization failure:')
        print(', '.join(suspect))
    else:
        print('Initialization is correct.')


if __name__ == "__main__":
    main(cli=True)
