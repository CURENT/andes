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
from argparse import ArgumentParser
from matplotlib import pyplot
import matplotlib as mpl
import os
import re
lfile = []
dfile = []

try:
    from blist import *
    BLIST = 1
except ImportError:
    BLIST = 0


def cli_parse():
    """command line input parser"""
    parser = ArgumentParser(prog='andesplot')
    parser.add_argument('datfile', nargs=1, default=[], help='dat file name.')
    parser.add_argument('x', nargs=1, type=int, help='x axis variable index')
    parser.add_argument('y', nargs='*', help='y axis variable index')
    parser.add_argument('--xmax', type=float, help='x axis maximum value')
    parser.add_argument('--xmin', type=float, help='x axis minimum value')
    parser.add_argument('--checkinit', action='store_true', help='check initialization value')
    parser.add_argument('--ylabel', type=str, help='y-axis text label')
    args = parser.parse_args()
    return args


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

            ylist = eval('range({}, {}, {})'.format(*ylist))
        else:
            print('* Error: Wrong format for y range')
    elif len(y) > 1:
        ylist = [int(i) for i in y]
    return ylist


def get_nvars(dat):
    try:
        with open(dat, 'r') as f:
            line1 = f.readline()
        line1 = line1.strip('\n').split()
        return int(line1[0])
    except IOError:
        print('* Error while opening the dat or lst files')


def read_dat(dat, x, y):
    global dfile
    errid = 0
    xv = []
    yv = [list() for _ in range(len(y))]

    try:
        dfile = open(dat)
    except IOError:
        print('* Error while opening the dat or lst files')
        return None, None

    for num, line in enumerate(dfile.readlines()):
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

    x.extend(y)
    try:
        lfile = open(lst)
    except IOError:
        print('* Error while opening the dat or lst files')
        return None, None

    xidx = sorted(range(len(x)), key=lambda i: x[i])
    xsorted = sorted(x)
    at = 0

    for line in lfile.readlines():
        thisline = line.rstrip('\n').split(',')
        thisline = [item.lstrip() for item in thisline]
        if not isfloat(thisline[0].strip()):
            continue

        varid = int(thisline[0])
        if varid == xsorted[at]:
            if xsorted[at] == x[0]:
                xl[0] = thisline[1]
                xl[1] = thisline[2].strip('#')
            else:
                yl[0][xidx[at] - 1] = thisline[1]
                yl[1][xidx[at] - 1] = thisline[2].strip('#')
            at += 1

        if at >= len(x):
            break

    return xl, yl


def do_plot(x, y, xl, yl, xmin=None, xmax=None, ylabel=None):
    # Configurate matplotlib
    mpl.rc('font', family='Arial')
    style = ['-', '--', '-.', ':'] * len(y)

    if not xmin:
        xmin = x[0] - 1e-6
    if not xmax:
        xmax = x[-1] + 1e-6

    fig, ax = pyplot.subplots()
    for idx in range(len(y)):
        ax.plot(x, y[idx], label=yl[0][idx], ls=style[idx])

    ax.set_xlabel(xl[0])
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.ticklabel_format(useOffset=False)
    ax.set_xlim(xmin=xmin)
    ax.set_xlim(xmax=xmax)

    legend = ax.legend(loc='upper right')

    pyplot.show()


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


def main():
    args = cli_parse()
    name, ext = os.path.splitext(args.datfile[0])
    dat = os.path.join(os.getcwd(), name + '.dat')
    lst = os.path.join(os.getcwd(), name + '.lst')

    y = parse_y(args.y, get_nvars(dat))
    try:
        xval, yval = read_dat(dat, args.x, y)
    except IndexError:
        print('* Error: X or Y index out of bound')
        return

    xl, yl = read_label(lst, args.x, y)

    if args.checkinit:
        check_init(yval, yl[0])
        return

    do_plot(xval, yval, xl, yl, xmin=args.xmin, xmax=args.xmax, ylabel=args.ylabel)


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
    main()
