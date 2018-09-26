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

'''Added by Evan'''
import csv

from argparse import ArgumentParser
from distutils.spawn import find_executable

from matplotlib import rc
from matplotlib import pyplot as plt
from numpy import concatenate, transpose

lfile = []
dfile = []




'''TruDat is a cleandat(matrix), nvars, and nsamples.'''
class TruDat:
    
    #Instantiation
    def __init__(self, datfilename):
        self.cleandat = [['','']]
        nvars = get_nvars(datfilename)
        lst = os.path.join(os.getcwd(), datfilename[0:-4] + '.lst')

        '''Read labels and make first 2 columns'''
    
        q, ys1 = read_label(lst, [0], list(range(1,nvars)))
        self.cleandat[0][0] = q[0]
        self.cleandat[0][1] = q[1]
        for i in range(len(ys1[0])):
            self.cleandat.append(['',''])
            self.cleandat[i+1][0] = ys1[0][i]
            self.cleandat[i+1][1] = ys1[1][i]

        '''Compensate for off-by-one error by bringing in opening values'''
        with open(datfilename) as f:
            first_line = f.readline().split()

        for i in range(0,nvars):
            self.cleandat[i].append(first_line[i])

        '''Read data and make rest of columns'''
            
        q, ys2 = read_dat(datfilename, [0], list(range(1,nvars)))
        self.cleandat[0].extend(q)
        for i in range(nvars-1):
            self.cleandat[i+1].extend(ys2[i])

        self.nvars = len(self.cleandat)
        self.nsamples = len(self.cleandat[0])-2

    '''Input string and prints matching vars'''
    def dfind(self, query):
        for i in range(len(self.cleandat)):
            if query.lower() in self.cleandat[i][0].lower() or query == '':
                print(i,self.cleandat[i][0])

    '''Creates a csv file of your data in local dir, 1 row per variable'''
    def csvout(self, filename):
        fullname = filename + '.csv'
        with open(fullname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.cleandat)
    

    '''Prints the data, not truncated'''
    def show(self):
        dobj = self.cleandat
        for i in range(len(dobj)):
            entry1 = dobj[i][0]
            entry2 = dobj[i][1]
            print('{:<15}'.format(entry1), end=' ')
            print('{:>25}'.format(entry2), end=' ')
            for k in range(2,len(dobj[i])):
                entry = dobj[i][k]
                print('{:12}'.format(entry), end=' ')
            print('', end='\n')

    '''Show the data truncated for visibility (Default truncates after 2 samples)'''
    def showt(self):
        dobj = self.cleandat
        if len(dobj) < 4:
            print('Too few samples to truncate (Default min = 2).')
        for i in range(len(dobj)):
            entry1 = dobj[i][0]
            entry2 = dobj[i][1]
            print('{:<15}'.format(entry1), end=' ')
            print('{:>25}'.format(entry2), end=' ')
            for k in range(2,4):
                entry = dobj[i][k]
                print('{:12}'.format(entry), end=' ')
            print('...', end='\n')        




"""Takes in dat filename, creates a matrix of ALL data w label names as 1st 2 cols"""
def clean_dat(datfilename):
    cleandat = [['','']]
    nvars = get_nvars(datfilename)
    lst = os.path.join(os.getcwd(), datfilename[0:-4] + '.lst')

    '''Read labels and make first 2 columns'''
    
    q, ys1 = read_label(lst, [0], list(range(1,nvars)))
    cleandat[0][0] = q[0]
    cleandat[0][1] = q[1]
    for i in range(len(ys1[0])):
        cleandat.append(['',''])
        cleandat[i+1][0] = ys1[0][i]
        cleandat[i+1][1] = ys1[1][i]

    '''Compensate for off-by-one error by bringing in opening values'''
    with open(datfilename) as f:
        first_line = f.readline().split()

    for i in range(0,nvars):
        cleandat[i].append(first_line[i])

    '''Read data and make rest of columns'''
        
    q, ys2 = read_dat(datfilename, [0], list(range(1,nvars)))
    cleandat[0].extend(q)
    for i in range(nvars-1):
        cleandat[i+1].extend(ys2[i])
        
    return cleandat

'''Input string and data object (made with clean_dat) and prints matching vars'''
def dfind(dobj, query):
    for i in range(len(dobj)):
        if query.lower() in dobj[i][0].lower() or query == '':
            print(i,dobj[i][0])

'''Creates a csv file of your data in local dir, 1 row per variable'''
def csvout(dobj, filename):
    fullname = filename + '.csv'
    with open(fullname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dobj)
    

'''Prints the data, not truncated'''
def show(dobj):
    for i in range(len(dobj)):
        entry1 = dobj[i][0]
        entry2 = dobj[i][1]
        print('{:<15}'.format(entry1), end=' ')
        print('{:>25}'.format(entry2), end=' ')
        for k in range(2,len(dobj[i])):
            entry = dobj[i][k]
            print('{:12}'.format(entry), end=' ')
        print('', end='\n')

'''Show the data truncated for visibility (Default truncates after 2 samples)'''
def showt(dobj):
    if len(dobj) < 4:
        print('Too few samples to truncate (Default min = 2).')
    for i in range(len(dobj)):
        entry1 = dobj[i][0]
        entry2 = dobj[i][1]
        print('{:<15}'.format(entry1), end=' ')
        print('{:>25}'.format(entry2), end=' ')
        for k in range(2,4):
            entry = dobj[i][k]
            print('{:12}'.format(entry), end=' ')
        print('...', end='\n')



    

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
    parser.add_argument(
        '--checkinit', action='store_true', help='check initialization value')
    parser.add_argument(
        '-x', '--xlabel', type=str, help='manual set x-axis text label')
    parser.add_argument('-y', '--ylabel', type=str, help='y-axis text label')
    parser.add_argument(
        '-s', '--save', action='store_true', help='save to file')
    parser.add_argument('-g', '--grid', action='store_true', help='grid on')
    parser.add_argument(
        '-d',
        '--no_latex',
        action='store_true',
        help='disable LaTex formatting')
    parser.add_argument(
        '-u',
        '--unattended',
        action='store_true',
        help='do not show the plot window')
    parser.add_argument('--ytimes', type=str, help='y times')
    parser.add_argument(
        '--dpi', type=int, help='image resolution in dot per inch (DPI)')
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


def do_plot(xdata,
            ydata,
            xname=None,
            yname=None,
            fig=None,
            ax=None,
            dpi=200,
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            xlabel=None,
            ylabel=None,
            no_latex=False,
            legend=True,
            grid=False,
            save=False,
            unattended=False,
            datfile='',
            noshow=False,
            **kwargs):

    # set styles and LaTex
    rc('font', family='Arial', size=12)
    linestyles = ['-', '--', '-.', ':'] * len(ydata)
    if not no_latex and find_executable('dvipng'):
        # use LaTex
        LATEX = True
        rc('text', usetex=True)
    else:
        LATEX = False
        rc('text', usetex=False)

    # get variable names from lst
    def get_lst_name(lst, LATEX):
        idx = 1 if LATEX else 0
        if lst is not None:
            return lst[idx]
        else:
            return None

    xl_data = get_lst_name(xname, LATEX)
    yl_data = get_lst_name(yname, LATEX)

    # set default x min based on simulation time
    if not xmin:
        xmin = xdata[0] - 1e-6
    if not xmax:
        xmax = xdata[-1] + 1e-6

    if not (fig and ax):
        fig = plt.figure(dpi=dpi)
        ax = plt.gca()

    for idx in range(len(ydata)):
        yl_data_idx = yl_data[idx] if yl_data else None
        ax.plot(xdata, ydata[idx], label=yl_data_idx, ls=linestyles[idx])

    if not xlabel:
        if xl_data is not None:
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

    if grid:
        ax.grid(b=True, linestyle='--')
    if legend and yl_data:
        legend = ax.legend(loc='upper right')

    plt.draw()

    # output to file

    if save or unattended:
        name, _ = os.path.splitext(datfile[0])
        count = 1
        cwd = os.getcwd()
        for file in os.listdir(cwd):
            if file.startswith(name) and file.endswith('.png'):
                count += 1

        outfile = name + '_' + str(count) + '.png'

        try:
            fig.savefig(outfile, dpi=1200)
            print('Figure saved to file {}'.format(outfile))
        except IOError:
            print('* Error occurred. Try disabling LaTex with "-d".')
            return

    if unattended:
        noshow = True

    if not noshow:
        plt.show()

    return fig, ax


def add_plot(x, y, xl, yl, fig, ax, LATEX=False, linestyle=None, **kwargs):
    """Add plots to an existing plot"""
    if LATEX:
        xl_data = xl[1]  # NOQA
        yl_data = yl[1]
    else:
        xl_data = xl[0]  # NOQA
        yl_data = yl[0]

    for idx in range(len(y)):
        ax.plot(x, y[idx], label=yl_data[idx], linestyle=linestyle)

    ax.legend(loc='upper right')
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
        # data = line.split()
        # TODO: complete this function
        pass


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
    ytimes = args.pop('ytimes', False)
    if ytimes:
        times = float(ytimes)
        new_yval = []
        for val in yval:
            new_yval.append([i * times for i in val])
        yval = new_yval

    args.pop('x')
    args.pop('y')
    do_plot(xval, yval, xl, yl, **args)


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

















'''Test commands
dffn = 'runtelldat5x10.dat'
df = TruDat(dffn)
df.show()
df.showt()
df.dfind('xt bus')
df.csvout('datcsv')
'''





if __name__ == "__main__":
    main(cli=True)









