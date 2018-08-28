#!/usr/bin/env python3

# ANDES, a power system simulation tool for research.
#
# Copyright 2015-2017 Hantao Cui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Andes main entry points
"""


import os
import sys
import glob
import io
import pstats
import cProfile
from time import sleep
from multiprocessing import Process
from argparse import ArgumentParser

from . import filters
from .consts import *
from .system import PowerSystem
from .utils import elapsed
from .variables import preamble
from .routines import powerflow, timedomain, eigenanalysis
# from .routines.fakemodule import EAGC


def cli_parse(help=False):
    """command line input argument parser"""

    parser = ArgumentParser(prog='andes')
    parser.add_argument('casename', nargs='*', default=[], help='Case file name.')

    # program general options
    parser.add_argument('-x', '--exit', help='Exit before solving the power flow. Enable this option to '
                                             'use andes ad a file format converter.', action='store_true')
    parser.add_argument('--no_preamble', action='store_true', help='Hide preamble')
    parser.add_argument('--license', action='store_true', help='Display MIT license and exit.')
    parser.add_argument('--version', action='store_true', help='Print current andes version and exit.')
    parser.add_argument('--warranty', action='store_true', help='Print warranty info and exit.')
    parser.add_argument('--what', action='store_true', help='Show me something and exit', default=None)
    parser.add_argument('-n', '--no_output', help='Force not to write any output, including log,'
                                                  'outputs and simulation dumps', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Enable Python profiler.')
    parser.add_argument('--dime', help='Speficy DiME streaming server address and port.')
    parser.add_argument('--tf', help='End time of time-domain simulation.', type=float)
    parser.add_argument('-l', '--log', help='Specify the name of log file.')
    parser.add_argument('-d', '--dat', help='Specify the name of file to save simulation results.')
    parser.add_argument('--ncpu', help='number of parallel processes', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Program logging level, an integer from 1 to 5.'
                                                'The level corresponding to TODO=0, DEBUG=10, INFO=20, WARNING=30,'
                                                'ERROR=40, CRITICAL=50, ALWAYS=60. The default verbose level is 20.',
                        type=int)
    parser.add_argument('--conf', help='Edit routine config', type=str, choices=('general', 'spf', 'tds', 'cpf', 'sssa'))

    # file and format related
    parser.add_argument('-c', '--clean', help='Clean output files and exit.', action='store_true')
    parser.add_argument('-p', '--path', help='Path to case files', default='')
    parser.add_argument('-P', '--pert', help='Path to perturbation file', default='')
    parser.add_argument('-s', '--settings', help='Specify a setting file. This will take precedence of .andesrc '
                                                 'file in the home directory.')
    parser.add_argument('-i', '--input_format', help='Specify input case file format.')
    parser.add_argument('-o', '--output_format', help='Specify output case file format. For example txt, latex.')
    parser.add_argument('-O', '--output', help='Specify the output file name. For different routines the same name'
                                               'as the case file with proper suffix and extension wil be given.')
    parser.add_argument('-a', '--addfile', help='Include additional files used by some formats.')
    parser.add_argument('-D', '--dynfile', help='Include an additional dynamic file in dm format.')
    parser.add_argument('-J', '--gis', help='JML format GIS file.')
    parser.add_argument('-m', '--map', help='Visualize power flow results on GIS. Neglected if no GIS file is given.')
    parser.add_argument('-e', '--dump_raw', help='Dump RAW format case file.')  # consider being used as batch converter
    parser.add_argument('-Y', '--summary', help='Show summary and statistics of the data case.', action='store_true')

    # Solver Options
    parser.add_argument('-r', '--routine', help='Routine after power flow solution: t[TD], c[CPF], s[SS], o[OPF].')
    parser.add_argument('-j', '--checkjacs', help='Check analytical Jacobian using numerical differentation.')

    # helps and documentations
    parser.add_argument('-u', '--usage', help='Write command line usage', action='store_true')
    parser.add_argument('-E', '--export', help='Export file format')
    parser.add_argument('-C', '--category', help='Dump device names belonging to the specified category.')
    parser.add_argument('-L', '--model_list', help='Dump the list of all supported model.', action='store_true')
    parser.add_argument('-f', '--model_format', help='Dump the format definition of specified devices,'
                                                     ' separated by comma.')
    parser.add_argument('-Q', '--model_var', help='Dump the variables of specified devices given by <DEV.VAR>.')
    parser.add_argument('-g', '--group', help='Dump all the devices in the specified group.')
    parser.add_argument('-q', '--quick_help', help='Print a quick help of the device.')
    parser.add_argument('--help_option', help='Print a quick help of a setting parameter')
    parser.add_argument('--help_settings', help='Print a quick help of a given setting class. Use ALL'
                                                'for all setting classes.')
    parser.add_argument('-S', '--search', help='Search devices that match the given expression.')

    if help is True:
        return parser.format_help()
    else:
        args = parser.parse_args()
        return args


def andeshelp(usage=None,
              group=None,
              category=None,
              model_list=None,
              model_format=None,
              model_var=None,
              quick_help=None,
              help_option=None,
              help_settings=None,
              export='plain',
              save=None,
              **kwargs):
    """
    Return the help

    :return: True if executed
    """
    out = []

    if not (usage or group or category or model_list or model_format or model_var
            or quick_help or help_option or help_settings):
        return False

    from .models import all_models_list

    ps = PowerSystem()

    if usage:
        out.append(cli_parse(help=True))

    if category:
        raise NotImplementedError

    if model_list:
        raise NotImplementedError

    if model_format:
        if model_format.lower() == 'all':
            model_format = all_models_list
        else:
            model_format = model_format.split(',')

        for item in model_format:
            if item not in all_models_list:
                ps.Log.warning('Model <{}> does not exist.'.format(item))
                model_format.remove(item)

        if len(model_format) > 0:
            for item in model_format:
                out.append(ps.__dict__[item].doc(export=export))

    if model_var:
        model_var = model_var.split('.')

        if len(model_var) == 1:
            ps.Log.error('Model and parameter not separated by dot.')

        elif len(model_var) > 2:
            ps.Log.error('Model parameter not specified correctly.')

        else:
            dev, var = model_var
            if not hasattr(ps, dev):
                ps.Log.error('Model <{}> does not exist.'.format(dev))
            else:
                if var not in ps.__dict__[dev]._data.keys():
                    ps.Log.error('Model <{}> does not have parameter <{}>.'.format(dev, var))
                else:
                    c1 = ps.__dict__[dev]._descr.get(var, 'No Description')
                    c2 = ps.__dict__[dev]._data.get(var)
                    c3 = ps.__dict__[dev]._units.get(var, 'No Unit')
                    out.append('{}: {}, default = {:g} {}'.format('.'.join(model_var), c1, c2, c3))

    if group:
        group_dict = {}

        for model in all_models_list:
            g = ps.__dict__[model]._group
            if g not in group_dict:
                group_dict[g] = []
            group_dict[g].append(model)

        if group.lower() == 'all':
            group = sorted(list(group_dict.keys()))

        else:
            group = [group]
            match = []

            # search for ``group`` in all group names and store in ``match``
            for item in group_dict.keys():
                if group[0].lower() in item.lower():
                    match.append(item)

            group = match

            # if group name not found
            if len(match) == 0:
                sys.stdout.write('Group <{:s}> not found.'.format(group[0]))

        for idx, item in enumerate(group):
            group_models = sorted(list(group_dict[item]))

            out.append('<{:s}>'.format(item))
            out.append(' '.join(group_models))
            out.append('')

    if quick_help:
        if quick_help not in all_models_list:
            sys.stdout.write('Model <{}> does not exist.'.format(quick_help))
        else:
            out.append(ps.__dict__[quick_help].doc(export=export))

    if help_option:
        raise NotImplementedError

    if help_settings:
        all_settings = ['Settings', 'SPF', 'TDS', 'SSSA', 'CPF']

        if help_settings.lower() == 'all':
            help_settings = all_settings

        else:
            help_settings = help_settings.split(',')

            for item in help_settings:
                if item not in all_settings:
                    ps.Log.warning('Setting <{}> does not exist.'.format(item))
                    help_settings.remove(item)

        if len(help_settings) > 0:
            for item in help_settings:
                out.append(ps.__dict__[item].doc(export=export))

    file = sys.stdout if save is None else save
    print('\n'.join(out), file=file)

    return True


def edit_conf(conf):
    """
    Edit Andes routine configuration

    :param conf: name of the routine
    :return: succeed flag
    """
    raise NotImplementedError("Not implemented")


def clean(run=False):
    """Clean up function for generated files"""
    if not run:
        return False

    found = False
    cwd = os.getcwd()

    for file in os.listdir(cwd):
        if file.endswith('_eig.txt') or file.endswith('_out.txt') or file.endswith('_out.lst') or \
                file.endswith('_out.dat') or file.endswith('_prof.txt'):
            found = True
            try:
                os.remove(file)
                print('<{:s}> removed.'.format(file))
            except IOError:
                print('Error removing file <{:s}>.'.format(file))
    if not found:
        print('--> No Andes output found in the working directory.')

    return True


def search(keyword):
    """
    Search for models whose names contain ``keyword``

    :param str keyword: partial or full model name
    :return: a list of model names in <file.model> format
    """

    from .models import non_jits, jits, all_models
    out = []

    if not keyword:
        return out

    keys = sorted(list(all_models.keys()))

    for key in keys:
        vals = all_models[key]
        val = list(vals.keys())
        val = sorted(val)

        for item in val:
            if keyword.lower() in item.lower():
                out.append(key + '.' + item)

    if out:
        print('Search result: <file.model> containing <{}>'.format(keyword))
        print(' '.join(out))
    else:
        print('No model containing <{:s}> found'.format(keyword))

    return out


def main():
    """
    Entry function
    """
    t0, s = elapsed()
    args = cli_parse()
    cases = []
    kwargs = {}

    # run clean-ups and exit
    if clean(args.clean):
        return

    if args.search:
        search(args.search)
        return
    if args.conf:
        edit_conf(args.conf)
        return

    # extract case names
    if len(args.casename) >= 2:
        args.no_preamble = True

    for item in args.casename:
        cases += glob.glob(os.path.join(args.path, item))
    cases = list(set(cases))
    args.casename = None

    # extract all arguments
    for arg, val in vars(args).items():
        if val is not None:
            kwargs[arg] = val

    # dump help and exit
    if andeshelp(**kwargs):
        return

    print(preamble(args.no_preamble))

    # exit if no case specified
    if len(cases) == 0:
        print('--> Data file undefined or path is invalid.')
        print('Use "andes -h" for command line help.')
        return

    # single case study
    elif len(cases) == 1:
        run(cases[0], **kwargs)
        t1, s = elapsed(t0)
        print('--> Single process finished in {0:s}.'.format(s))
        return

    # multiple studies on multiple processors
    else:
        jobs = []
        kwargs['verbose'] = ERROR
        ncpu = kwargs['ncpu']
        if ncpu == 0 or ncpu > os.cpu_count():
            ncpu = os.cpu_count()

        print('Multi-processing: {njob} jobs started on {ncpu} CPUs'.format(njob=len(cases), ncpu=ncpu))

        for idx, case_name in enumerate(cases):
            kwargs['pid'] = idx
            job = Process(name='Process {0:d}'.format(idx), target=run, args=(case_name,), kwargs=kwargs)
            jobs.append(job)
            job.start()
            print('Process {:d} <{:s}> started.'.format(idx, case_name))

            if (idx % ncpu == ncpu - 1) or (idx == len(cases) - 1):
                sleep(0.1)
                for job in jobs:
                    job.join()
                jobs = []

        t0, s0 = elapsed(t0)
        print('--> Multiple jobs finished in {0:s}.'.format(s0))
        return


def run(case, **kwargs):
    """Run a single case study"""
    profile = kwargs.pop('profile', False)
    dump_raw = kwargs.get('dump_raw', False)
    summary = kwargs.pop('summary', False)
    exitnow = kwargs.pop('exit', False)
    pid = kwargs.get('pid', -1)
    pr = cProfile.Profile()

    # enable profiler if requested
    if profile:
        pr.enable()

    # create a power system object
    system = PowerSystem(case, **kwargs)

    t0, _ = elapsed()

    # parse input file
    if not filters.guess(system):
        return

    if not filters.parse(system):
        return

    # dump system as raw file if requested
    if dump_raw:
        filters.dump_raw(system)

    # print summary only
    if summary:
        system.Report.write(content='summary')
        return

    # exit without solving power flow
    if exitnow:
        system.Log.info('Exiting before solving power flow.')
        return

    # set up everything in system
    system.setup()

    # initialize power flow study
    system.pf_init()

    powerflow.run(system)

    # initialize variables for output even if not running TDS
    system.td_init()

    system.Report.write(content='powerflow')

    # run more studies
    # t0, s = elapsed()
    routine = kwargs.pop('routine', None)
    if not routine:
        pass
    elif routine.lower() in ['time', 'td', 't']:
        routine = 'td'
    elif routine.lower() in ['cpf', 'c']:
        routine = 'cpf'
    elif routine.lower() in ['small', 'ss', 'sssa', 's']:
        routine = 'sssa'

    if routine is 'td':
        t1, s = elapsed(t0)
        # system.hack_EAGC()

        ret = timedomain.run(system)

        t2, s = elapsed(t1)
        if ret and (not system.Files.no_output):
            system.VarOut.dump()
            t3, s = elapsed(t2)
            system.Log.info('Simulation data dumped in {:s}.'.format(s))
    elif routine == 'sssa':
        t1, s = elapsed(t0)
        system.Log.info('')
        system.Log.info('Eigenvalue Analysis:')
        eigenanalysis.run(system)
        t2, s = elapsed(t1)
        system.Log.info('Analysis finished in {:s}.'.format(s))

    # Disable profiler and output results
    if profile:
        pr.disable()
        if system.Files.no_output:
            s = io.StringIO()
            nlines = 20
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(nlines)
            print(s.getvalue())
            s.close()
        else:
            s = open(system.Files.prof, 'w')
            nlines = 999
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(nlines)
            s.close()
            system.Log.info('cProfile results for job{:s} written.'.format(' ' + str(pid) if pid >= 0 else ''))

    if pid >= 0:
        t3, s = elapsed(t0)
        print('Process {:d} finished in {:s}.'.format(pid, s))


if __name__ == '__main__':
    main()
