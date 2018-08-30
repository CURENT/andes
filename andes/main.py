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
import platform
from time import sleep, strftime
import pprint

from multiprocessing import Process
from argparse import ArgumentParser

from . import filters
from .consts import ERROR
from .system import PowerSystem
from .utils import elapsed
from .routines import timedomain, eigenanalysis
from . import routines

# from .routines.fakemodule import EAGC

import logging
logger = None


def config_logger(name='andes', logfile='andes.log', stream=True, stream_level=logging.INFO):
    """
    Configure a logger for the andes package

    Parameters
    ----------
    name: str
        base logger name, ``andes`` by default
    logfile
        logger file name

    Returns
    -------
    None

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # logging formatter
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh_formatter = logging.Formatter('%(message)s')

    # file handler which logs debug messages
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # stream handler for errors
    if stream is True:
        sh = logging.StreamHandler()
        sh.setLevel(stream_level)
        sh.setFormatter(sh_formatter)
        logger.addHandler(sh)

    globals()['logger'] = logger


def preamble():
    """
    Log Andes preamble at the ``INFO`` level

    Returns
    -------
    None
    """
    from . import __version__ as version
    logger.info('ANDES {ver} (Build {b}, Python {p} on {os})'.format(ver=version[:5], b=version[-8:],
                                                                     p=platform.python_version(),
                                                                     os=platform.system()))

    logger.info('Session:     ' + strftime("%m/%d/%Y %I:%M:%S %p"))
    logger.info('')


def cli_new(help=False):
    """
    Command line argument parser
    Parameters
    ----------
    help: bool
        Return argparse help in a string

    Returns
    -------
    None (str)

    """
    parser = ArgumentParser()
    parser.add_argument('filename', help='Case file name', nargs='*')

    # general options
    general_group = parser.add_argument_group('General options')
    general_group.add_argument('-r', '--run', choices=routines.__cli__, help='Routine to run', nargs='?',
                        default='info')
    general_group.add_argument('--conf', choices=routines.__cli__ + ['system'],
                               help='Edit configuration', default='system')
    general_group.add_argument('--license', action='store_true', help='Display software license')

    # I/O
    io_group = parser.add_argument_group('I/O options', 'Optional arguments for managing I/Os')
    io_group.add_argument('-p', '--path', help='Path to case files', type=str, default='')
    io_group.add_argument('-a', '--addfile', help='Additional files used by some formats.')
    io_group.add_argument('-D', '--dynfile', help='Additional dynamic file in dm format.')
    io_group.add_argument('-P', '--pert', help='Perturbation file path', default='')
    io_group.add_argument('-d', '--dump-raw', help='Dump RAW format case file.')
    io_group.add_argument('-n', '--no_output', help='Force no output of any kind',
                          action='store_true')
    io_group.add_argument('-C', '--clean', help='Clean output files', action='store_true')

    # helps and documentations
    group_help = parser.add_argument_group('Help and documentation',
                                           'Optional arguments for usage, model and config documentation')
    group_help.add_argument(
        '-G', '--group', help='Show the models in the group.')
    group_help.add_argument(
        '-q', '--quick-help', help='Show a quick help of model format.')
    group_help.add_argument(
        '-c',
        '--category',
        help='Show model names in the given category.')
    group_help.add_argument(
        '-l',
        '--model-list',
        help='Show a full list of all models.',
        action='store_true')
    group_help.add_argument(
        '-f',
        '--model-format',
        help='Show the format definition of models.', nargs='*', default=[])
    group_help.add_argument(
        '-Q',
        '--model-var',
        help='Show the definition of variables <MODEL.VAR>.')
    group_help.add_argument(
        '--config-option', help='Show a quick help of a config option <CONFIG.OPTION>')
    group_help.add_argument(
        '--help-config',
        help='Show help of the <CONFIG> class. Use ALL for all configs.')
    group_help.add_argument(
        '-s',
        '--search',
        help='Search for models that match the pattern.')

    # simulation control
    sim_options = parser.add_argument_group('Simulation control options',
                                            'Overwrites the simulation configs')
    sim_options.add_argument(
        '--dime', help='Specify DiME streaming server address and port')
    sim_options.add_argument(
        '--tf', help='End time of time-domain simulation', type=float)

    # developer options
    dev_group = parser.add_argument_group('Developer options', 'Options for developer debugging')
    dev_group.add_argument(
        '-v',
        '--verbose',
        help='Program logging level.'
        'Available levels are 10-DEBUG, 20-INFO, 30-WARNING, '
        '40-ERROR or 50-CRITICAL. The default level is 20-INFO',
        type=int, default=20, choices=(10, 20, 30, 40, 50))
    dev_group.add_argument(
        '--profile', action='store_true', help='Enable Python cProfiler')
    dev_group.add_argument(
        '--ncpu', help='Number of parallel processes', type=int, default=0)

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

    if not (usage or group or category or model_list or model_format
            or model_var or quick_help or help_option or help_settings):
        return False

    from .models import all_models_list

    if category:
        raise NotImplementedError

    if model_list:
        raise NotImplementedError

    system = PowerSystem()

    if model_format:
        if model_format.lower() == 'all':
            model_format = all_models_list
        else:
            model_format = model_format.split(',')

        for item in model_format:
            if item not in all_models_list:
                logger.warning('Model <{}> does not exist.'.format(item))
                model_format.remove(item)

        if len(model_format) > 0:
            for item in model_format:
                out.append(system.__dict__[item].doc(export=export))

    if model_var:
        model_var = model_var.split('.')

        if len(model_var) == 1:
            logger.error('Model and parameter not separated by dot.')

        elif len(model_var) > 2:
            logger.error('Model parameter not specified correctly.')

        else:
            dev, var = model_var
            if not hasattr(system, dev):
                logger.error('Model <{}> does not exist.'.format(dev))
            else:
                if var not in system.__dict__[dev]._data.keys():
                    logger.error(
                        'Model <{}> does not have parameter <{}>.'.format(
                            dev, var))
                else:
                    c1 = system.__dict__[dev]._descr.get(var, 'No Description')
                    c2 = system.__dict__[dev]._data.get(var)
                    c3 = system.__dict__[dev]._units.get(var, 'No Unit')
                    out.append('{}: {}, default = {:g} {}'.format(
                        '.'.join(model_var), c1, c2, c3))

    if group:
        group_dict = {}

        for model in all_models_list:
            g = system.__dict__[model]._group
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
            out.append(system.__dict__[quick_help].doc(export=export))

    if help_option:
        raise NotImplementedError

    if help_settings:
        all_settings = ['Config', 'SPF', 'TDS', 'SSSA', 'CPF']

        if help_settings.lower() == 'all':
            help_settings = all_settings

        else:
            help_settings = help_settings.split(',')

            for item in help_settings:
                if item not in all_settings:
                    logger.warning('Setting <{}> does not exist.'.format(item))
                    help_settings.remove(item)

        if len(help_settings) > 0:
            for item in help_settings:
                out.append(system.__dict__[item].doc(export=export))

    logger.info('\n'.join(out))  # NOQA

    return True


def edit_conf(conf, **kwargs):
    """
    Edit Andes routine configuration

    :param conf: name of the routine
    :return: succeed flag
    """
    # raise NotImplementedError("Not implemented")
    return False


def remove_output(clean=False, **kwargs):
    """Clean up function for generated files"""
    if not clean:
        return False

    found = False
    cwd = os.getcwd()

    for file in os.listdir(cwd):
        if file.endswith('_eig.txt') or \
                file.endswith('_out.txt') or \
                file.endswith('_out.lst') or \
                file.endswith('_out.dat') or \
                file.endswith('_prof.txt'):
            found = True
            try:
                os.remove(file)
                print('<{:s}> removed.'.format(file))
            except IOError:
                print('Error removing file <{:s}>.'.format(file))
    if not found:
        print('--> No Andes output found in the working directory.')

    return True


def search(search, **kwargs):
    """
    TODO: merge to ``andeshelp``
    Search for models whose names contain ``keyword``

    :param str search: partial or full model name
    :return: a list of model names in <file.model> format
    """

    from .models import all_models
    out = []

    if not search:
        return out

    keys = sorted(list(all_models.keys()))

    for key in keys:
        vals = all_models[key]
        val = list(vals.keys())
        val = sorted(val)

        for item in val:
            if search.lower() in item.lower():
                out.append(key + '.' + item)

    if out:
        print('Search result: <file.model> containing <{}>'.format(search))
        print(' '.join(out))
    else:
        print('No model containing <{:s}> found'.format(search))

    return out


def main():
    """
    The new main function
    Returns
    -------
    None
    """
    t0, s = elapsed()

    # parser command line arguments
    args = vars(cli_new())

    # configure stream handler verbose level
    config_logger(stream_level=args['verbose'])

    logger.debug('command line arguments:')
    logger.debug(pprint.pformat(args))

    if andeshelp(**args) or search(**args) or edit_conf(**args) or remove_output(**args):
        return

    # show preamble
    preamble()

    # process input files
    if len(args['filename']) == 0:
        logger.info('-> No input file. Try \'andes -h\' for help.')

    # preprocess cli args
    path = args.get('path', os.getcwd())
    ncpu = args['ncpu']
    if ncpu == 0 or ncpu > os.cpu_count():
        ncpu = os.cpu_count()

    cases = []

    for file in args['filename']:
        full_paths = os.path.join(path, file)
        found = glob.glob(full_paths)
        if len(found) == 0:
            logger.info('-> File {} does not exist.'.format(full_paths))
        else:
            cases += found

    # remove folders and make cases unique
    cases = list(set(cases))
    valid_cases = []
    for case in cases:
        if os.path.isfile(case):
            valid_cases.append(case)

    logger.debug('Found files: ' + pprint.pformat(valid_cases))

    if len(valid_cases) <= 0:
        pass
    elif len(valid_cases) == 1:
        run(valid_cases[0], **args)
    else:
        # set verbose level for multi processing
        logger.info('Processing {} jobs on {} CPUs'.format(len(valid_cases), ncpu))
        logger.handlers[1].setLevel(logging.WARNING)

        # start processes
        jobs = []
        for idx, file in enumerate(valid_cases):
            args['pid'] = idx
            job = Process(
                name='Process {0:d}'.format(idx),
                target=run,
                args=(file, ),
                kwargs=args)
            jobs.append(job)
            job.start()

            start_msg = 'Process {:d} <{:s}> started.'.format(idx, file)
            print(start_msg)
            logger.debug(start_msg)

            if (idx % ncpu == ncpu - 1) or (idx == len(valid_cases) - 1):
                sleep(0.1)
                for job in jobs:
                    job.join()
                jobs = []

        # restore command line output when all jobs are done
        logger.handlers[1].setLevel(logging.INFO)

    t0, s0 = elapsed(t0)

    if len(valid_cases) == 1:
        logger.info('-> Single process finished in {:s}.'.format(s0))
    elif len(valid_cases) >=2:
        logger.info('-> Multiple processes finished in {:s}.'.format(s0))

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
        system.log.info('Exiting before solving power flow.')
        return

    # set up everything in system
    system.setup()

    # run power flow study
    system.powerflow.run()

    # initialize variables for output even if not running TDS
    system.td_init()

    system.Report.write(content='powerflow')

    # run more studies
    # t0, s = elapsed()
    routine = kwargs.pop('run', None)
    if not routine:
        pass
    elif routine.lower() in ['time', 'tds', 't']:
        routine = 'tds'
    elif routine.lower() in ['cpf', 'c']:
        routine = 'cpf'
    elif routine.lower() in ['small', 'ss', 'sssa', 's']:
        routine = 'sssa'

    if routine is 'tds':
        t1, s = elapsed(t0)
        # system.hack_EAGC()

        ret = timedomain.run(system)

        t2, s = elapsed(t1)
        if ret and (not system.Files.no_output):
            system.VarOut.dump()
            t3, s = elapsed(t2)
            system.log.info('Simulation data dumped in {:s}.'.format(s))
    elif routine == 'sssa':
        t1, s = elapsed(t0)
        system.log.info('')
        system.log.info('Eigenvalue Analysis:')
        eigenanalysis.run(system)
        t2, s = elapsed(t1)
        system.log.info('Analysis finished in {:s}.'.format(s))

    # Disable profiler and output results
    if profile:
        pr.disable()
        if system.Files.no_output:
            s = io.StringIO()
            nlines = 20
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(nlines)
            logging.info(s.getvalue())
            s.close()
        else:
            s = open(system.Files.prof, 'w')
            nlines = 999
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(nlines)
            s.close()
            system.log.info('cProfile results for job{:s} written.'.format(
                ' ' + str(pid) if pid >= 0 else ''))

    if pid >= 0:
        t3, s = elapsed(t0)
        logging.info('Process {:d} finished in {:s}.'.format(pid, s))


if __name__ == '__main__':
    main()
