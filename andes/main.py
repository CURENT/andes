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

import cProfile
import glob
import io
import logging
import os
import platform
import pprint
import pstats
import sys
from argparse import ArgumentParser
from multiprocessing import Process
from time import sleep, strftime

import pathlib
from andes import filters
from andes import routines
from andes.system import PowerSystem
from andes.utils import elapsed, get_config_load_path
from subprocess import call

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def config_logger(name='andes',
                  logfile='andes.log',
                  stream=True,
                  stream_level=logging.INFO
                  ):
    """
    Configure a logger for the andes package with options for a `FileHandler`
    and a `StreamHandler`. This function is called at the beginning of
    ``andes.main.main()``.

    Parameters
    ----------
    name : str, optional
        Base logger name, ``'andes'`` by default. Changing this
        parameter will affect the loggers in modules and
        cause unexpected behaviours.
    logfile : str, optional
        Logg file name for `FileHandler`, ``'andes.log'`` by default.
        If ``None``, the `FileHandler` will not be created.
    stream : bool, optional
        Create a `StreamHandler` for `stdout` if ``True``.
        If ``False``, the handler will not be created.
    stream_level : {10, 20, 30, 40, 50}, optional
        `StreamHandler` verbosity level.

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
    Log the Andes command-line preamble at the `logging.INFO` level

    Returns
    -------
    None
    """
    from andes import __version__ as version
    logger.info('ANDES {ver} (Build {b}, Python {p} on {os})'
                .format(ver=version[:5], b=version[-8:],
                        p=platform.python_version(),
                        os=platform.system()))

    logger.info('Session: ' + strftime("%m/%d/%Y %I:%M:%S %p"))
    logger.info('')


def cli_new():
    """
    Construct a CLI argument parser and return the parsed arguments.

    Returns
    -------
    Namespace
        A namespace object containing the parsed command-line arguments
    """
    parser = ArgumentParser()
    parser.add_argument('filename', help='Case file name', nargs='*')

    # general options
    general_group = parser.add_argument_group('General options')
    general_group.add_argument('-r', '--routine', choices=routines.__cli__,
                               help='Routine to run', nargs='*',
                               default=['pflow'], )
    general_group.add_argument('--edit-config', help='Quick edit of the config file',
                               action='store_true')
    general_group.add_argument('--license', action='store_true', help='Display software license')

    # I/O
    io_group = parser.add_argument_group('I/O options', 'Optional arguments for managing I/Os')
    io_group.add_argument('-p', '--path', help='Path to case files', type=str, default='')
    io_group.add_argument('-a', '--addfile', help='Additional files used by some formats.')
    io_group.add_argument('-D', '--dynfile', help='Additional dynamic file in dm format.')
    io_group.add_argument('-P', '--pert', help='Perturbation file path', default='')
    io_group.add_argument('-d', '--dump-raw', help='Dump RAW format case file.')
    io_group.add_argument('-n', '--no-output', help='Force no output of any '
                                                    'kind',
                          action='store_true')
    io_group.add_argument('-C', '--clean', help='Clean output files', action='store_true')

    config_exclusive = parser.add_mutually_exclusive_group()
    config_exclusive.add_argument('--load-config', help='path to the rc config to load',
                                  dest='config')
    config_exclusive.add_argument('--save-config', help='save configuration to file name',
                                  nargs='?', type=str, default='')

    # helps and documentations
    group_help = parser.add_argument_group('Help and documentation',
                                           'Optional arguments for usage, model and config documentation')
    group_help.add_argument(
        '-g', '--group', help='Show the models in the group.')
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


def andeshelp(group=None,
              category=None,
              model_list=None,
              model_format=None,
              model_var=None,
              quick_help=None,
              help_option=None,
              help_config=None,
              export='plain',
              **kwargs):
    """
    Print the requested help and documentation to stdout.

    Parameters
    ----------
    group : None or str
        Name of a group whose model names will be printed

    category : None or str
        Name of a category whose models will be printed

    model_list : bool
        If ``True``, print the full model list.

    model_format : None or str
        Names of models whose parameter definitions will be printed. Model
        names are separated by comma without space.

    model_var : None or str
        A pair of model name and parameter name separated by dot. For
        example, ``Bus.voltage`` stands for the ``voltage`` parameter of
        ``Bus``.

    quick_help : None or str
        Name of a model to print a quick help of parameter definitions.

    help_option : None or str
        A pair of config name and option name separated by dot. For example,
        ``System.sparselib`` stands for the ``sparselib`` option of
        the ``System`` config.

    help_config : None or str
        Config names whose option definitions will be printed. Configs are
        separated by comma without space. For example, ``System,Pflow``. In
        the naming convention, the first letter is captialized.

    export : None or {'plain', 'latex'}
        Formatting style available in plain text or LaTex format. This option
        has not been implemented.

    kwargs : None or dict
        Other keyword arguments

    Returns
    -------
    bool
        True if any help function is executed.

    Notes
    -----
    The outputs can be written to a text file using shell command,
    for example, ::

        andes -q Bus > bus_help.txt

    """
    out = []

    if not (group or category or model_list or model_format
            or model_var or quick_help or help_option or help_config):
        return False

    from andes.models import all_models_list

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
                    c1 = system.__dict__[dev]._descr.get(var, 'no Description')
                    c2 = system.__dict__[dev]._data.get(var)
                    c3 = system.__dict__[dev]._units.get(var, 'no Unit')
                    out.append('{}: {}, default = {:g} {}'.format(
                        '.'.join(model_var), c1, c2, c3))

    if group:
        group_dict = {}
        match = []

        for model in all_models_list:
            g = system.__dict__[model]._group
            if g not in group_dict:
                group_dict[g] = []
            group_dict[g].append(model)

        if group.lower() == 'all':
            match = sorted(list(group_dict.keys()))

        else:
            group = [group]
            # search for ``group`` in all group names and store in ``match``
            for item in group_dict.keys():
                if group[0].lower() in item.lower():
                    match.append(item)

        # if group name not found
        if len(match) == 0:
            out.append('Group <{:s}> not found.'.format(group[0]))

        for idx, item in enumerate(match):
            group_models = sorted(list(group_dict[item]))
            out.append('<{:s}>'.format(item))
            out.append(', '.join(group_models))
            out.append('')

    if quick_help:
        if quick_help not in all_models_list:
            out.append('Model <{}> does not exist.'.format(quick_help))
        else:
            out.append(system.__dict__[quick_help].doc(export=export))

    if help_option:
        raise NotImplementedError

    if help_config:

        all_config = ['system', 'pflow', 'tds', 'eig']

        if help_config.lower() == 'all':
            help_config = all_config

        else:
            help_config = help_config.split(',')

            for item in help_config:
                if item.lower() not in all_config:
                    logger.warning('Config <{}> does not exist.'.format(item))
                    help_config.remove(item)

        if len(help_config) > 0:
            for item in help_config:
                if item == 'system':
                    out.append(system.config.doc(export=export))
                else:
                    out.append(system.__dict__[item.lower()].config.doc(
                        export=export))

    print('\n'.join(out))  # NOQA

    return True


def edit_conf(edit_config=False, load_config=None, **kwargs):
    """
    Edit the Andes config file which occurs first in the search path.

    Parameters
    ----------
    edit_config : bool
        If ``True``, try to open up an editor and edit the config file.
        Otherwise returns.

    load_config : None or str, optional
        Path to the config file, which will be placed to the first in the
        search order.

    kwargs : dict
        Other keyword arguments.

    Returns
    -------
    bool
        ``True`` is a config file is found and an editor is opened. ``False``
        if ``edit_config`` is False.

    """
    ret = False

    if edit_config is False:
        return ret

    conf_path = get_config_load_path(load_config)

    if conf_path is not None:
        logger.info('Editing config file {}'.format(conf_path))

        editor = ''
        if platform.system() == 'Linux':
            editor = os.environ.get('EDITOR', 'gedit')
        elif platform.system() == 'Darwin':
            editor = os.environ.get('EDITOR', 'vim')
        elif platform.system() == 'Windows':
            editor = 'notepad.exe'

        call([editor, conf_path])

    else:
        logger.info('Config file does not exist. Save config with \'andes '
                    '--save-config\'')
        ret = True

    return ret


def remove_output(clean=False, **kwargs):
    """
    Remove the outputs generated by Andes, including power flow reports
    ``_out.txt``, time-domain list ``_out.lst`` and data ``_out.dat``,
    eigenvalue analysis report ``_eig.txt``.

    Parameters
    ----------
    clean : bool
        If ``True``, execute the function body. Returns otherwise.

    kwargs : dict
        Other keyword arguments

    Returns
    -------
    bool
        ``True`` is the function body executes with success. ``False``
        otherwise.
    """
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
                logger.info('<{:s}> removed.'.format(file))
            except IOError:
                logger.error('Error removing file <{:s}>.'.format(file))
    if not found:
        logger.info('no output found in the working directory.')

    return True


def search(search, **kwargs):
    """
    Search for models whose names matches the given pattern. Print the
    results to stdout.

    .. deprecated :: 1.0.0
        `search` will be moved to ``andeshelp`` in future versions.

    Parameters
    ----------
    search : str
        Partial or full name of the model to search for

    kwargs : dict
        Other keyword arguments.

    Returns
    -------
    list
        The list of model names that match the given pattern.
    """

    from andes.models import all_models
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
        print('Search result: <file.model> containing <{}>'
              .format(search))
        print(' '.join(out))
    else:
        print('No model containing <{:s}> found'.format(search))

    return out


def save_config(save_config='', **kwargs):
    """
    Save the Andes config to a file at the path specified by ``save_config``.
    The save action will not run if `save_config = ''`.

    Parameters
    ----------
    save_config : None or str, optional, ('' by default)
        Path to the file to save the config file. If the path is an emtpy
        string, the save action will not run. Save to
        `~/.andes/andes.conf` if ``None``.

    kwargs : dict, optional
        Other keyword arguments

    Returns
    -------
    bool
        ``True`` is the save action is run. ``False`` otherwise.
    """
    ret = False
    cf_path = save_config

    # no ``--save-config ``
    if cf_path == '':
        return ret

    if cf_path is None:
        cf_path = 'andes.conf'
        home = str(pathlib.Path.home())

        path = os.path.join(home, '.andes')
        if not os.path.exists(path):
            os.makedirs(path)

        cf_path = os.path.join(path, cf_path)

    ps = PowerSystem()
    ps.dump_config(cf_path)
    ret = True

    return ret


def main():
    """
    The main function of the Andes command-line tool.

    This function executes the following workflow:

     * Parse the command line inputs
     * Show the tool preamble
     * Output the requested helps, edit/save configs or remove outputs. Exit
       the main program if any of the above is executed
     * Process the input files and call ``main.run()`` using single- or
       multi-processing
     * Show the execution time and exit

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

    # show preamble
    preamble()

    if andeshelp(**args) or search(**args) or edit_conf(**args) or remove_output(**args) \
            or save_config(**args):
        return

    # process input files
    if len(args['filename']) == 0:
        logger.info('error: no input file. Try \'andes -h\' for help.')

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
            logger.info('error: file {} does not exist.'.format(full_paths))
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
    elif len(valid_cases) >= 2:
        logger.info('-> Multiple processes finished in {:s}.'.format(s0))

    return


def run(case, profile=False, dump_raw=False, routine=('pflow',), pid=-1,
        **kwargs):
    """
    Entry function to run a single case study. This function executes the
    following workflow:

     * Turn on cProfile if requested
     * Populate a ``PowerSystem`` object
     * Parse the input files using filters
     * Dump the case file is requested
     * Set up the system
     * Run the specified routine(s)

    Parameters
    ----------
    case : str
        Path to the case file

    profile : bool, optional
        Enable cProfile if ``True``.

    dump_raw : ``False`` or str, optional
        Path to the file to dump the system parameters in the dm format

    routine : Iterable, optional
        A list of routines to run in sequence (``('pflow')`` as default)

    pid : idx, optional
        Process idx of the current run

    kwargs : dict, optional
        Other keyword arguments

    Returns
    -------
    PowerSystem
        Andes PowerSystem object

    """
    t0, _ = elapsed()

    # enable profiler if requested
    pr = cProfile.Profile()
    if profile is True:
        pr.enable()

    system = PowerSystem(case, **kwargs)

    if not filters.guess(system):
        return

    if not filters.parse(system):
        return

    # dump system as raw file if requested
    if dump_raw:
        filters.dump_raw(system)

    system.setup()

    # run power flow study by default
    if 'pflow' in routine:
        routine.remove('pflow')

    system.pflow.run()
    system.tds.init()
    system.report.write(content='powerflow')

    for r in routine:
        system.__dict__[r.lower()].run()

    # Disable profiler and output results
    if profile:
        pr.disable()

        if system.files.no_output:
            s = io.StringIO()
            nlines = 20
            ps = pstats.Stats(pr, stream=sys.stdout).sort_stats('cumtime')
            ps.print_stats(nlines)
            logger.info(s.getvalue())
            s.close()
        else:
            s = open(system.files.prof, 'w')
            nlines = 999
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(nlines)
            s.close()
            logger.info('cProfile results for job{:s} written.'.format(
                ' ' + str(pid) if pid >= 0 else ''))

    if pid >= 0:
        t3, s = elapsed(t0)
        msg_finish = 'Process {:d} finished in {:s}.'.format(pid, s)
        logger.info(msg_finish)
        print(msg_finish)

    return system


if __name__ == '__main__':
    main()
