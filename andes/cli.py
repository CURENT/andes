"""
ANDES command-line interface and argument parsers.
"""

import argparse
import importlib
import logging
import platform
import sys
from time import strftime

from andes.shared import NCPUS_PHYSICAL

from andes.main import config_logger, find_log_path
from andes.routines import routine_cli
from andes.utils.paths import get_log_dir

logger = logging.getLogger(__name__)

command_aliases = {
    'prepare': ['prep'],
    'selftest': ['st'],
}


def create_parser():
    """
    Create a parser for the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all ANDES options
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v', '--verbose',
        help='Verbosity level in 10-DEBUG, 20-INFO, 30-WARNING, '
             'or 40-ERROR.',
        type=int, default=20, choices=(1, 10, 20, 30, 40))

    sub_parsers = parser.add_subparsers(dest='command', help='[run] run simulation routine; '
                                                             '[plot] plot results; '
                                                             '[doc] quick documentation; '
                                                             '[misc] misc. functions; '
                                                             '[prepare] prepare the numerical code; '
                                                             '[selftest] run self test; '
                                        )

    run = sub_parsers.add_parser('run')
    run.add_argument('filename', help='Case file name. Power flow is calculated by default.', nargs='*')
    run.add_argument('-r', '--routine', nargs='*', default=('pflow', ),
                     action='store', help='Simulation routine(s). Single routine or multiple separated with '
                                          'space. Run PFlow by default.',
                     choices=list(routine_cli.keys()))
    run.add_argument('-p', '--input-path', help='Path to case files', type=str, default='')
    run.add_argument('-a', '--addfile', help='Additional files used by some formats.')
    run.add_argument('-P', '--pert', help='Perturbation file path', default='')
    run.add_argument('-o', '--output-path', help='Output path prefix', type=str, default='')
    run.add_argument('-n', '--no-output', help='Force no output of any kind', action='store_true')
    run.add_argument('--ncpu', help='Number of parallel processes', type=int, default=NCPUS_PHYSICAL)
    run.add_argument('--dime-address', help='DiME streaming server protocol, address and port,'
                                            'e.g., tcp://127.0.0.1:5000 or ipc:///tmp/dime2', type=str)
    run.add_argument('--tf', help='End time of time-domain simulation', type=float)
    run.add_argument('--qrt', help='Enable quasi-real-time stepping', action='store_true')
    run.add_argument('--kqrt', help='Scaling factor for quasi-real-time; e.g., kqrt=2 means the wall-clock time '
                                    'to complete a simulation is 2x the time of that simulation', type=float)
    run.add_argument('-c', '--convert', help='Convert to format.', type=str, default='', nargs='?')
    run.add_argument('-b', '--add-book', help='Add a template workbook for the specified model.', type=str)
    run.add_argument('--convert-all', help='Convert to format with all templates.', type=str, default='',
                     nargs='?')
    run.add_argument('--state-matrix', help='Export state matrix to a .mat file. Need to run with `-r eig`',
                     action='store_true')
    run.add_argument('--profile', action='store_true', help='Enable Python cProfiler')
    run.add_argument('-s', '--shell', action='store_true', help='Start in IPython shell')
    run.add_argument('--no-preamble', action='store_true', help='Hide preamble')
    run.add_argument('--no-pbar', action='store_true', help='Hide progress bar for time-domain')
    run.add_argument('--flat', action='store_true', help='Run no-disturbance (flat) simulation')
    run.add_argument('--pool', action='store_true', help='Start multiprocess with Pool '
                                                         'and return a list of Systems')
    run.add_argument('--from-csv', help='Use data from a CSV file instead of from simulation')
    run.add_argument('-O', '--config-option',
                     help='Set configuration option specificied by '
                          'NAME.FIELD=VALUE with no space. For example, "TDS.tf=2"',
                     type=str, default='', nargs='*')
    run.add_argument('--init', help='Initialize variables only for time-domain simulation without running',
                     action='store_true')

    plot = sub_parsers.add_parser('plot')
    plot.add_argument('filename', nargs=1, default=[], help='simulation output file name, which should end '
                                                            'with `out`. File extension can be omitted.')
    plot.add_argument('x', nargs='?', type=int, help='the X-axis variable index, typically 0 for Time',
                      default='0')
    plot.add_argument('y', nargs='*', help='Y-axis variable indices. Space-separated indices or a '
                                           'colon-separated range is accepted')
    plot.add_argument('--xmin', type=float, help='minimum value for X axis', dest='left')
    plot.add_argument('--xmax', type=float, help='maximum value for X axis', dest='right')
    plot.add_argument('--ymax', type=float, help='maximum value for Y axis')
    plot.add_argument('--ymin', type=float, help='minimum value for Y axis')
    find_or_xargs = plot.add_mutually_exclusive_group()
    find_or_xargs.add_argument('--find', type=str, help='find variable indices that matches the given pattern')
    find_or_xargs.add_argument('-a', '--xargs', type=str,
                               help='find variable indices and return as a list of arguments '
                                    'usable with "|xargs andes plot"')
    plot.add_argument('--exclude', type=str, help='pattern to exclude in find or xargs results')
    plot.add_argument('-x', '--xlabel', type=str, help='x-axis label text')
    plot.add_argument('-y', '--ylabel', type=str, help='y-axis label text')
    plot.add_argument('-s', '--savefig', action='store_true', help='save figure. The default format is `png`')
    plot.add_argument('-f', '--format', dest='save_format',
                      help='format for savefig. Common formats such as png, pdf, jpg are supported')
    plot.add_argument('--dpi', type=int, help='image resolution in dot per inch (DPI)')
    plot.add_argument('--font-size', type=float, help='Text font size', default=12)
    plot.add_argument('--line-width', type=float, help='Plot line width', default=1.5)
    plot.add_argument('-g', '--grid', action='store_true', help='grid on')
    plot.add_argument('-t', '--title', help='title text')
    plot.add_argument('--greyscale', action='store_true', help='greyscale on')
    plot.add_argument('-d', '--no-latex', action='store_false', dest='latex', help='disable LaTex formatting')
    plot.add_argument('-n', '--no-show', action='store_false', dest='show', help='do not show the plot window')
    plot.add_argument('--ytimes', type=str, help='scale the y-axis values by YTIMES')
    plot.add_argument('-c', '--to-csv', help='convert npy output to csv', action='store_true')
    plot.add_argument('--hline1', help='dashed horizontal line 1', type=float)
    plot.add_argument('--hline2', help='dashed horizontal line 2', type=float)
    plot.add_argument('--vline1', help='dashed vertical line 1', type=float)
    plot.add_argument('--vline2', help='dashed vertical line 2', type=float)

    doc = sub_parsers.add_parser('doc')
    doc.add_argument('attribute', help='System attribute name to get documentation', nargs='?')
    doc.add_argument('--config', '-c', help='Config help')
    doc.add_argument('--list', '-l', help='List supported models and groups', action='store_true',
                     dest='list_supported')
    doc.add_argument('--init-seq', help='Show model initialization sequence', action='store_true',
                     )

    misc = sub_parsers.add_parser('misc')
    config_exclusive = misc.add_mutually_exclusive_group()
    config_exclusive.add_argument('--edit-config', help='Quick edit of the config file',
                                  default='', nargs='?', type=str)
    config_exclusive.add_argument('--save-config', help='save configuration to file name',
                                  nargs='?', type=str, default='')
    misc.add_argument('--license', action='store_true', help='Display software license', dest='show_license')
    misc.add_argument('-C', '--clean', help='Clean output files', action='store_true')
    misc.add_argument('-r', '--recursive', help='Recursively clean outputs (combined useage with --clean)',
                      action='store_true')
    misc.add_argument('-O', '--config-option',
                      help='Set configuration option specificied by '
                      'NAME.FIELD=VALUE with no space. For example, "TDS.tf=2"',
                      type=str, default='', nargs='*')
    misc.add_argument('--version', action='store_true', help='Display version information')

    prep = sub_parsers.add_parser('prepare', aliases=command_aliases['prepare'])
    prep_mode = prep.add_mutually_exclusive_group()
    prep_mode.add_argument('-q', '--quick', action='store_true',
                           help='quick codegen by skipping pretty prints')
    prep_mode.add_argument('-f', '--full', action='store_true', help='full codegen')
    prep_mode.add_argument('-i', '--incremental', action='store_true',
                           help='rapid incrementally generate for updated models')
    prep.add_argument('-c', '--compile', help='compile the code with numba after codegen',
                      action='store_true', dest='precompile')
    prep.add_argument('--pycode-path', help='Save path for generated pycode')
    prep.add_argument('-m', '--models', nargs='*', help='model names to be individually prepared',
                      )
    prep.add_argument('--ncpu', help='Number of parallel processes', type=int, default=NCPUS_PHYSICAL)
    prep.add_argument('--nomp', help='Disable multiprocessing', action='store_true',)
    prep.add_argument('--incubate', help='Save generated pycode under the ANDES code directory to avoid codegen',
                      action='store_true')

    selftest = sub_parsers.add_parser('selftest', aliases=command_aliases['selftest'])
    quick_or_extra = selftest.add_mutually_exclusive_group()
    quick_or_extra.add_argument('-q', '--quick', action='store_true',
                                help='quick selftest by skipping codegen')
    quick_or_extra.add_argument('-e', '--extra', action='store_true',
                                help='run all standard tests plus the extra')

    demo = sub_parsers.add_parser('demo')  # NOQA

    return parser


def preamble():
    """
    Log the ANDES command-line preamble at the `logging.INFO` level
    """
    from andes import __version__ as version

    py_version = platform.python_version()
    system_name = platform.system()
    date_time = strftime('%m/%d/%Y %I:%M:%S %p')
    logger.info("\n"
                rf"    _           _         | Version {version}" + '\n'
                rf"   /_\  _ _  __| |___ ___ | Python {py_version} on {system_name}, {date_time}" + '\n'
                r"  / _ \| ' \/ _` / -_|_-< | " + "\n"
                r' /_/ \_\_||_\__,_\___/__/ | This program comes with ABSOLUTELY NO WARRANTY.' + '\n')

    log_path = find_log_path(logging.getLogger("andes"))

    if len(log_path):
        logger.debug('Logging to file "%s"', log_path[0])


def main():
    """
    Entry point of the ANDES command-line interface.
    """

    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    config_logger(stream=True,
                  stream_level=args.verbose,
                  file=True,
                  log_path=get_log_dir(),
                  )
    logger.debug(args)

    module = importlib.import_module('andes.main')

    if args.command in ('plot', 'doc', 'misc'):
        pass
    elif args.command == 'run' and args.no_preamble is True:
        pass
    else:
        preamble()

    # Run the command
    if args.command is None:
        parser.parse_args(sys.argv.append('--help'))

    else:
        cmd = args.command
        for fullcmd, aliases in command_aliases.items():
            if cmd in aliases:
                cmd = fullcmd

        func = getattr(module, cmd)
        return func(cli=True, **vars(args))
