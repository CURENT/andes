import argparse
import sys
import os
import logging
import platform
import importlib

from time import strftime
from andes.main import config_logger
from andes.utils.paths import get_log_dir

logger = logging.getLogger(__name__)


def create_parser():
    """
    The main level of command-line interface.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v', '--verbose',
        help='Program logging level. '
             'Available levels are 10-DEBUG, 20-INFO, 30-WARNING, '
             '40-ERROR or 50-CRITICAL. The default level is 20-INFO.',
        type=int, default=20, choices=(10, 20, 30, 40, 50))

    sub_parsers = parser.add_subparsers(dest='command', help='[run]: run simulation routine; '
                                                             '[plot]: plot simulation results; '
                                                             '[doc]: quick documentation;'
                                                             '[prepare]: run the symbolic-to-numeric preparation; '
                                                             '[misc]: miscellaneous functions.'
                                        )

    run = sub_parsers.add_parser('run')
    run.add_argument('filename', help='Case file name. Power flow is calculated by default.', nargs='*')
    run.add_argument('-r', '--routine',
                     action='store', help='Simulation routine to run.',
                     choices=('tds', 'eig'))
    run.add_argument('-p', '--input-path', help='Path to case files', type=str, default='')
    run.add_argument('-a', '--addfile', help='Additional files used by some formats.')
    run.add_argument('-D', '--dynfile', help='Additional dynamic file in dm format.')
    run.add_argument('-P', '--pert', help='Perturbation file path', default='')
    run.add_argument('-o', '--output-path', help='Output path prefix', type=str, default='')
    run.add_argument('-n', '--no-output', help='Force no output of any kind', action='store_true')
    run.add_argument('--ncpu', help='Number of parallel processes', type=int, default=os.cpu_count())
    run.add_argument('--dime', help='Specify DiME streaming server address and port', type=str)
    run.add_argument('--tf', help='End time of time-domain simulation', type=float)
    run.add_argument('--convert', help='Convert to format.', type=str, default='', nargs='?')
    run.add_argument('--convert-all', help='Convert to format with all templates.', type=str, default='',
                     nargs='?')
    run.add_argument('--add-book', help='Add a template workbook for the specified model.', type=str)
    run.add_argument('--profile', action='store_true', help='Enable Python cProfiler')

    plot = sub_parsers.add_parser('plot')
    plot.add_argument('filename', nargs=1, default=[], help='data file name.')
    plot.add_argument('x', nargs='?', type=int, help='x axis variable index', default='0')
    plot.add_argument('y', nargs='*', help='y axis variable indices. Space separated or ranges accepted')
    plot.add_argument('--xmin', type=float, help='x axis minimum value', dest='left')
    plot.add_argument('--xmax', type=float, help='x axis maximum value', dest='right')
    plot.add_argument('--ymax', type=float, help='y axis maximum value')
    plot.add_argument('--ymin', type=float, help='y axis minimum value')
    plot.add_argument('--find', type=str, help='find variable indices that matches the given pattern')
    plot.add_argument('--exclude ', type=str, help='exclude pattern in find')
    plot.add_argument('-x', '--xlabel', type=str, help='manual x-axis text label')
    plot.add_argument('-y', '--ylabel', type=str, help='y-axis text label')
    plot.add_argument('-s', '--savefig', action='store_true', help='save figure to file')
    plot.add_argument('-g', '--grid', action='store_true', help='grid on')
    plot.add_argument('-d', '--no-latex', action='store_false', dest='latex', help='disable LaTex formatting')
    plot.add_argument('-n', '--no-show', action='store_false', dest='show', help='do not show the plot window')
    plot.add_argument('--ytimes', type=str, help='y switch_times')
    plot.add_argument('--dpi', type=int, help='image resolution in dot per inch (DPI)')
    plot.add_argument('-c', '--tocsv', help='convert .npy output to a csv file', action='store_true')

    misc = sub_parsers.add_parser('misc')
    config_exclusive = misc.add_mutually_exclusive_group()
    config_exclusive.add_argument('--edit-config', help='Quick edit of the config file',
                                  default='', nargs='?', type=str)
    config_exclusive.add_argument('--save-config', help='save configuration to file name',
                                  nargs='?', type=str, default='')
    misc.add_argument('--license', action='store_true', help='Display software license', dest='show_license')
    misc.add_argument('-C', '--clean', help='Clean output files', action='store_true')

    prep = sub_parsers.add_parser('prepare')  # NOQA
    prep.add_argument('-q', '--quick', action='store_true', help='quick processing by skipping pretty prints')

    doc = sub_parsers.add_parser('doc')  # NOQA
    doc.add_argument('attribute', help='System attribute name to get documentation', nargs='?')
    doc.add_argument('--config', '-c', help='Config help')
    doc.add_argument('--list', '-l', help='List supported models and groups', action='store_true',
                     dest='list_supported')

    selftest = sub_parsers.add_parser('selftest')  # NOQA

    return parser


def preamble():
    """
    Log the ANDES command-line preamble at the `logging.INFO` level
    """
    from andes import __version__ as version
    logger.info('ANDES {ver} (Git commit id {b}, Python {p} on {os})'
                .format(ver=version[:5], b=version[-8:],
                        p=platform.python_version(),
                        os=platform.system()))
    try:
        username = os.getlogin() + ', '
    except OSError:
        username = ''

    logger.info('Session: {}{}'.format(username, strftime("%m/%d/%Y %I:%M:%S %p")))
    logger.info('This program comes with ABSOLUTELY NO WARRANTY.')
    logger.info('')


def main():
    """Main command-line interface"""
    parser = create_parser()
    args = parser.parse_args()

    config_logger(log_path=get_log_dir(), file=True, stream=True,
                  stream_level=args.verbose)
    preamble()
    logger.debug(args)

    module = importlib.import_module('andes.main')

    if args.command is None:
        parser.parse_args(sys.argv.append('--help'))

    else:
        func = getattr(module, args.command)
        func(**vars(args))
