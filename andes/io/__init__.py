import importlib
import logging
import os

from andes.utils.misc import elapsed
from andes.io import xlsx, psse   # NOQA


logger = logging.getLogger(__name__)

# Input formats is a dictionary of supported format names and the accepted file extensions
# The first file will be parsed by read() function and the addfile will be parsed by read_add()
# Typically, column based formats, such as IEEE CDF and PSS/E RAW, are faster to parse

input_formats = {
    'xlsx': ('xlsx',),
    'matpower': ('m', ),
    'psse': ('raw', 'dyr'),
}

# Output formats is a dictionary of supported output formats and their extensions
# The static data will be written by write() function and the addfile by writeadd()

output_formats = ['']


def guess(system):
    """
    input format guess function. First guess by extension, then test by lines
    """
    files = system.files
    maybe = []
    if files.input_format:
        maybe.append(files.input_format)
    # first, guess by extension
    for key, val in input_formats.items():
        if files.ext.strip('.').lower() in val:
            maybe.append(key)

    # second, guess by lines
    true_format = ''
    with open(files.case, 'r') as fid:
        for item in maybe:
            parser = importlib.import_module('.' + item, __name__)
            testlines = getattr(parser, 'testlines')
            if testlines(fid):
                true_format = item
                files.input_format = true_format
                logger.debug(f'Input format guessed as {true_format}.')
                break

    if not true_format:
        logger.error('Unable to determine case format.')

    # guess addfile format
    if files.addfile:
        _, add_ext = os.path.splitext(files.addfile)
        for key, val in input_formats.items():
            if add_ext[1:] in val:
                files.add_format = key
                logger.debug(f'Addfile format guessed as {key}.')
                break

    return true_format


def parse(system):
    """
    Parse input file with the given format in system.files.input_format
    """

    t, _ = elapsed()

    # exit when no input format is given
    if not system.files.input_format:
        if not guess(system):
            logger.error('Input format is not specified and cannot be inferred.')
            return False

    input_format = system.files.input_format
    add_format = system.files.add_format

    # exit if the format parser could not be imported
    try:
        parser = importlib.import_module('.' + input_format, __name__)
        if add_format:
            addparser = importlib.import_module('.' + add_format, __name__)
    except ImportError:
        logger.error(f'Parser for {input_format} format not found. Program will exit.')
        return False

    # try parsing the base case file
    logger.info('Parsing input file <{:s}>'.format(system.files.case))

    if not parser.read(system, system.files.case):
        logger.error(
            'Error parsing case file {:s} with {:s} format parser.'.format(
                system.files.fullname, input_format))
        return False

    # Try parsing the addfile
    if system.files.addfile:
        if not system.files.add_format:
            logger.error('Unknown addfile format.')
            return
        logger.info(f'Parsing additional file {system.files.addfile}')
        if not addparser.read_add(system, system.files.addfile):
            logger.error('Error parsing addfile {:s} with {:s} parser.'.format(system.files.addfile, input_format))
            return False

    _, s = elapsed(t)
    logger.info(f'Input file {system.files.fullname} parsed in {s}.')

    return True


def dump(system, output_format):
    """
    Dump the System data into the requested output format.

    Parameters
    ----------
    system
        System object
    output_format : str
        Output format name. 'xlsx' will be used if is not an instance of `str`.
    """
    if system.files.no_output:
        return

    if not isinstance(output_format, str):
        output_format = 'xlsx'

    outfile = system.files.dump
    writer = importlib.import_module('.' + output_format, __name__)

    t, _ = elapsed()
    ret = writer.write(system, outfile)
    _, s = elapsed(t)
    if ret:
        logger.info(f'Converted file {system.files.dump} written in {s}.')
    else:
        logger.error('Format conversion aborted.')
