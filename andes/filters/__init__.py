import importlib
import os

from ..utils import elapsed
import logging

logger = logging.getLogger(__name__)
#
# Input formats is a dictionary of supported format names and the accepted
#   file extensions
#
# The first file will be parsed by read() function and the addfile will be
#   parsed by readadd()
#
# Typically, column based formats, such as IEEE CDF and PSS/E RAW,
#   are faster to parse
#
input_formats = {
    'dome': 'dm',
    'matpower': 'm',
    'psse': ['raw', 'dyr'],
    'card': ['andc'],
}

# Output formats is a dictionary of supported output formats and their
#   extensions
# The static data will be written by write() function and the addfile by
#   writeadd()
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
        if type(val) == list:
            for item in val:
                if files.ext.strip('.').lower() == item:
                    maybe.append(key)
        else:
            if files.ext.strip('.').lower() == val:
                maybe.append(key)

    # second, guess by lines
    true_format = ''
    fid = open(files.case, 'r')
    for item in maybe:
        try:
            parser = importlib.import_module('.' + item, __name__)
            testlines = getattr(parser, 'testlines')
            if testlines(fid):
                true_format = item
                break
        except ImportError:
            logger.debug(
                'Parser for {:s} format is not found. '
                'Format guess will continue.'.
                format(item))
    fid.close()

    if true_format:
        logger.debug('Input format guessed as {:s}.'.format(true_format))
    else:
        logger.error('Unable to determine case format.')

    files.input_format = true_format

    # guess addfile format
    if files.addfile:
        _, add_ext = os.path.splitext(files.addfile)
        for key, val in input_formats.items():
            if type(val) == list:
                if add_ext[1:] in val:
                    files.add_format = key
            else:
                if add_ext[1:] == val:
                    files.add_format = key

    return true_format


def parse(system):
    """
    Parse input file with the given format in system.files.input_format
    """

    t, _ = elapsed()

    input_format = system.files.input_format
    add_format = system.files.add_format
    # exit when no input format is given
    if not input_format:
        logger.error(
            'No input format found. Specify or guess a format before parsing.')
        return False

    # exit if the format parser could not be imported
    try:
        parser = importlib.import_module('.' + input_format, __name__)
        dmparser = importlib.import_module('.' + 'dome', __name__)
        if add_format:
            addparser = importlib.import_module('.' + add_format, __name__)
    except ImportError:
        logger.error(
            'Parser for {:s} format not found. Program will exit.'.format(
                input_format))
        return False

    # try parsing the base case file
    logger.info('Parsing input file <{:s}>'.format(system.files.fullname))

    if not parser.read(system.files.case, system):
        logger.error(
            'Error parsing case file {:s} with {:s} format parser.'.format(
                system.files.fullname, input_format))
        return False

    # Try parsing the addfile
    if system.files.addfile:
        if not system.files.add_format:
            logger.error('Unknown addfile format.')
            return
        logger.info('Parsing additional file {:s}.'.format(
            system.files.addfile))
        if not addparser.readadd(system.files.addfile, system):
            logger.error(
                'Error parsing addfile {:s} with {:s} format parser.'.format(
                    system.files.addfile, input_format))
            return False

    # Try parsing the dynfile with dm filter
    if system.files.dynfile:
        logger.info('Parsing input file {:s}.'.format(
            system.files.dynfile))
        if not dmparser.read(system.files.dynfile, system):
            logger.error(
                'Error parsing dynfile {:s} with dm format parser.'.format(
                    system.files.dynfile))
            return False

    _, s = elapsed(t)
    logger.debug('Case file {:s} parsed in {:s}.'.format(
        system.files.fullname, s))

    return True


def dump_raw(system):
    t, _ = elapsed()

    outfile = system.files.dump_raw
    dmparser = importlib.import_module('.' + 'dome', __name__)

    ret = dmparser.write(outfile, system)

    _, s = elapsed(t)
    if ret:
        logger.info('Raw file dump {:s} written in {:s}.'.format(
            system.files.dump_raw, s))
    else:
        logger.error('Dump raw file failed.')
