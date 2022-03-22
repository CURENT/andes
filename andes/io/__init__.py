"""
ANDES input parsers and output formatters.
"""

import importlib
import io
import logging
import os
import chardet

from typing import Union

from andes.utils.misc import elapsed
from andes.io import xlsx, psse, json, matpower   # NOQA


logger = logging.getLogger(__name__)

# Input formats is a dictionary of supported format names and the accepted file extensions
# The first file will be parsed by read() function and the addfile will be parsed by read_add()
# Typically, column based formats, such as IEEE CDF and PSS/E RAW, are faster to parse

input_formats = {
    'xlsx': ('xlsx',),
    'json': ('json',),
    'matpower': ('m', ),
    'psse': ('raw', 'dyr'),
}

# Output formats is a dictionary of supported output formats and their extensions
# The static data will be written by write() function and the addfile by writeadd()

output_formats = {
    'xlsx': ('xlsx',),
    'json': ('json',),
}


def get_output_ext(out_format):
    """
    Helper function to get the output extension for the given output format.

    Parameters
    ----------
    out_format : str
        Output format name.

    Returns
    -------
    str : file extension without dot or empty if not supported
    """

    if (out_format is None) or (out_format is True):
        logger.warning('Dump to XLSX format by default')
        return 'xlsx'
    if out_format in output_formats:
        return output_formats[out_format][0]
    else:
        logger.error(f"Dump format <{out_format}> not supported.")
        return ''


def guess(system):
    """
    Guess the input format based on extension and content.

    Also stores the format name to `system.files.input_format`.

    Parameters
    ----------
    system : System
        System instance with the file name set to `system.files`

    Returns
    -------
    str
        format name
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

    for item in maybe:
        parser = importlib.import_module('.' + item, __name__)
        testlines = getattr(parser, 'testlines')
        if testlines(files.case):
            true_format = item
            files.input_format = true_format
            logger.debug('Input format guessed as %s.', true_format)
            break

    if not true_format:
        logger.error('Unable to determine case format.')

    # guess addfile format
    if files.addfile:
        _, add_ext = os.path.splitext(files.addfile)
        for key, val in input_formats.items():
            if add_ext[1:] in val:
                files.add_format = key
                logger.debug('Addfile format guessed as %s.', key)
                break

    return true_format


def parse(system):
    """
    Parse input file with the given format in `system.files.input_format`.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """

    t, _ = elapsed()

    # exit when no input format is given
    if not system.files.input_format:
        if not guess(system):
            logger.error('Input format unknown for file "%s".', system.files.case)
            return False

    # try parsing the base case file
    logger.info('Parsing input file "%s"...', system.files.case)
    input_format = system.files.input_format
    parser = importlib.import_module('.' + input_format, __name__)
    if not parser.read(system, system.files.case):
        logger.error('Error parsing file "%s" with <%s> parser.', system.files.fullname, input_format)
        return False

    _, s = elapsed(t)
    logger.info('Input file parsed in %s.', s)

    # Try parsing the addfile
    t, _ = elapsed()

    if system.files.addfile:
        logger.info('Parsing additional file "%s"...', system.files.addfile)
        add_format = system.files.add_format
        add_parser = importlib.import_module('.' + add_format, __name__)
        if not add_parser.read_add(system, system.files.addfile):
            logger.error('Error parsing addfile "%s" with %s parser.', system.files.addfile, input_format)
            return False
        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    return True


def dump(system, output_format, full_path=None, overwrite=False, **kwargs):
    """
    Dump the System data into the requested output format.

    Parameters
    ----------
    system
        System object
    output_format : str
        Output format name. 'xlsx' will be used if is not an instance of `str`.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """
    if (output_format is None) or (output_format is True):
        output_format = 'xlsx'

    output_ext = get_output_ext(output_format)
    if output_ext == '':
        return False

    if full_path is not None:
        system.files.dump = full_path
    else:
        system.files.dump = os.path.join(system.files.output_path,
                                         system.files.name + '.' + output_ext)

    writer = importlib.import_module('.' + output_format, __name__)

    t, _ = elapsed()
    ret = writer.write(system, system.files.dump, overwrite=overwrite, **kwargs)
    _, s = elapsed(t)
    if ret:
        logger.info('Format conversion completed in %s.', s)
        return True
    else:
        logger.error('Format conversion failed.')
        return False


def read_file_like(infile: Union[str, io.IOBase]):
    """
    Read a file-like object and return a list of splitted lines.
    """

    if isinstance(infile, str):
        with open(infile, 'rb') as fb:
            charset = chardet.detect(fb.read())
            logger.debug("Detected raw file encoding: %s", charset)

        f = open(infile, 'r', encoding=charset['encoding'])
    else:
        f = infile

    in_data = f.read()
    lines_list = in_data.splitlines()

    if f is not infile:
        f.close()

    return lines_list
