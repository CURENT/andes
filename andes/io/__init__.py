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
            logger.error('Input format is not specified and cannot be inferred.')
            return False

    # try parsing the base case file
    logger.info(f'Parsing input file "{system.files.case}"')
    input_format = system.files.input_format
    parser = importlib.import_module('.' + input_format, __name__)
    if not parser.read(system, system.files.case):
        logger.error(f'Error parsing case file {system.files.fullname} with {input_format} format parser.')
        return False

    _, s = elapsed(t)
    logger.info(f'Input file parsed in {s}.')

    # Try parsing the addfile
    t, _ = elapsed()

    if system.files.addfile:
        logger.info(f'Parsing additional file "{system.files.addfile}"')
        add_format = system.files.add_format
        add_parser = importlib.import_module('.' + add_format, __name__)
        if not add_parser.read_add(system, system.files.addfile):
            logger.error(f'Error parsing addfile {system.files.addfile} with {input_format} parser.')
            return False
        _, s = elapsed(t)
        logger.info(f'Addfile parsed in {s}.')

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
    if system.files.no_output:
        logger.info('no_output is True. Case dump not processed.')
        return False

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
        logger.info(f'Format conversion completed in {s}.')
        return True
    else:
        logger.error('Format conversion failed.')
        return False
