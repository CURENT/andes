import importlib
import os

# input formats is a dictionary of supported format names and the accepted file extensions
#   The first file will be parsed by read() function and the addfile will be parsed by readadd()
#   Typically, column based formats, such as IEEE CDF and PSS/E RAW, are faster to parse
input_formats = {'dome': 'dm',
                 'matpower': 'm',
                 'psse': ['raw', 'dyr'],
                 'card': ['andc'],
                 }

# output formats is a dictionary of supported output formats and their extensions
#   The static data will be written by write() function and the addfile by writeadd()
output_formats = ['']


def guess(system):
    """input format guess function. First guess by extension, then test by lines"""
    files = system.Files
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
    true_format = None
    fid = open(files.case, 'r')
    for item in maybe:
        try:
            parser = importlib.import_module('.' + item, __name__)
            testlines = getattr(parser, 'testlines')
            if testlines(fid):
                true_format = item
                break
        except ImportError:
            system.Log.debug('Parser for {:s} format is not found. Format guess will continue.'.format(item))
    fid.close()

    if true_format:
        system.Log.debug('Input format guessed as {:s}.'.format(true_format))

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
    """Parse input file with the given format in system.Files.input_format"""
    input_format = system.Files.input_format
    add_format = system.Files.add_format
    # exit when no input format is given
    if not input_format:
        system.Log.error('No input format found. Specify or guess a format before parsing.')
        return False

    # exit if the format parser could not be imported
    try:
        parser = importlib.import_module('.' + input_format, __name__)
        dmparser = importlib.import_module('.' + 'dome', __name__)
        if add_format:
            addparser = importlib.import_module('.' + add_format, __name__)
    except ImportError:
        system.Log.error('Parser for {:s} format not found. Program will exit.'.format(input_format))
        return False

    # try parsing the base case file
    system.Log.info('Parsing input file {:s}.'.format(system.Files.fullname))

    if not parser.read(system.Files.case, system):
        system.Log.error('Error parsing case file {:s} with {:s} format parser.'.format(system.Files.fullname,
                                                                                        input_format))
        return False

    # Try parsing the addfile
    if system.Files.addfile:
        if not system.Files.add_format:
            system.Log.error('Unknown addfile format.')
            return
        system.Log.info('Parsing additional file {:s}.'.format(system.Files.addfile))
        if not addparser.readadd(system.Files.addfile, system):
            system.Log.error('Error parsing addfile {:s} with {:s} format parser.'.format(system.Files.addfile,
                                                                                          input_format))
            return False

    # Try parsing the dynfile with dm filter
    if system.Files.dynfile:
        system.Log.info('Parsing input file {:s}.'.format(system.Files.dynfile))
        if not dmparser.read(system.Files.dynfile, system):
            system.Log.error('Error parsing dynfile {:s} with dm format parser.'.format(system.Files.dynfile))
            return False

    return True


def dump_raw(system):
    # output_format = system.Files.output_format
    # if output_format.lower() not in output_formats:
    #     system.Log.warning('Dump output format \'{:s}\'not recognized'.format(output_format))
    #     return False
    outfile = system.Files.dump_raw
    dmparser = importlib.import_module('.' + 'dome', __name__)
    return dmparser.write(outfile, system)
