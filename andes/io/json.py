"""
JSON reader and writer for ANDES.

"""
import io
import json
import logging

from collections import OrderedDict
from typing import Union

from andes.utils.paths import confirm_overwrite

logger = logging.getLogger(__name__)


def testlines(infile):
    return True


def write(system, outfile, skip_empty=True, overwrite=None, **kwargs):
    """
    Write loaded ANDES system data into a JSON file.

    Parameters
    ----------
    system : System
        A loaded system with parameters
    outfile : str
        Path to the output file
    skip_empty : bool
        Skip output of empty models (n = 0)
    overwrite : bool
        None to prompt for overwrite selection; True to overwrite; False to not overwrite

    Returns
    -------
    bool
        True if file written; False otherwise
    """
    if not confirm_overwrite(outfile, overwrite):
        return False

    if hasattr(outfile, 'write'):
        outfile.write(_dump_system(system, skip_empty))
    else:
        with open(outfile, 'w') as writer:
            writer.write(_dump_system(system, skip_empty))
            logger.info('JSON file written to "%s"', outfile)

    return True


def _dump_system(system, skip_empty, orient='records'):
    """
    Dump parameters of each model into a json string and return
    them all in an OrderedDict.
    """

    out = OrderedDict()
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        out[name] = instance.cache.df_in.to_dict(orient=orient)

    return json.dumps(out, indent=2)


def read(system, infile: Union[str, io.IOBase]):
    """
    Read JSON file with ANDES model data into an empty system.

    Parameters
    ----------
    system : System
        Empty System instance
    infile : str or io.BaseIO
        str: path to the input file; or io.BaseIO: a stream to
        read from

    Returns
    -------
    System
        System instance after succeeded
    """
    if isinstance(infile, str):
        f = open(infile, 'r')
    else:
        f = infile

    json_in = json.load(f)

    if f is not infile:
        f.close()

    for name, dct in json_in.items():
        for row in dct:
            system.add(name, row)

    return system
