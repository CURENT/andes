"""
JSON reader and writer for ANDES.

"""
import json
import logging
from collections import OrderedDict
from andes.utils.paths import confirm_overwrite

logger = logging.getLogger(__name__)


def testlines(fid):
    return True


def write(system, outfile, skip_empty=True, overwrite=None, **kwargs):
    """
    Write loaded ANDES system data into a JSON file

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

    with open(outfile, 'w') as writer:
        writer.write(_dump_system(system, writer, skip_empty))
        logger.info(f'JSON file written to "{outfile}"')

    return True


def _dump_system(system, writer, skip_empty, orient='records'):
    """
    Write system to JSON output.
    """
    out = OrderedDict()
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        out[name] = instance.cache.df_in.to_dict(orient=orient)

    return json.dumps(out, indent=2)


def read(system, infile):
    """
    Read JSON file with ANDES model data into an empty system.

    Parameters
    ----------
    system : System
        Empty System instance
    infile : str
        Path to the input file

    Returns
    -------
    System
        System instance after succeeded
    """
    with open(infile, 'r') as f:
        json_in = json.load(f)

    for name, dct in json_in.items():
        for row in dct:
            system.add(name, row)

    return system
