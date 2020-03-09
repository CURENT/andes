"""
Excel reader and writer for ANDES power system parameters

This module utilizes xlsxwriter and pandas.Frame. While I like the simplicity of the dome format, spreadsheet
data is easier to read and edit.
"""
import os
import logging
import warnings
from andes.shared import pd

logger = logging.getLogger(__name__)


def testlines(fid):
    # hard coded yet
    return True


def write(system, outfile, skip_empty=True, overwrite=None, add_book=None):
    """
    Write loaded ANDES system data into an xlsx file

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
    add_book : str, optional
        An optional model to be added to the output spreadsheet

    Returns
    -------
    bool
        True if file written; False otherwise
    """
    if os.path.isfile(outfile):
        if overwrite is None:
            choice = input(f'xlsx file {outfile} already exist. Overwrite? [y/N]').lower()
            if len(choice) == 0 or choice[0] != 'y':
                logger.info('No ANDES xlsx file overwritten.')
                return False
        elif overwrite is False:
            return False

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        instance.cache.df_in.to_excel(writer, sheet_name=name, freeze_panes=(1, 0))

    if add_book is not None:
        if ',' in add_book:
            add_book = add_book.split(',')
        else:
            add_book = [add_book]

        for item in add_book:
            if item in system.models:
                system.models[item].cache.df_in.to_excel(writer, sheet_name=item, freeze_panes=(1, 0))
                logger.info(f'<{item}> template sheet added.')
            else:
                logger.error(f'<{item}> is not a valid model name.')

    writer.save()
    logger.info(f'xlsx file written to <{outfile}>')
    return True


def read(system, infile):
    """
    Read an xlsx file with ANDES model data into an empty system

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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        df_models = pd.read_excel(infile, sheet_name=None, index_col=0)

    for name, df in df_models.items():
        for row in df.to_dict(orient='record'):
            system.add(name, row)

    return system
