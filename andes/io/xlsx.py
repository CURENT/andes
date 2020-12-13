"""
Excel reader and writer for ANDES power system parameters

This module utilizes openpyxl, xlsxwriter and pandas.Frame.

While I like the simplicity of the dome format,
spreadsheets are easier to view and edit.
"""

import logging

from andes.utils.paths import confirm_overwrite
from andes.shared import pd

logger = logging.getLogger(__name__)


def testlines(infile):
    return True


def write(system, outfile, skip_empty=True, overwrite=None, add_book=None, **kwargs):
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
    overwrite : bool, optional
        None to prompt for overwrite selection; True to overwrite; False to not overwrite
    add_book : str, optional
        An optional model to be added to the output spreadsheet

    Returns
    -------
    bool
        True if file written; False otherwise
    """
    if not confirm_overwrite(outfile, overwrite=overwrite):
        return False

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    writer = _write_system(system, writer, skip_empty)
    writer = _add_book(system, writer, add_book)

    writer.save()
    logger.info(f'xlsx file written to "{outfile}"')
    return True


def _write_system(system, writer, skip_empty):
    """
    Write the system to pandas ExcelWriter
    """
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        instance.cache.refresh("df_in")
        instance.cache.df_in.to_excel(writer, sheet_name=name, freeze_panes=(1, 0))
    return writer


def _add_book(system, writer, add_book):
    """
    Add workbook to an existing pandas ExcelWriter
    """
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
    return writer


def read(system, infile):
    """
    Read an xlsx file with ANDES model data into an empty system

    Parameters
    ----------
    system : System
        Empty System instance
    infile : str or file-like
        Path to the input file, or a file-like object

    Returns
    -------
    System
        System instance after succeeded
    """
    df_models = pd.read_excel(infile,
                              sheet_name=None,
                              index_col=0,
                              engine='openpyxl',
                              )

    for name, df in df_models.items():
        # drop rows that all nan
        df.dropna(axis=0, how='all', inplace=True)
        for row in df.to_dict(orient='records'):
            system.add(name, row)

    # --- for debugging ---
    system.df_in = df_models

    return system
