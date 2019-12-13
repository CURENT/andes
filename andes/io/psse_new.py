from andes.io.em_psse import parse_raw

import logging
logger = logging.getLogger(__name__)

raw2andes = {}


def is_format(fid):
    """
    Check the raw file for frequency base
    """
    first = fid.readline()
    first = first.strip().split('/')
    first = first[0].split(',')
    if float(first[5]) == 50.0 or float(first[5]) == 60.0:
        return True
    else:
        return False


def read(system, file_name):
    raw_data = parse_raw(file_name)
    for psse_model in raw_data:
        andes_model = raw2andes.get_model(psse_model)  # NOQA
        df = raw_data[psse_model]['df']
        for psse_param in df:
            pass

    return raw_data
