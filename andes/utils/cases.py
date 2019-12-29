"""
Utility functions for loading andes stock test cases
"""
import os


def cases_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir_name, '..', '..', 'cases')


def stock(rpath):
    """Return the path to the stock cases"""
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    if not os.path.isfile(case_path):
        raise FileNotFoundError("Relative path {} is not valid for stock case".format(rpath))

    return case_path
