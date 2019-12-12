"""Utility functions for loading andes stock test cases"""
import os


def stock_case_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir_name, '..', '..', 'cases')


def get_stock_case(rpath):
    """Return the path to the stock cases"""
    case_root = stock_case_root()
    case_path = os.path.join(case_root, rpath)

    if not os.path.isfile(case_path):
        raise FileNotFoundError("Relative path {} is not valid for stock case".format(rpath))

    return case_path
