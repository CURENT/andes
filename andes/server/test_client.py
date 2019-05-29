import requests
import logging
import time  # NOQA
import os

logger = logging.getLogger(__name__)


def load_case(server_url, case):
    """
    Use the REST API to load a data in andes server

    Parameters
    ----------
    case

    Returns
    -------

    """
    url = os.path.join(server_url, 'load')

    params = {'name': case}
    r = requests.get(url=url, params=params)
    if not r.ok:
        logger.error('load <{}> error. file may not exist in server root.'.format(case))

    return r


def run_case(server_url, simulation_time):
    """
    Run a loaded system
    Parameters
    ----------
    simulation_time

    Returns
    -------

    """
    if simulation_time <= 0:
        logger.error('Simulation time must be greater than 0')
        return

    url = os.path.join(server_url, 'run')
    params = {'time': simulation_time}

    r = requests.get(url=url, params=params)

    if not r.ok:
        logger.error('Run simulation error. System may not have been loaded.')

    return r


def param_get(server_url, model, var_name=None, idx=None, base='sysbase'):
    """
    Get parameter from the loaded system

    Parameters
    ----------
    server_url
    model
    var_name
    idx
    base

    Returns
    -------

    """
    pass


if __name__ == '__main__':
    local_server = 'http://127.0.0.1:5000'
    r = load_case(local_server, 'ieee14_syn.dm')
    if r.ok:
        data = r.json()

    r = run_case(local_server, 50)
